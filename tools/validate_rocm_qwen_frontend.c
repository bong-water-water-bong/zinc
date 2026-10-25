// GPU-node-only validation; no model file is loaded. From the repository root:
// cc -O2 -std=c11 -D__HIP_PLATFORM_AMD__=1 -Isrc/cuda -I/opt/rocm/include \
//   tools/validate_rocm_qwen_frontend.c src/rocm/rocm_shim.c \
//   -L/opt/rocm/lib -Wl,-rpath,/opt/rocm/lib -lamdhip64 -lhiprtc -lhipblas -lm \
//   -o /tmp/validate_rocm_qwen_frontend
// ROCR_VISIBLE_DEVICES=<device> /tmp/validate_rocm_qwen_frontend
// Compares every output bit with the six original GPU launches and checks
// indexing/math independently on the CPU, including untouched KV guard rows.
#include "cuda_shim.h"
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef struct { unsigned N; float eps; } RmsPush;
typedef struct { unsigned hd, nh, T; } DeintPush;
typedef struct { unsigned stride, rd, nh, pos, freq, scale; } RopePush;
typedef struct { unsigned kd, pos, T; } KvPush;
typedef struct { unsigned hd; float eps; unsigned rd, nh, nk, pos, off; } QkvPush;
static uint32_t rng = 0x81a49b23;
static float random_value(void) {
    rng ^= rng << 13; rng ^= rng >> 17; rng ^= rng << 5;
    return ((int)(rng % 8193) - 4096) / 1024.0f;
}
static void fail(const char *message) {
    fprintf(stderr, "FAIL: %s (%s)\n", message, cuda_last_error());
    exit(1);
}
static float *values(size_t n) {
    float *p = malloc(n * sizeof *p);
    if (!p) fail("host allocation");
    return p;
}
static CudaBuf *device(CudaCtx *ctx, size_t n, const float *data) {
    CudaBuf *b = cuda_create_buffer(ctx, n * sizeof(float));
    if (!b) fail("device allocation");
    if (data) cuda_upload(ctx, b, data, n * sizeof(float));
    return b;
}
static void dispatch(CudaCmd *cmd, CudaPipe *pipe, unsigned gx, unsigned gy,
                     unsigned block, CudaBuf **bufs, unsigned count,
                     const void *push, size_t size, unsigned shared) {
    unsigned grid[] = { gx, gy, 1 }, threads[] = { block, 1, 1 };
    cuda_dispatch(cmd, pipe, grid, threads, bufs, count, push, size, shared);
}
static void compare(CudaCtx *ctx, CudaBuf *a, CudaBuf *b, size_t n, const char *label) {
    float *x = values(n), *y = values(n);
    cuda_download(ctx, a, x, n * sizeof(float));
    cuda_download(ctx, b, y, n * sizeof(float));
    if (memcmp(x, y, n * sizeof(float))) {
        for (size_t i = 0; i < n; ++i) {
            if (memcmp(x+i, y+i, sizeof(float))) {
                fprintf(stderr, "%s[%zu]: reference=%.9g fused=%.9g\n", label, i, x[i], y[i]);
                fail("GPU output differs");
            }
        }
    }
    free(x); free(y);
}
static void close_value(float expected, float actual) {
    if (!isfinite(actual) || fabsf(expected-actual) > 0.0003f * (1.0f+fabsf(expected))) {
        fprintf(stderr, "CPU=%.9g GPU=%.9g\n", expected, actual);
        fail("CPU reference differs");
    }
}
static void cpu_head(const float *x, const float *w, const float *freq,
                     float *out, unsigned hd, unsigned rd, unsigned pos) {
    double ss = 0;
    for (unsigned i = 0; i < hd; ++i) ss += (double)x[i]*x[i];
    const float inv = (float)(1.0/sqrt(ss/hd+1e-6));
    for (unsigned i = 0; i < hd; ++i) out[i] = w[i]*(x[i]*inv);
    for (unsigned i = 0; i < rd/2; ++i) {
        float a = out[i], b = out[i+rd/2];
        float theta = (float)pos*freq[i], c = cosf(theta), s = sinf(theta);
        out[i] = a*c-b*s;
        out[i+rd/2] = a*s+b*c;
    }
}
static void test_case(CudaCtx *ctx, CudaPipe **pipes, unsigned T, unsigned hd,
                      unsigned rd, unsigned pos) {
    const unsigned nh = hd == 128 ? 40 : 8, nk = hd == 128 ? 8 : 2;
    const unsigned qd = nh*hd, kd = nk*hd, ctx_rows = pos+T+2;
    const size_t qn = (size_t)T*qd, kn = (size_t)T*kd, cn = (size_t)ctx_rows*kd;
    float *q = values(2*qn), *k = values(kn), *v = values(kn);
    float *wq = values(hd), *wk = values(hd), *freq = values(hd);
    float *guards = values(cn);
    for (size_t i = 0; i < 2*qn; ++i) q[i] = pos == 0 ? 0 : random_value();
    for (size_t i = 0; i < kn; ++i) { k[i] = random_value(); v[i] = random_value(); }
    for (unsigned i = 0; i < hd; ++i) {
        wq[i] = random_value(); wk[i] = random_value();
        freq[i] = powf(10000.0f, -(float)(2*i)/(float)hd);
    }
    for (size_t i = 0; i < cn; ++i) guards[i] = -123.25f;
    CudaBuf *buf[] = {
        device(ctx,2*qn,q), device(ctx,kn,k), device(ctx,kn,v),
        device(ctx,hd,wq), device(ctx,hd,wk), device(ctx,hd,freq),
        device(ctx,qn,NULL), device(ctx,qn,NULL), device(ctx,cn,guards), device(ctx,cn,guards),
        device(ctx,qn,NULL), device(ctx,qn,NULL), device(ctx,cn,guards), device(ctx,cn,guards),
        device(ctx,kn,k),
    };
    CudaCmd *cmd = cuda_begin_command(ctx);
    DeintPush deint = { hd, nh, T };
    dispatch(cmd,pipes[0],(qd+255)/256,T,256,(CudaBuf*[]){buf[0],buf[6],buf[7]},3,&deint,sizeof deint,0);
    RmsPush rms = { hd,1e-6f };
    dispatch(cmd,pipes[1],nh*T,1,256,(CudaBuf*[]){buf[6],buf[3],buf[6]},3,&rms,sizeof rms,0);
    dispatch(cmd,pipes[1],nk*T,1,256,(CudaBuf*[]){buf[14],buf[4],buf[14]},3,&rms,sizeof rms,0);
    RopePush rope = { hd,rd,nh,pos,0,0 };
    dispatch(cmd,pipes[2],nh,T,256,(CudaBuf*[]){buf[6],buf[6],buf[5]},3,&rope,sizeof rope,0);
    rope.nh = nk;
    dispatch(cmd,pipes[2],nk,T,256,(CudaBuf*[]){buf[14],buf[14],buf[5]},3,&rope,sizeof rope,0);
    KvPush kv = { kd,pos,T };
    dispatch(cmd,pipes[3],(kd+63)/64,T,64,(CudaBuf*[]){buf[14],buf[8],buf[2],buf[9]},4,&kv,sizeof kv,0);
    QkvPush fused = { hd,1e-6f,rd,nh,nk,pos,pos*kd };
    dispatch(cmd,pipes[4],nh+2*nk,T,256,(CudaBuf*[]){buf[0],buf[1],buf[2],buf[3],buf[4],buf[5],buf[10],buf[11],buf[12],buf[13]},10,&fused,sizeof fused,hd*sizeof(float));
    cuda_commit_and_wait(cmd);
    if (*cuda_last_error()) fail("dispatch");
    compare(ctx,buf[6],buf[10],qn,"Q");
    compare(ctx,buf[7],buf[11],qn,"gate");
    compare(ctx,buf[8],buf[12],cn,"K cache/guards");
    compare(ctx,buf[9],buf[13],cn,"V cache/guards");
    float *actual_q = values(qn), *actual_k = values(cn), *actual_v = values(cn), *head = values(hd);
    cuda_download(ctx,buf[10],actual_q,qn*sizeof(float));
    cuda_download(ctx,buf[12],actual_k,cn*sizeof(float));
    cuda_download(ctx,buf[13],actual_v,cn*sizeof(float));
    for (unsigned row = 0; row < ctx_rows; ++row) {
        if (row >= pos && row < pos+T) continue;
        for (unsigned i = 0; i < kd; ++i) {
            if (actual_k[(size_t)row*kd+i] != -123.25f || actual_v[(size_t)row*kd+i] != -123.25f)
                fail("untouched KV guard row changed");
        }
    }
    for (unsigned t = 0; t < T; ++t) {
        for (unsigned h = 0; h < nh; ++h) {
            cpu_head(q+(size_t)t*2*qd+2*h*hd,wq,freq,head,hd,rd,pos+t);
            for (unsigned i = 0; i < hd; ++i) close_value(head[i],actual_q[(size_t)t*qd+h*hd+i]);
        }
        for (unsigned h = 0; h < nk; ++h) {
            cpu_head(k+(size_t)t*kd+h*hd,wk,freq,head,hd,rd,pos+t);
            for (unsigned i = 0; i < hd; ++i) {
                close_value(head[i],actual_k[(size_t)(pos+t)*kd+h*hd+i]);
                close_value(v[(size_t)t*kd+h*hd+i],actual_v[(size_t)(pos+t)*kd+h*hd+i]);
            }
        }
    }
    free(actual_q); free(actual_k); free(actual_v); free(head);
    for (unsigned i = 0; i < sizeof buf/sizeof buf[0]; ++i) cuda_free_buffer(buf[i]);
    free(q); free(k); free(v); free(wq); free(wk); free(freq); free(guards);
}
int main(void) {
    FILE *file = fopen("src/shaders/cuda/kernels.cu","rb");
    if (!file) fail("run from repository root");
    fseek(file,0,SEEK_END); long size = ftell(file); rewind(file);
    char *source = malloc((size_t)size+1);
    if (!source || fread(source,1,(size_t)size,file) != (size_t)size) fail("read kernel source");
    source[size] = 0; fclose(file);
    CudaCtx *ctx = cuda_init(0);
    if (!ctx) fail("GPU initialization");
    const char *names[] = {"deinterleave_qgate_batched","rms_norm","rope_batched","kv_cache_write_batched","qwen_norm_rope_qkv"};
    CudaPipe *pipes[5];
    for (unsigned i = 0; i < 5; ++i) {
        pipes[i] = cuda_create_pipeline(ctx,source,names[i],NULL,0);
        if (!pipes[i]) fail(names[i]);
    }
    unsigned count = 0;
    const unsigned dimensions[] = {128,256}, positions[] = {0,97,4091};
    for (unsigned T = 1; T <= 5; ++T)
        for (unsigned d = 0; d < 2; ++d)
            for (unsigned r = 1; r <= 2; ++r)
                for (unsigned p = 0; p < 3; ++p) {
                    test_case(ctx,pipes,T,dimensions[d],dimensions[d]*r/2,positions[p]);
                    ++count;
                }
    for (unsigned i = 0; i < 5; ++i) cuda_free_pipeline(pipes[i]);
    cuda_destroy(ctx); free(source);
    printf("PASS: %u cases, GPU outputs bit-identical, independent CPU math and KV guards match\n",count);
    return 0;
}
