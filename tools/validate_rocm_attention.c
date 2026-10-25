// GPU test host only; no model is loaded.
// cc -O2 -std=c11 -D_POSIX_C_SOURCE=200809L -D__HIP_PLATFORM_AMD__=1 \
//   -Isrc/cuda -I/opt/rocm/include tools/validate_rocm_attention.c \
//   src/rocm/rocm_shim.c -L/opt/rocm/lib -Wl,-rpath,/opt/rocm/lib \
//   -lamdhip64 -lhiprtc -lhipblas -lm -o /tmp/validate-rocm-attention
// ROCR_VISIBLE_DEVICES=<device> /tmp/validate-rocm-attention
// Checks tiled decode, causal prefill, speculative offsets, slot isolation,
// graph capture/update, guards, and sticky failures against independent double
// precision CPU math. Legal short cases also compare with the old GPU kernels.
#include "cuda_shim.h"
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef struct { unsigned hd, heads, kv_heads, seq, scale, sink; } Decode;
typedef struct { unsigned hd, heads, kv_heads, T, scale, sink, base; } Causal;
typedef struct { unsigned hd, heads, kv_heads, ctx, scale, sink; } Slots;
static unsigned cases;
static uint32_t rng = 0x19ab42e1;
static float random_value(void) {
    rng ^= rng << 13; rng ^= rng >> 17; rng ^= rng << 5;
    return ((int)(rng % 2049) - 1024) / 1024.0f;
}
static void fail(const char *what) {
    fprintf(stderr, "FAIL %s: %s\n", what, cuda_last_error());
    exit(1);
}
static void *allocate(size_t n) {
    void *p = malloc(n);
    if (!p) fail("host allocation");
    return p;
}
static char *source(void) {
    FILE *f = fopen("src/shaders/cuda/kernels.cu", "rb");
    if (!f || fseek(f, 0, SEEK_END)) fail("open kernel source");
    long n = ftell(f);
    if (n < 0 || fseek(f, 0, SEEK_SET)) fail("kernel source size");
    char *s = allocate((size_t)n + 1);
    if (fread(s, 1, (size_t)n, f) != (size_t)n) fail("read kernel source");
    fclose(f); s[n] = 0; return s;
}
static CudaBuf *upload(CudaCtx *ctx, const void *p, size_t n) {
    CudaBuf *b = cuda_create_buffer(ctx, n);
    if (!b) fail("device allocation");
    cuda_upload(ctx, b, p, n);
    if (cuda_execution_failed(ctx)) fail("upload");
    return b;
}
static void close_value(double expected, float actual) {
    if (!isfinite(actual) || fabs(expected - actual) > 0.0003 * (1 + fabs(expected))) {
        fprintf(stderr, "expected=%.12g actual=%.9g\n", expected, actual);
        fail("numerical reference");
    }
}
static void cpu_row(const float *q, const float *k, const float *v, const float *out,
                    unsigned hd, unsigned heads, unsigned kv_heads, unsigned seq,
                    double scale, const float *sinks) {
    double *scores = allocate((size_t)seq * sizeof(double));
    for (unsigned h = 0; h < heads; h++) {
        unsigned kh = h / (heads / kv_heads);
        double m = -INFINITY;
        for (unsigned i = 0; i < seq; i++) {
            double dot = 0;
            for (unsigned d = 0; d < hd; d++)
                dot += (double)q[(size_t)h * hd + d] * k[((size_t)i * kv_heads + kh) * hd + d];
            scores[i] = dot * scale;
            m = fmax(m, scores[i]);
        }
        if (!isnan(sinks[h])) m = fmax(m, sinks[h]);
        double sum = isnan(sinks[h]) ? 0 : exp(sinks[h] - m);
        for (unsigned i = 0; i < seq; i++) { scores[i] = exp(scores[i] - m); sum += scores[i]; }
        for (unsigned d = 0; d < hd; d++) {
            double acc = 0;
            for (unsigned i = 0; i < seq; i++)
                acc += scores[i] * v[((size_t)i * kv_heads + kh) * hd + d];
            close_value(acc / sum, out[(size_t)h * hd + d]);
        }
    }
    free(scores);
}
// mode: 0 decode, 1 causal prefill, 2 independently positioned/permuted slots.
static void test(CudaCtx *ctx, CudaPipe **pipes, unsigned seq, unsigned hd,
                 unsigned T, unsigned mode, unsigned stress, unsigned graph) {
    const unsigned heads = 4, kv_heads = 2, ns = mode == 2 ? 3 : 1;
    const unsigned sink_offset = 3, stride = hd * kv_heads;
    unsigned positions[3] = {seq - 1, 0, seq > 1025 ? 1024 : seq / 2};
    unsigned slots[3] = {2, 0, 1};
    unsigned rows = mode == 0 ? 1 : mode == 1 ? T : ns;
    size_t qn = (size_t)rows * heads * hd, kn = (size_t)ns * seq * stride;
    float *q = allocate(qn * 4), *k = allocate(kn * 4), *v = allocate(kn * 4);
    float sinks[7], *out = allocate((qn + 8) * 4), *reference = allocate((qn + 8) * 4);
    for (size_t i = 0; i < qn; i++) q[i] = random_value() * (stress ? 40 : 1);
    for (size_t i = 0; i < kn; i++) { k[i] = random_value(); v[i] = random_value(); }
    for (unsigned i = 0; i < 7; i++) sinks[i] = i % 2 ? NAN : (stress ? 800.0f : -0.25f);
    for (size_t i = 0; i < qn + 8; i++) out[i] = reference[i] = -999.25f;
    CudaBuf *bufs[] = {upload(ctx,q,qn*4), upload(ctx,k,kn*4), upload(ctx,v,kn*4),
        upload(ctx,sinks,sizeof sinks), upload(ctx,out,(qn+8)*4),
        upload(ctx,positions,sizeof positions), upload(ctx,slots,sizeof slots)};
    Decode dec = {hd,heads,kv_heads,seq,0,sink_offset};
    Causal causal = {hd,heads,kv_heads,T,0,sink_offset,seq-T};
    Slots slot = {hd,heads,kv_heads,seq,0,sink_offset};
    float custom_scale = 0.375f;
    if (stress) { memcpy(&dec.scale,&custom_scale,4); causal.scale=slot.scale=dec.scale; }
    const void *push = mode == 0 ? (void*)&dec : mode == 1 ? (void*)&causal : (void*)&slot;
    size_t push_size = mode == 1 ? sizeof causal : sizeof dec;
    unsigned grid[] = {heads,rows,(hd+255)/256}, block[] = {256,1,1};
    CudaGraph *g = graph ? cuda_graph_create() : NULL;
    unsigned launches = graph ? 2 : 1;
    for (unsigned launch = 0; launch < launches; launch++) {
        if (launch > 0) {
            // Verify changed graph parameters, not just an identical replay.
            custom_scale = 0.625f;
            memcpy(&dec.scale,&custom_scale,4);
            causal.scale=slot.scale=dec.scale;
        }
        if (graph && !cuda_graph_begin(ctx)) fail("graph begin");
        CudaCmd *cmd = cuda_begin_command(ctx);
        if (!cmd) fail("command begin");
        cuda_dispatch(cmd,pipes[mode],grid,block,bufs,mode == 2 ? 7 : 5,push,push_size,0);
        if (graph) {
            cuda_release_completed(cmd);
            if (!cuda_graph_end_launch(ctx,g)) fail("graph launch/update");
        } else cuda_commit_and_wait(cmd);
        if (cuda_execution_failed(ctx)) fail("tiled launch");
    }
    cuda_download(ctx,bufs[4],out,(qn+8)*4);
    if (cuda_execution_failed(ctx)) fail("download");
    double scale = dec.scale ? custom_scale : 1 / sqrt(hd);
    for (unsigned row = 0; row < rows; row++) {
        // Check every decode/slot row; sample early/middle/last causal queries.
        if (mode == 1 && row != 0 && row != rows/2 && row != rows-1) continue;
        unsigned length = mode == 0 ? seq : mode == 1 ? causal.base + row + 1 : positions[row] + 1;
        size_t off = mode == 2 ? (size_t)slots[row] * seq * stride : 0;
        cpu_row(q+(size_t)row*heads*hd,k+off,v+off,out+(size_t)row*heads*hd,
                hd,heads,kv_heads,length,scale,sinks+sink_offset);
    }
    for (unsigned i = 0; i < 8; i++) if (out[qn+i] != -999.25f) fail("output guard");
    if (seq <= 8192) {
        unsigned old_grid[] = {heads,rows,1};
        unsigned shared = (seq + (mode == 1 ? hd : 0)) * 4;
        CudaCmd *cmd = cuda_begin_command(ctx);
        cuda_dispatch(cmd,pipes[mode+3],old_grid,block,bufs,mode == 2 ? 7 : 5,push,push_size,shared);
        cuda_commit_and_wait(cmd);
        cuda_download(ctx,bufs[4],reference,(qn+8)*4);
        if (cuda_execution_failed(ctx)) fail("original kernel");
        for (size_t i = 0; i < qn; i++) close_value(reference[i],out[i]);
    }
    if (g) cuda_graph_free(g);
    for (unsigned i = 0; i < 7; i++) cuda_free_buffer(bufs[i]);
    free(q); free(k); free(v); free(out); free(reference);
    printf("PASS mode=%u seq=%u hd=%u rows=%u stress=%u graph=%u\n",mode,seq,hd,rows,stress,graph);
    fflush(stdout); cases++;
}
static void failures(void) {
    const char *s = "extern \"C\" __global__ void store(float* x) { x[0]=1.0f; }";
    for (unsigned mode = 0; mode < 2; mode++) {
        CudaCtx *ctx = cuda_init(0);
        if (!ctx) fail("failure-test init");
        CudaPipe *p = cuda_create_pipeline(ctx,s,"store",NULL,0);
        if (!p) fail("failure-test kernel compile");
        float initial = 0;
        CudaBuf *b = upload(ctx,&initial,4);
        unsigned grid[] = {1,1,1}, block[] = {mode ? cuda_pipeline_max_threads(p)+1 : 1,1,1};
        CudaCmd *cmd = cuda_begin_command(ctx);
        cuda_dispatch(cmd,p,grid,block,&b,1,NULL,0,mode ? 0 : cuda_max_shared_mem_per_block(ctx)+4);
        cuda_commit_and_wait(cmd);
        if (!cuda_execution_failed(ctx) || cuda_begin_command(ctx) != NULL) fail("execution failure not sticky");
        cuda_free_buffer(b); cuda_free_pipeline(p); cuda_destroy(ctx);
        printf("PASS sticky %s failure\n",mode ? "launch" : "shared-memory");
        cases++;
    }
}
int main(void) {
    char *s = source();
    CudaCtx *ctx = cuda_init(0);
    if (!ctx) fail("GPU init");
    const char *names[] = {"attention_tiled","attention_causal_tiled","attention_batched_seq_tiled",
        "naive_attention","attention_causal_batched_v2","naive_attention_batched_seq"};
    CudaPipe *pipes[6];
    for (unsigned i = 0; i < 6; i++) {
        pipes[i] = cuda_create_pipeline(ctx,s,names[i],NULL,0);
        if (!pipes[i]) fail("kernel compile");
        if (i < 3 && cuda_pipeline_shared_mem(pipes[i]) > 8192) fail("unbounded shared memory");
        if (i < 3) printf("%s: %u static shared bytes\n", names[i], cuda_pipeline_shared_mem(pipes[i]));
    }
    const unsigned lengths[] = {1,1023,1024,1025,4097,8192,8193,16080,16384,16385,32768};
    for (unsigned i = 0; i < sizeof lengths/sizeof lengths[0]; i++)
        test(ctx,pipes,lengths[i],256,1,0,0,0);
    for (unsigned mode = 0; mode < 3; mode++) {
        test(ctx,pipes,8191,128,3,mode,1,0);
        test(ctx,pipes,32768,513,4,mode,0,1);
        test(ctx,pipes,16385,64,3,mode,1,0);
    }
    test(ctx,pipes,1025,128,1025,1,0,0);
    test(ctx,pipes,16080,256,16080,1,0,0);
    test(ctx,pipes,32768,128,4096,1,0,0);
    for (unsigned i = 0; i < 6; i++) cuda_free_pipeline(pipes[i]);
    cuda_destroy(ctx); free(s);
    failures();
    printf("PASS %u attention/error cases\n",cases);
    return 0;
}
