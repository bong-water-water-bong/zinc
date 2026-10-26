// Run only on a GPU test host. No model is loaded.
// git show 70c840bd:src/shaders/cuda/kernels.cu > /tmp/zinc-ssm-reference.cu
// cc -O2 -std=c11 -D__HIP_PLATFORM_AMD__=1 -Isrc/cuda -I/opt/rocm/include \
//   tools/validate_rocm_ssm_warp.c src/rocm/rocm_shim.c \
//   -L/opt/rocm/lib -Wl,-rpath,/opt/rocm/lib -lamdhip64 -lhiprtc -lhipblas -lm \
//   -o /tmp/validate-rocm-ssm
// ROCR_VISIBLE_DEVICES=<device> /tmp/validate-rocm-ssm /tmp/zinc-ssm-reference.cu
// Compares one tiled scan with the original kernel (full sequences through
// 8192 tokens, then legal 4096-token chunks), checks all output/state bits
// and guards, and checks selected state
// rows/output independently on the CPU over the full sequence.
#include "cuda_shim.h"
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef struct {
    unsigned heads, hv, ds, groups, a_half, bias_half, has_bias, has_a;
    unsigned T, conv_stride, ab_stride, out_stride;
} Push;
static uint32_t rng = 0x5319ac71;
static float random_float(void) {
    rng ^= rng << 13; rng ^= rng >> 17; rng ^= rng << 5;
    return ((int)(rng % 2049) - 1024) / 2048.0f;
}
static void fail(const char *what) {
    fprintf(stderr, "FAIL: %s (%s)\n", what, cuda_last_error());
    exit(1);
}
static char *source(const char *name) {
    FILE *f = fopen(name, "rb");
    if (!f || fseek(f, 0, SEEK_END)) fail("open source");
    long n = ftell(f);
    if (n < 0 || fseek(f, 0, SEEK_SET)) fail("source size");
    char *s = malloc((size_t)n + 1);
    if (!s || fread(s, 1, (size_t)n, f) != (size_t)n) fail("read source");
    fclose(f); s[n] = 0; return s;
}
static float *values(size_t n) {
    float *p = malloc(n * sizeof(float));
    if (!p) fail("host allocation");
    return p;
}
static CudaBuf *buffer(CudaCtx *ctx, const void *data, size_t bytes) {
    CudaBuf *p = cuda_create_buffer(ctx, bytes);
    if (!p) fail("device allocation");
    if (data) cuda_upload(ctx, p, data, bytes);
    return p;
}
static void close_value(float expected, float actual) {
    if (!isfinite(actual) || fabsf(expected - actual) > 0.0002f * (1 + fabsf(expected))) {
        fprintf(stderr, "CPU=%.9g GPU=%.9g\n", expected, actual);
        fail("independent CPU reference");
    }
}
static void cpu_rows(Push p, const float *conv, const float *alpha, const float *beta,
                     const float *initial, const float *actual, const float *output) {
    for (unsigned h = 0; h < p.heads; h++) for (unsigned which = 0; which < 2; which++) {
        unsigned row = which ? p.hv - 1 : 0;
        float state[128], q[128], k[128];
        for (unsigned c = 0; c < p.hv; c++) state[c] = initial[(h * p.hv + row) * p.hv + c];
        for (unsigned t = 0; t < p.T; t++) {
            unsigned group = h % p.groups, base = t * p.conv_stride;
            double qs = 0, ks = 0;
            for (unsigned c = 0; c < p.ds; c++) {
                q[c] = conv[base + group * p.ds + c];
                k[c] = conv[base + p.groups * p.ds + group * p.ds + c];
                qs += (double)q[c] * q[c]; ks += (double)k[c] * k[c];
            }
            float qi = (float)(1 / sqrt(fmax(qs, 1e-12) * p.ds));
            float ki = (float)(1 / sqrt(fmax(ks, 1e-12)));
            for (unsigned c = 0; c < p.hv; c++) {
                q[c] = c < p.ds ? q[c] * qi : 0;
                k[c] = c < p.ds ? k[c] * ki : 0;
            }
            float a = alpha[t * p.heads + h] + (p.has_bias ? 0.125f : 0);
            float sp = logf(1 + expf(a));
            float gate = expf(sp * (p.has_a ? -0.5f : -1));
            float b = 1 / (1 + expf(-beta[t * p.heads + h]));
            double sk = 0;
            for (unsigned c = 0; c < p.hv; c++) { state[c] *= gate; sk += (double)state[c] * k[c]; }
            float v = conv[base + 2 * p.groups * p.ds + h * p.hv + row];
            float delta = b * (v - (float)sk);
            double out = 0;
            for (unsigned c = 0; c < p.hv; c++) { state[c] += k[c] * delta; out += (double)state[c] * q[c]; }
            close_value((float)out, output[t * p.out_stride + h * p.hv + row]);
        }
        for (unsigned c = 0; c < p.hv; c++) close_value(state[c], actual[(h * p.hv + row) * p.hv + c]);
    }
}
static void test_case(CudaCtx *ctx, CudaCtx *ref_ctx, CudaPipe *pipe, CudaPipe *ref,
                      unsigned T, unsigned mode) {
    Push p = {2,128,mode == 2 ? 64 : 128,1,mode == 1,mode == 1,mode != 3,mode != 3,T,0,2,256};
    if (mode == 4) { p.heads = 48; p.groups = 16; p.ab_stride = 48; p.out_stride = 48*128; }
    p.conv_stride = 2 * p.ds * p.groups + p.heads * p.hv;
    size_t cn = (size_t)T * p.conv_stride, an = (size_t)T * p.heads;
    size_t sn = (size_t)p.heads * p.hv * p.hv, yn = (size_t)T * p.out_stride;
    float *conv = values(cn), *alpha = values(an), *beta = values(an), *state = values(sn + 8), *out = values(yn + 8);
    for (size_t i = 0; i < cn; i++) conv[i] = random_float();
    for (size_t i = 0; i < an; i++) { alpha[i] = random_float(); beta[i] = random_float(); }
    for (size_t i = 0; i < sn + 8; i++) state[i] = i < sn ? random_float() / 8 : -999.25f;
    for (size_t i = 0; i < yn + 8; i++) out[i] = -999.25f;
    float bias[64], a[64];
    uint16_t bias16[64], a16[64];
    for (unsigned i = 0; i < 64; i++) { bias[i] = 0.125f; a[i] = -0.5f; bias16[i] = 0x3000; a16[i] = 0xb800; }
    CudaBuf *b[] = {
        buffer(ctx,conv,cn*4), buffer(ctx,p.bias_half ? (void*)bias16 : (void*)bias, p.heads*(p.bias_half ? 2 : 4)),
        buffer(ctx,alpha,an*4), buffer(ctx,beta,an*4), buffer(ctx,p.a_half ? (void*)a16 : (void*)a,p.heads*(p.a_half ? 2 : 4)),
        buffer(ctx,state,(sn+8)*4), buffer(ctx,out,(yn+8)*4),
    };
    CudaBuf *rs = buffer(ctx,state,(sn+8)*4), *ry = buffer(ctx,out,(yn+8)*4);
    unsigned grid[] = {p.heads,(p.hv+3)/4,1}, block[] = {32,4,1};
    CudaCmd *cmd = cuda_begin_command(ctx);
    cuda_dispatch(cmd,pipe,grid,block,b,7,&p,sizeof p,2*(T < 4096 ? T : 4096)*4);
    cuda_commit_and_wait(cmd);
    if (*cuda_last_error()) fail("tiled launch");
    unsigned reference_chunk = T <= 8192 ? T : 4096;
    for (unsigned start = 0; start < T; start += reference_chunk) {
        Push rp = p; rp.T = T-start < reference_chunk ? T-start : reference_chunk;
        CudaBuf *v[] = {
            cuda_alias_buffer(b[0],(size_t)start*p.conv_stride*4,(size_t)rp.T*p.conv_stride*4), b[1],
            cuda_alias_buffer(b[2],(size_t)start*p.heads*4,(size_t)rp.T*p.heads*4),
            cuda_alias_buffer(b[3],(size_t)start*p.heads*4,(size_t)rp.T*p.heads*4), b[4], rs,
            cuda_alias_buffer(ry,(size_t)start*p.out_stride*4,(size_t)rp.T*p.out_stride*4),
        };
        cmd = cuda_begin_command(ref_ctx);
        cuda_dispatch(cmd,ref,grid,block,v,7,&rp,sizeof rp,2*rp.T*4);
        cuda_commit_and_wait(cmd);
        cuda_free_buffer(v[0]); cuda_free_buffer(v[2]); cuda_free_buffer(v[3]); cuda_free_buffer(v[6]);
    }
    if (*cuda_last_error()) fail("reference launch");
    float *s1 = values(sn+8), *s2 = values(sn+8), *y1 = values(yn+8), *y2 = values(yn+8);
    cuda_download(ctx,b[5],s1,(sn+8)*4); cuda_download(ctx,rs,s2,(sn+8)*4);
    cuda_download(ctx,b[6],y1,(yn+8)*4); cuda_download(ctx,ry,y2,(yn+8)*4);
    if (memcmp(s1,s2,(sn+8)*4) || memcmp(y1,y2,(yn+8)*4)) fail("reference bits or guards differ");
    for (unsigned i = 0; i < 8; i++) if (s1[sn+i] != -999.25f || y1[yn+i] != -999.25f) fail("guard changed");
    cpu_rows(p,conv,alpha,beta,state,s1,y1);
    for (unsigned i = 0; i < 7; i++) cuda_free_buffer(b[i]);
    cuda_free_buffer(rs); cuda_free_buffer(ry);
    free(conv); free(alpha); free(beta); free(state); free(out); free(s1); free(s2); free(y1); free(y2);
    printf("PASS T=%u mode=%u: every GPU output/state bit and CPU rows match\n",T,mode);
    fflush(stdout);
}
int main(int argc, char **argv) {
    if (argc != 2) { fprintf(stderr,"Pass the original kernel source path.\n"); return 2; }
    char *current = source("src/shaders/cuda/kernels.cu"), *original = source(argv[1]);
    CudaCtx *ctx = cuda_init(0), *ref_ctx = cuda_init(0);
    if (!ctx || !ref_ctx) fail("GPU init");
    CudaPipe *pipe = cuda_create_pipeline(ctx,current,"ssm_delta_net_warp",NULL,0);
    CudaPipe *ref = cuda_create_pipeline(ref_ctx,original,"ssm_delta_net_warp",NULL,0);
    if (!pipe || !ref) fail("compile kernels");
    const unsigned lengths[] = {1,3,4,127,4095,4096,4097,8192,8193,9200,16384,32768};
    unsigned count = 0;
    for (unsigned i = 0; i < sizeof lengths/sizeof lengths[0]; i++) {
        test_case(ctx,ref_ctx,pipe,ref,lengths[i],0); count++;
    }
    for (unsigned mode = 1; mode <= 3; mode++) { test_case(ctx,ref_ctx,pipe,ref,8193,mode); count++; }
    test_case(ctx,ref_ctx,pipe,ref,6537,4); count++;
    test_case(ctx,ref_ctx,pipe,ref,8193,4); count++;
    cuda_free_pipeline(pipe); cuda_free_pipeline(ref); cuda_destroy(ctx); cuda_destroy(ref_ctx);
    free(current); free(original);
    printf("PASS: %u SSM cases; bounded LDS, exact GPU reference, independent CPU math\n",count);
    return 0;
}
