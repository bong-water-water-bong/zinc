// GPU host only; no model is loaded. Build with the same shim/library flags as
// validate_rocm_attention.c. Checks the single-row and 2..4-row v2 projections
// against independent double-precision packed-weight math and each other.
#include "cuda_shim.h"
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef struct { unsigned M, K, a_offset, x_offset, y_offset, acc_mode; } Mat;
typedef struct { unsigned M0, M1, K, pair_reduce; } Pair;
static uint32_t rng = 0x6bd03917;
static unsigned count;
static unsigned random_u32(void) {
    rng ^= rng << 13; rng ^= rng >> 17; rng ^= rng << 5; return rng;
}
static void fail(const char *what) {
    fprintf(stderr, "FAIL %s: %s\n", what, cuda_last_error()); exit(1);
}
static void *allocate(size_t n) {
    void *p = malloc(n); if (!p) fail("host allocation"); return p;
}
static char *source(void) {
    FILE *f = fopen("src/shaders/cuda/kernels.cu", "rb");
    if (!f || fseek(f, 0, SEEK_END)) fail("kernel source");
    long n = ftell(f);
    if (n < 0 || fseek(f, 0, SEEK_SET)) fail("kernel source size");
    char *s = allocate((size_t)n + 1);
    if (fread(s, 1, (size_t)n, f) != (size_t)n) fail("kernel source read");
    fclose(f); s[n] = 0; return s;
}
static void put_half(unsigned char *p, float value) {
    _Float16 h = (_Float16)value; memcpy(p, &h, 2);
}
static double half(const unsigned char *p) {
    _Float16 h; memcpy(&h, p, 2); return (double)h;
}
static CudaBuf *upload(CudaCtx *ctx, const void *p, size_t n) {
    CudaBuf *b = cuda_create_buffer(ctx, n);
    if (!b) fail("device allocation");
    cuda_upload(ctx, b, p, n);
    if (cuda_execution_failed(ctx)) fail("upload"); return b;
}
static void close_value(double expected, float actual) {
    if (!isfinite(actual) || fabs(expected - actual) > 0.0003 * (1 + fabs(expected))) {
        fprintf(stderr, "expected=%.12g actual=%.9g\n", expected, actual);
        fail("packed matvec reference");
    }
}
static void weights(unsigned char *p, size_t n, unsigned block) {
    for (size_t i = 0; i < n; i++) p[i] = (unsigned char)random_u32();
    for (size_t i = 0; i < n; i += block) {
        put_half(p + i, 0.001f * (1 + random_u32() % 5));
        put_half(p + i + 2, 0.0002f * (1 + random_u32() % 5));
    }
}
// The four Q8_1 groups in each 144-byte tile retain their own scale and sum.
static void activations(unsigned char *p, unsigned tiles) {
    for (unsigned t = 0; t < tiles * 4; t++) {
        unsigned char *tile = p + (size_t)t * 144;
        for (unsigned g = 0; g < 4; g++) {
            put_half(tile + g * 4, 0.005f * (1 + random_u32() % 5));
            int sum = 0;
            for (unsigned j = 0; j < 32; j++) {
                int q = (int)(random_u32() % 255) - 127;
                tile[16 + g * 32 + j] = (unsigned char)(signed char)q; sum += q;
            }
            put_half(tile + g * 4 + 2, (float)(sum * half(tile + g * 4)));
        }
    }
}
static double dot(const unsigned char *w, const unsigned char *x,
                  unsigned K, unsigned B, unsigned token, unsigned q5) {
    unsigned block = q5 ? 176 : 144;
    double result = 0;
    for (unsigned sb = 0; sb < K / 256; sb++) {
        const unsigned char *b = w + (size_t)sb * block, *s = b + 4;
        for (unsigned g = 0; g < 8; g++) {
            unsigned scale = g < 4 ? s[g] & 63 : (s[g + 4] & 15) | ((s[g - 4] >> 6) << 4);
            unsigned minimum = g < 4 ? s[g + 4] & 63 : (s[g + 4] >> 4) | ((s[g] >> 6) << 4);
            const unsigned char *a = x + ((size_t)(2 * sb + g / 4) * B + token) * 144;
            int total = 0;
            for (unsigned j = 0; j < 32; j++) {
                unsigned packed = b[(q5 ? 48 : 16) + (g / 2) * 32 + j];
                unsigned q = g & 1 ? packed >> 4 : packed & 15;
                if (q5) q += ((b[16 + j] >> g) & 1) * 16;
                total += (int)q * (int)(signed char)a[16 + (g % 4) * 32 + j];
            }
            result += half(b) * scale * half(a + (g % 4) * 4) * total
                    - half(b + 2) * minimum * half(a + (g % 4) * 4 + 2);
        }
    }
    return result;
}
int main(void) {
    char *s = source();
    CudaCtx *ctx = cuda_init(0); if (!ctx) fail("GPU init");
    const char *prefix[] = {"dmmv_q4k_q8", "dmmv_q5k_q8",
                           "dmmv_q4k_pair_q8", "dmmv_q4k_gate_up_swiglu_q8"};
    CudaPipe *pipes[4][4];
    for (unsigned mode = 0; mode < 4; mode++) for (unsigned B = 1; B <= 4; B++) {
        char name[96]; snprintf(name, sizeof name, "%s_btok%u_v2", prefix[mode], B);
        pipes[mode][B - 1] = cuda_create_pipeline(ctx, s, name, NULL, 0);
        if (!pipes[mode][B - 1]) fail("projection compile");
    }
    const unsigned dimensions[] = {256, 512, 5120, 17408};
    for (unsigned mode = 0; mode < 4; mode++) for (unsigned shape = 0; shape < 4; shape++) {
        const unsigned K = dimensions[shape], M = 17 + shape % 2, M1 = mode == 2 ? M - 3 : M;
        const unsigned block = mode == 1 ? 176 : 144, offset = mode < 2 ? block : 0;
        const size_t bytes = (size_t)M * (K / 256) * block + offset;
        unsigned char *w0 = allocate(bytes), *w1 = allocate(bytes);
        unsigned char *all = allocate((size_t)(K / 128) * 4 * 144);
        weights(w0, bytes, block); weights(w1, bytes, block); activations(all, K / 128);
        CudaBuf *a0 = upload(ctx, w0, bytes), *a1 = upload(ctx, w1, bytes);
        float first[2][24];
        for (unsigned B = 1; B <= 4; B++) {
            const size_t xn = (size_t)(K / 128) * B * 144, yn = (size_t)(B * M + 8);
            unsigned char *x = allocate(xn);
            for (unsigned tile = 0; tile < K / 128; tile++) for (unsigned t = 0; t < B; t++)
                memcpy(x + ((size_t)tile * B + t) * 144, all + ((size_t)tile * 4 + t) * 144, 144);
            float *initial = allocate(yn * 4), *out = allocate(yn * 4);
            for (size_t i = 0; i < yn; i++) initial[i] = 123456.0f;
            CudaBuf *xb = upload(ctx, x, xn), *y0 = upload(ctx, initial, yn * 4), *y1 = upload(ctx, initial, yn * 4);
            CudaBuf *args[] = {a0, a1, xb, y0, y1};
            Mat pc = {M, K, offset, 0, mode < 2 ? 16 : 0, 0}; Pair pair = {M, M1, K, 1};
            unsigned grid[] = {M, 1, 1}, threads[] = {32, 1, 1};
            CudaCmd *cmd = cuda_begin_command(ctx);
            if (mode < 2) { CudaBuf *a[] = {a0, xb, y0}; cuda_dispatch(cmd, pipes[mode][B - 1], grid, threads, a, 3, &pc, sizeof pc, 0); }
            else cuda_dispatch(cmd, pipes[mode][B - 1], grid, threads, args, mode == 2 ? 5 : 4, mode == 2 ? (void*)&pair : (void*)&pc, mode == 2 ? sizeof pair : sizeof pc, 0);
            cuda_commit_and_wait(cmd); if (cuda_execution_failed(ctx)) fail("projection launch");
            for (unsigned side = 0; side < (mode == 2 ? 2u : 1u); side++) {
                unsigned rows = side ? M1 : M, start = mode < 2 ? 4 : 0;
                cuda_download(ctx, side ? y1 : y0, out, yn * 4);
                for (unsigned t = 0; t < B; t++) for (unsigned row = 0; row < rows; row++) {
                    const unsigned char *w = (side ? w1 : w0) + offset + (size_t)row * (K / 256) * block;
                    double expected = dot(w, x, K, B, t, mode == 1);
                    if (mode == 3) expected = expected / (1 + exp(-expected)) * dot(w1 + (size_t)row * (K / 256) * block, x, K, B, t, 0);
                    close_value(expected, out[start + t * rows + row]);
                    if (t == 0) {
                        if (B == 1) first[side][row] = out[start + row];
                        else if (memcmp(&first[side][row], &out[start + row], 4)) fail("single/verifier row differs");
                    }
                }
                for (size_t i = 0; i < yn; i++) if ((i < start || i >= start + B * rows) && out[i] != initial[i]) fail("projection guard");
            }
            cuda_free_buffer(xb); cuda_free_buffer(y0); cuda_free_buffer(y1); free(x); free(initial); free(out);
            count++;
        }
        cuda_free_buffer(a0); cuda_free_buffer(a1); free(w0); free(w1); free(all);
    }
    for (unsigned mode = 0; mode < 4; mode++) for (unsigned B = 0; B < 4; B++) cuda_free_pipeline(pipes[mode][B]);
    cuda_destroy(ctx); free(s); printf("PASS %u packed projection/reference/guard cases\n", count); return 0;
}
