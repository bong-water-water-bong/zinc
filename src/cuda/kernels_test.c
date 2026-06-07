// Numeric validation for the ZINC CUDA kernels (src/shaders/cuda/kernels.cu).
// Reads the .cu at runtime, NVRTC-compiles each kernel via the shim, runs it on
// the GPU, and compares against an independent CPU reference. Standalone — no
// repo / Zig needed.
//
// Build (on the box, from ~/cuda_proto with kernels.cu present):
//   gcc -O2 -I. -I/usr/local/cuda/include kernels_test.c cuda_shim.c -o kernels_test \
//       -L/usr/local/cuda/lib64 -L/usr/lib/wsl/lib -lcuda -lnvrtc -lm \
//       -Wl,-rpath,/usr/local/cuda/lib64

#include "cuda_shim.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdint.h>

// ---- host mirrors of the kernel math (the ground-truth Q4_K spec) -----------
static float half_to_float_h(uint16_t h) {
    unsigned sign = (unsigned)(h >> 15) & 1u;
    unsigned exp = (unsigned)(h >> 10) & 0x1Fu;
    unsigned mant = (unsigned)h & 0x3FFu;
    unsigned f;
    if (exp == 0u) {
        if (mant == 0u) { f = sign << 31; }
        else { int e = 1; while ((mant & 0x400u) == 0u) { mant <<= 1; e--; }
               mant &= 0x3FFu; f = (sign << 31) | ((unsigned)(127 - 15 + e) << 23) | (mant << 13); }
    } else if (exp == 0x1Fu) { f = (sign << 31) | (0xFFu << 23) | (mant << 13); }
    else { f = (sign << 31) | ((exp - 15u + 127u) << 23) | (mant << 13); }
    float out; memcpy(&out, &f, 4); return out;
}

static void get_scale_min_k4_h(int j, const uint8_t* q, uint8_t* d, uint8_t* m) {
    if (j < 4) { *d = q[j] & 63u; *m = q[j + 4] & 63u; }
    else { *d = (q[j + 4] & 0xFu) | ((q[j - 4] >> 6) << 4);
           *m = (q[j + 4] >> 4) | ((q[j] >> 6) << 4); }
}

// Canonical llama.cpp dequant_row_q4_K for one 256-elem block (36 u32).
static void deq_q4k_block_h(const uint32_t* blk, float* out) {
    uint32_t dd = blk[0];
    float d = half_to_float_h((uint16_t)(dd & 0xFFFF));
    float dmin = half_to_float_h((uint16_t)(dd >> 16));
    const uint8_t* scales = (const uint8_t*)(blk + 1);
    const uint8_t* qs = (const uint8_t*)(blk + 4);
    int is = 0; const uint8_t* q = qs; float* y = out;
    for (int j = 0; j < 256; j += 64) {
        uint8_t sc, m;
        get_scale_min_k4_h(is + 0, scales, &sc, &m); float d1 = d * sc, m1 = dmin * m;
        get_scale_min_k4_h(is + 1, scales, &sc, &m); float d2 = d * sc, m2 = dmin * m;
        for (int l = 0; l < 32; l++) *y++ = d1 * (q[l] & 0xF) - m1;
        for (int l = 0; l < 32; l++) *y++ = d2 * (q[l] >> 4) - m2;
        q += 32; is += 2;
    }
}

// Canonical llama.cpp dequant_row_q5_K for one 256-elem block (176 bytes).
static void deq_q5k_block_h(const unsigned char* blk, float* out) {
    float d = half_to_float_h((uint16_t)(blk[0] | (blk[1] << 8)));
    float dmin = half_to_float_h((uint16_t)(blk[2] | (blk[3] << 8)));
    const uint8_t* scales = blk + 4;
    const uint8_t* qh = blk + 16;
    const uint8_t* qlp = blk + 48;
    int is = 0; uint8_t u1 = 1, u2 = 2; float* y = out;
    for (int j = 0; j < 256; j += 64) {
        uint8_t sc, m;
        get_scale_min_k4_h(is + 0, scales, &sc, &m); float d1 = d * sc, m1 = dmin * m;
        get_scale_min_k4_h(is + 1, scales, &sc, &m); float d2 = d * sc, m2 = dmin * m;
        for (int l = 0; l < 32; l++) *y++ = d1 * ((qlp[l] & 0xF) + ((qh[l] & u1) ? 16 : 0)) - m1;
        for (int l = 0; l < 32; l++) *y++ = d2 * ((qlp[l] >> 4)  + ((qh[l] & u2) ? 16 : 0)) - m2;
        qlp += 32; is += 2; u1 <<= 2; u2 <<= 2;
    }
}

// Canonical llama.cpp dequant_row_q6_K for one 256-elem block (210 bytes).
static void deq_q6k_block_h(const unsigned char* blk, float* out) {
    float d = half_to_float_h((uint16_t)(blk[208] | (blk[209] << 8)));
    const uint8_t* ql = blk + 0;
    const uint8_t* qh = blk + 128;
    const int8_t* sc = (const int8_t*)(blk + 192);
    float* y = out;
    for (int n = 0; n < 256; n += 128) {
        const uint8_t* qlh = ql + (n / 128) * 64;
        const uint8_t* qhh = qh + (n / 128) * 32;
        const int8_t* sch = sc + (n / 128) * 8;
        for (int l = 0; l < 32; l++) {
            int is = l / 16;
            int q1 = (int)((qlh[l] & 0xF) | (((qhh[l] >> 0) & 3) << 4)) - 32;
            int q2 = (int)((qlh[l + 32] & 0xF) | (((qhh[l] >> 2) & 3) << 4)) - 32;
            int q3 = (int)((qlh[l] >> 4) | (((qhh[l] >> 4) & 3) << 4)) - 32;
            int q4 = (int)((qlh[l + 32] >> 4) | (((qhh[l] >> 6) & 3) << 4)) - 32;
            y[n + l +  0] = d * sch[is + 0] * q1;
            y[n + l + 32] = d * sch[is + 2] * q2;
            y[n + l + 64] = d * sch[is + 4] * q3;
            y[n + l + 96] = d * sch[is + 6] * q4;
        }
    }
}

// ---- deterministic PRNG + helpers -------------------------------------------
static uint32_t rng = 0x9e3779b9u;
static uint32_t xrand(void) { rng ^= rng << 13; rng ^= rng >> 17; rng ^= rng << 5; return rng; }
static float frand(void) { return (float)(xrand() & 0xFFFFFF) / (float)0x1000000 * 2.0f - 1.0f; }
// "nice" positive normal half: exp field 7..13 -> magnitudes ~0.004..0.25.
static uint16_t nice_half(void) { uint16_t e = 7 + (uint16_t)(xrand() % 7); return (uint16_t)((e << 10) | (xrand() & 0x3FF)); }

static char* read_file(const char* path) {
    FILE* f = fopen(path, "rb"); if (!f) return NULL;
    fseek(f, 0, SEEK_END); long n = ftell(f); fseek(f, 0, SEEK_SET);
    char* buf = (char*)malloc(n + 1); if (!buf) { fclose(f); return NULL; }
    size_t rd = fread(buf, 1, n, f); buf[rd] = 0; fclose(f); return buf;
}

static int pick_best_device(void) {
    int best = -1; unsigned best_cc = 0;
    for (int i = 0; i < 8; i++) {
        CudaCtx* c = cuda_init(i); if (!c) break;
        unsigned cc = cuda_compute_capability(c);
        if (cc > best_cc) { best_cc = cc; best = i; }
        cuda_destroy(c);
    }
    return best;
}

int main(void) {
    int dev = pick_best_device();
    if (dev < 0) { printf("FAIL: no CUDA device\n"); return 1; }
    CudaCtx* c = cuda_init(dev);
    char nm[128]; cuda_device_name(c, nm, sizeof nm);
    printf("device: %s (cc=%u)\n", nm, cuda_compute_capability(c));

    char* src = read_file("kernels.cu");
    if (!src) { printf("FAIL: cannot read kernels.cu (run from ~/cuda_proto)\n"); return 1; }
    int all_ok = 1;

    // ===== Test 1: rms_norm =====
    {
        const unsigned tokens = 3, N = 2048; const float eps = 1e-5f;
        float* x = malloc((size_t)tokens * N * 4);
        float* w = malloc((size_t)N * 4);
        float* yref = malloc((size_t)tokens * N * 4);
        float* ygpu = malloc((size_t)tokens * N * 4);
        for (unsigned i = 0; i < tokens * N; i++) x[i] = frand();
        for (unsigned i = 0; i < N; i++) w[i] = frand() * 0.5f + 1.0f;
        for (unsigned t = 0; t < tokens; t++) {
            double ss = 0; for (unsigned i = 0; i < N; i++) { float v = x[t * N + i]; ss += (double)v * v; }
            float rinv = 1.0f / sqrtf((float)(ss / N) + eps);
            for (unsigned i = 0; i < N; i++) yref[t * N + i] = w[i] * (x[t * N + i] * rinv);
        }
        CudaBuf* dx = cuda_create_buffer(c, (size_t)tokens * N * 4);
        CudaBuf* dw = cuda_create_buffer(c, (size_t)N * 4);
        CudaBuf* dy = cuda_create_buffer(c, (size_t)tokens * N * 4);
        cuda_upload(c, dx, x, (size_t)tokens * N * 4);
        cuda_upload(c, dw, w, (size_t)N * 4);
        CudaPipe* p = cuda_create_pipeline(c, src, "rms_norm", NULL, 0);
        if (!p) { printf("FAIL rms_norm compile: %s\n", cuda_last_error()); return 2; }
        struct { unsigned N; float eps; } push = { N, eps };
        uint32_t grid[3] = { tokens, 1, 1 }, block[3] = { 256, 1, 1 };
        CudaBuf* bufs[3] = { dx, dw, dy };
        CudaCmd* cmd = cuda_begin_command(c);
        cuda_dispatch(cmd, p, grid, block, bufs, 3, &push, sizeof push, 0);
        cuda_commit_and_wait(cmd);
        cuda_download(c, dy, ygpu, (size_t)tokens * N * 4);
        float maxrel = 0;
        for (unsigned i = 0; i < tokens * N; i++) {
            float a = yref[i], b = ygpu[i], r = fabsf(a - b) / (fabsf(a) + 1e-4f);
            if (r > maxrel) maxrel = r;
        }
        int ok = maxrel < 1e-3f; all_ok &= ok;
        printf("rms_norm [%ux%u]: max_rel_err=%.2e -> %s\n", tokens, N, maxrel, ok ? "PASS" : "FAIL");
        cuda_free_buffer(dx); cuda_free_buffer(dw); cuda_free_buffer(dy); cuda_free_pipeline(p);
        free(x); free(w); free(yref); free(ygpu);
    }

    // ===== Test 2: dmmv_q4k =====
    {
        const unsigned M = 5, K = 512; unsigned bpr = K / 256; unsigned nblk = M * bpr;
        uint32_t* a = malloc((size_t)nblk * 36 * 4);
        for (unsigned bi = 0; bi < nblk; bi++) {
            uint32_t* blk = a + (size_t)bi * 36;
            blk[0] = nice_half() | ((uint32_t)nice_half() << 16);
            uint8_t* scales = (uint8_t*)(blk + 1); for (int k = 0; k < 12; k++) scales[k] = xrand() & 0xFF;
            uint8_t* qs = (uint8_t*)(blk + 4); for (int k = 0; k < 128; k++) qs[k] = xrand() & 0xFF;
        }
        float* x = malloc((size_t)K * 4); for (unsigned i = 0; i < K; i++) x[i] = frand();
        float* yref = malloc((size_t)M * 4);
        float deq[256];
        for (unsigned row = 0; row < M; row++) {
            double acc = 0;
            for (unsigned b = 0; b < bpr; b++) {
                deq_q4k_block_h(a + (size_t)(row * bpr + b) * 36, deq);
                for (int e = 0; e < 256; e++) acc += (double)deq[e] * x[b * 256 + e];
            }
            yref[row] = (float)acc;
        }
        CudaBuf* da = cuda_create_buffer(c, (size_t)nblk * 36 * 4);
        CudaBuf* dx = cuda_create_buffer(c, (size_t)K * 4);
        CudaBuf* dy = cuda_create_buffer(c, (size_t)M * 4);
        cuda_upload(c, da, a, (size_t)nblk * 36 * 4);
        cuda_upload(c, dx, x, (size_t)K * 4);
        CudaPipe* p = cuda_create_pipeline(c, src, "dmmv_q4k", NULL, 0);
        if (!p) { printf("FAIL dmmv_q4k compile: %s\n", cuda_last_error()); return 2; }
        struct { unsigned M, K, a_off, x_off, y_off, acc; } push = { M, K, 0, 0, 0, 0 };
        uint32_t grid[3] = { M, 1, 1 }, block[3] = { 256, 1, 1 };
        CudaBuf* bufs[3] = { da, dx, dy };
        CudaCmd* cmd = cuda_begin_command(c);
        cuda_dispatch(cmd, p, grid, block, bufs, 3, &push, sizeof push, 0);
        cuda_commit_and_wait(cmd);
        float* ygpu = malloc((size_t)M * 4); cuda_download(c, dy, ygpu, (size_t)M * 4);
        float maxrel = 0;
        for (unsigned r = 0; r < M; r++) {
            float rr = fabsf(yref[r] - ygpu[r]) / (fabsf(yref[r]) + 1e-2f);
            printf("  row %u: ref=%.3f gpu=%.3f\n", r, yref[r], ygpu[r]);
            if (rr > maxrel) maxrel = rr;
        }
        int ok = maxrel < 2e-3f; all_ok &= ok;
        printf("dmmv_q4k [M=%u K=%u]: max_rel_err=%.2e -> %s\n", M, K, maxrel, ok ? "PASS" : "FAIL");
        cuda_free_buffer(da); cuda_free_buffer(dx); cuda_free_buffer(dy); cuda_free_pipeline(p);
        free(a); free(x); free(yref); free(ygpu);
    }

    // ===== Test 3: swiglu =====
    {
        const unsigned N = 4096;
        float* gate = malloc((size_t)N * 4); float* up = malloc((size_t)N * 4);
        float* yref = malloc((size_t)N * 4); float* ygpu = malloc((size_t)N * 4);
        for (unsigned i = 0; i < N; i++) { gate[i] = frand() * 4.0f; up[i] = frand(); }
        for (unsigned i = 0; i < N; i++) { float g = gate[i]; yref[i] = (g / (1.0f + expf(-g))) * up[i]; }
        CudaBuf* dg = cuda_create_buffer(c, (size_t)N * 4); CudaBuf* du = cuda_create_buffer(c, (size_t)N * 4);
        CudaBuf* dy = cuda_create_buffer(c, (size_t)N * 4);
        cuda_upload(c, dg, gate, (size_t)N * 4); cuda_upload(c, du, up, (size_t)N * 4);
        CudaPipe* p = cuda_create_pipeline(c, src, "swiglu", NULL, 0);
        if (!p) { printf("FAIL swiglu compile: %s\n", cuda_last_error()); return 2; }
        struct { unsigned N; } push = { N };
        uint32_t grid[3] = { (N + 255) / 256, 1, 1 }, block[3] = { 256, 1, 1 };
        CudaBuf* bufs[3] = { dg, du, dy };
        CudaCmd* cmd = cuda_begin_command(c);
        cuda_dispatch(cmd, p, grid, block, bufs, 3, &push, sizeof push, 0);
        cuda_commit_and_wait(cmd);
        cuda_download(c, dy, ygpu, (size_t)N * 4);
        float mr = 0; for (unsigned i = 0; i < N; i++) { float r = fabsf(yref[i] - ygpu[i]) / (fabsf(yref[i]) + 1e-4f); if (r > mr) mr = r; }
        int ok = mr < 1e-3f; all_ok &= ok;
        printf("swiglu [%u]: max_rel_err=%.2e -> %s\n", N, mr, ok ? "PASS" : "FAIL");
        cuda_free_buffer(dg); cuda_free_buffer(du); cuda_free_buffer(dy); cuda_free_pipeline(p);
        free(gate); free(up); free(yref); free(ygpu);
    }

    // ===== Test 4: scale_accumulate (a += scale*b) =====
    {
        const unsigned N = 4096; const float scale = 0.37f;
        float* a0 = malloc((size_t)N * 4); float* b = malloc((size_t)N * 4);
        float* aref = malloc((size_t)N * 4); float* agpu = malloc((size_t)N * 4);
        for (unsigned i = 0; i < N; i++) { a0[i] = frand(); b[i] = frand(); }
        for (unsigned i = 0; i < N; i++) aref[i] = a0[i] + scale * b[i];
        CudaBuf* da = cuda_create_buffer(c, (size_t)N * 4); CudaBuf* db = cuda_create_buffer(c, (size_t)N * 4);
        cuda_upload(c, da, a0, (size_t)N * 4); cuda_upload(c, db, b, (size_t)N * 4);
        CudaPipe* p = cuda_create_pipeline(c, src, "scale_accumulate", NULL, 0);
        if (!p) { printf("FAIL scale_accumulate compile: %s\n", cuda_last_error()); return 2; }
        struct { unsigned N; float scale; } push = { N, scale };
        uint32_t grid[3] = { (N + 255) / 256, 1, 1 }, block[3] = { 256, 1, 1 };
        CudaBuf* bufs[2] = { da, db };
        CudaCmd* cmd = cuda_begin_command(c);
        cuda_dispatch(cmd, p, grid, block, bufs, 2, &push, sizeof push, 0);
        cuda_commit_and_wait(cmd);
        cuda_download(c, da, agpu, (size_t)N * 4);
        float mr = 0; for (unsigned i = 0; i < N; i++) { float r = fabsf(aref[i] - agpu[i]) / (fabsf(aref[i]) + 1e-4f); if (r > mr) mr = r; }
        int ok = mr < 1e-3f; all_ok &= ok;
        printf("scale_accumulate [%u]: max_rel_err=%.2e -> %s\n", N, mr, ok ? "PASS" : "FAIL");
        cuda_free_buffer(da); cuda_free_buffer(db); cuda_free_pipeline(p);
        free(a0); free(b); free(aref); free(agpu);
    }

    // ===== Test 5: sigmoid_scale_acc (a += sigmoid(c0)*b) =====
    {
        const unsigned N = 4096; const float cgate = 0.6f;
        float* a0 = malloc((size_t)N * 4); float* b = malloc((size_t)N * 4);
        float* aref = malloc((size_t)N * 4); float* agpu = malloc((size_t)N * 4);
        for (unsigned i = 0; i < N; i++) { a0[i] = frand(); b[i] = frand(); }
        float g = 1.0f / (1.0f + expf(-cgate));
        for (unsigned i = 0; i < N; i++) aref[i] = a0[i] + g * b[i];
        CudaBuf* da = cuda_create_buffer(c, (size_t)N * 4); CudaBuf* db = cuda_create_buffer(c, (size_t)N * 4);
        CudaBuf* dc = cuda_create_buffer(c, 4);
        cuda_upload(c, da, a0, (size_t)N * 4); cuda_upload(c, db, b, (size_t)N * 4); cuda_upload(c, dc, &cgate, 4);
        CudaPipe* p = cuda_create_pipeline(c, src, "sigmoid_scale_acc", NULL, 0);
        if (!p) { printf("FAIL sigmoid_scale_acc compile: %s\n", cuda_last_error()); return 2; }
        struct { unsigned N; } push = { N };
        uint32_t grid[3] = { (N + 255) / 256, 1, 1 }, block[3] = { 256, 1, 1 };
        CudaBuf* bufs[3] = { da, db, dc };
        CudaCmd* cmd = cuda_begin_command(c);
        cuda_dispatch(cmd, p, grid, block, bufs, 3, &push, sizeof push, 0);
        cuda_commit_and_wait(cmd);
        cuda_download(c, da, agpu, (size_t)N * 4);
        float mr = 0; for (unsigned i = 0; i < N; i++) { float r = fabsf(aref[i] - agpu[i]) / (fabsf(aref[i]) + 1e-4f); if (r > mr) mr = r; }
        int ok = mr < 1e-3f; all_ok &= ok;
        printf("sigmoid_scale_acc [%u]: max_rel_err=%.2e -> %s\n", N, mr, ok ? "PASS" : "FAIL");
        cuda_free_buffer(da); cuda_free_buffer(db); cuda_free_buffer(dc); cuda_free_pipeline(p);
        free(a0); free(b); free(aref); free(agpu);
    }

    // ===== Test 6: dmmv_f32 =====
    {
        const unsigned M = 5, K = 512;
        float* w = malloc((size_t)M * K * 4); float* x = malloc((size_t)K * 4);
        float* yref = malloc((size_t)M * 4); float* ygpu = malloc((size_t)M * 4);
        for (unsigned i = 0; i < M * K; i++) w[i] = frand();
        for (unsigned i = 0; i < K; i++) x[i] = frand();
        for (unsigned r = 0; r < M; r++) { double acc = 0; for (unsigned k = 0; k < K; k++) acc += (double)w[r * K + k] * x[k]; yref[r] = (float)acc; }
        CudaBuf* dw = cuda_create_buffer(c, (size_t)M * K * 4); CudaBuf* dx = cuda_create_buffer(c, (size_t)K * 4); CudaBuf* dy = cuda_create_buffer(c, (size_t)M * 4);
        cuda_upload(c, dw, w, (size_t)M * K * 4); cuda_upload(c, dx, x, (size_t)K * 4);
        CudaPipe* p = cuda_create_pipeline(c, src, "dmmv_f32", NULL, 0);
        if (!p) { printf("FAIL dmmv_f32 compile: %s\n", cuda_last_error()); return 2; }
        struct { unsigned M, K, aoff, xoff, yoff, acc; } push = { M, K, 0, 0, 0, 0 };
        uint32_t grid[3] = { M, 1, 1 }, block[3] = { 256, 1, 1 }; CudaBuf* bufs[3] = { dw, dx, dy };
        CudaCmd* cmd = cuda_begin_command(c); cuda_dispatch(cmd, p, grid, block, bufs, 3, &push, sizeof push, 0); cuda_commit_and_wait(cmd);
        cuda_download(c, dy, ygpu, (size_t)M * 4);
        float mr = 0; for (unsigned r = 0; r < M; r++) { float rr = fabsf(yref[r] - ygpu[r]) / (fabsf(yref[r]) + 1e-2f); if (rr > mr) mr = rr; }
        int ok = mr < 2e-3f; all_ok &= ok;
        printf("dmmv_f32 [M=%u K=%u]: max_rel_err=%.2e -> %s\n", M, K, mr, ok ? "PASS" : "FAIL");
        cuda_free_buffer(dw); cuda_free_buffer(dx); cuda_free_buffer(dy); cuda_free_pipeline(p);
        free(w); free(x); free(yref); free(ygpu);
    }

    // ===== Test 7: dmmv_q8_0 =====
    {
        const unsigned M = 5, K = 512; unsigned bpr = K / 32; size_t bytes = (size_t)M * bpr * 34;
        unsigned char* a = malloc(bytes);
        for (unsigned bi = 0; bi < M * bpr; bi++) {
            unsigned char* blk = a + (size_t)bi * 34;
            uint16_t d = nice_half(); blk[0] = d & 0xFF; blk[1] = d >> 8;
            for (int i = 0; i < 32; i++) blk[2 + i] = (unsigned char)(xrand() & 0xFF);
        }
        float* x = malloc((size_t)K * 4); for (unsigned i = 0; i < K; i++) x[i] = frand();
        float* yref = malloc((size_t)M * 4); float* ygpu = malloc((size_t)M * 4);
        for (unsigned r = 0; r < M; r++) {
            double acc = 0;
            for (unsigned b = 0; b < bpr; b++) {
                unsigned char* blk = a + (size_t)(r * bpr + b) * 34;
                uint16_t db = (uint16_t)(blk[0] | (blk[1] << 8)); float d = half_to_float_h(db);
                for (int i = 0; i < 32; i++) { signed char q = (signed char)blk[2 + i]; acc += (double)(d * (float)q) * x[b * 32 + i]; }
            }
            yref[r] = (float)acc;
        }
        CudaBuf* da = cuda_create_buffer(c, bytes); CudaBuf* dx = cuda_create_buffer(c, (size_t)K * 4); CudaBuf* dy = cuda_create_buffer(c, (size_t)M * 4);
        cuda_upload(c, da, a, bytes); cuda_upload(c, dx, x, (size_t)K * 4);
        CudaPipe* p = cuda_create_pipeline(c, src, "dmmv_q8_0", NULL, 0);
        if (!p) { printf("FAIL dmmv_q8_0 compile: %s\n", cuda_last_error()); return 2; }
        struct { unsigned M, K, aoff, xoff, yoff, acc; } push = { M, K, 0, 0, 0, 0 };
        uint32_t grid[3] = { M, 1, 1 }, block[3] = { 256, 1, 1 }; CudaBuf* bufs[3] = { da, dx, dy };
        CudaCmd* cmd = cuda_begin_command(c); cuda_dispatch(cmd, p, grid, block, bufs, 3, &push, sizeof push, 0); cuda_commit_and_wait(cmd);
        cuda_download(c, dy, ygpu, (size_t)M * 4);
        float mr = 0; for (unsigned r = 0; r < M; r++) { float rr = fabsf(yref[r] - ygpu[r]) / (fabsf(yref[r]) + 1e-2f); if (rr > mr) mr = rr; }
        int ok = mr < 2e-3f; all_ok &= ok;
        printf("dmmv_q8_0 [M=%u K=%u]: max_rel_err=%.2e -> %s\n", M, K, mr, ok ? "PASS" : "FAIL");
        cuda_free_buffer(da); cuda_free_buffer(dx); cuda_free_buffer(dy); cuda_free_pipeline(p);
        free(a); free(x); free(yref); free(ygpu);
    }

    // ===== Test 8: dmmv_q5k =====
    {
        const unsigned M = 5, K = 512; unsigned bpr = K / 256; size_t bytes = (size_t)M * bpr * 176;
        unsigned char* a = malloc(bytes);
        for (unsigned bi = 0; bi < M * bpr; bi++) {
            unsigned char* blk = a + (size_t)bi * 176;
            uint16_t d = nice_half(), dm = nice_half();
            blk[0] = d & 0xFF; blk[1] = d >> 8; blk[2] = dm & 0xFF; blk[3] = dm >> 8;
            for (int k = 4; k < 176; k++) blk[k] = (unsigned char)(xrand() & 0xFF);
        }
        float* x = malloc((size_t)K * 4); for (unsigned i = 0; i < K; i++) x[i] = frand();
        float* yref = malloc((size_t)M * 4); float* ygpu = malloc((size_t)M * 4); float deq[256];
        for (unsigned r = 0; r < M; r++) {
            double acc = 0;
            for (unsigned b = 0; b < bpr; b++) { deq_q5k_block_h(a + (size_t)(r * bpr + b) * 176, deq); for (int e = 0; e < 256; e++) acc += (double)deq[e] * x[b * 256 + e]; }
            yref[r] = (float)acc;
        }
        CudaBuf* da = cuda_create_buffer(c, bytes); CudaBuf* dx = cuda_create_buffer(c, (size_t)K * 4); CudaBuf* dy = cuda_create_buffer(c, (size_t)M * 4);
        cuda_upload(c, da, a, bytes); cuda_upload(c, dx, x, (size_t)K * 4);
        CudaPipe* p = cuda_create_pipeline(c, src, "dmmv_q5k", NULL, 0);
        if (!p) { printf("FAIL dmmv_q5k compile: %s\n", cuda_last_error()); return 2; }
        struct { unsigned M, K, aoff, xoff, yoff, acc; } push = { M, K, 0, 0, 0, 0 };
        uint32_t grid[3] = { M, 1, 1 }, block[3] = { 256, 1, 1 }; CudaBuf* bufs[3] = { da, dx, dy };
        CudaCmd* cmd = cuda_begin_command(c); cuda_dispatch(cmd, p, grid, block, bufs, 3, &push, sizeof push, 0); cuda_commit_and_wait(cmd);
        cuda_download(c, dy, ygpu, (size_t)M * 4);
        float mr = 0; for (unsigned r = 0; r < M; r++) { float rr = fabsf(yref[r] - ygpu[r]) / (fabsf(yref[r]) + 1e-2f); if (rr > mr) mr = rr; }
        int ok = mr < 2e-3f; all_ok &= ok;
        printf("dmmv_q5k [M=%u K=%u]: max_rel_err=%.2e -> %s\n", M, K, mr, ok ? "PASS" : "FAIL");
        cuda_free_buffer(da); cuda_free_buffer(dx); cuda_free_buffer(dy); cuda_free_pipeline(p);
        free(a); free(x); free(yref); free(ygpu);
    }

    // ===== Test 9: dmmv_q6k =====
    {
        const unsigned M = 5, K = 512; unsigned bpr = K / 256; size_t bytes = (size_t)M * bpr * 210;
        unsigned char* a = malloc(bytes);
        for (unsigned bi = 0; bi < M * bpr; bi++) {
            unsigned char* blk = a + (size_t)bi * 210;
            for (int k = 0; k < 208; k++) blk[k] = (unsigned char)(xrand() & 0xFF);
            uint16_t d = nice_half(); blk[208] = d & 0xFF; blk[209] = d >> 8;
        }
        float* x = malloc((size_t)K * 4); for (unsigned i = 0; i < K; i++) x[i] = frand();
        float* yref = malloc((size_t)M * 4); float* ygpu = malloc((size_t)M * 4); float deq[256];
        for (unsigned r = 0; r < M; r++) {
            double acc = 0;
            for (unsigned b = 0; b < bpr; b++) { deq_q6k_block_h(a + (size_t)(r * bpr + b) * 210, deq); for (int e = 0; e < 256; e++) acc += (double)deq[e] * x[b * 256 + e]; }
            yref[r] = (float)acc;
        }
        CudaBuf* da = cuda_create_buffer(c, bytes); CudaBuf* dx = cuda_create_buffer(c, (size_t)K * 4); CudaBuf* dy = cuda_create_buffer(c, (size_t)M * 4);
        cuda_upload(c, da, a, bytes); cuda_upload(c, dx, x, (size_t)K * 4);
        CudaPipe* p = cuda_create_pipeline(c, src, "dmmv_q6k", NULL, 0);
        if (!p) { printf("FAIL dmmv_q6k compile: %s\n", cuda_last_error()); return 2; }
        struct { unsigned M, K, aoff, xoff, yoff, acc; } push = { M, K, 0, 0, 0, 0 };
        uint32_t grid[3] = { M, 1, 1 }, block[3] = { 256, 1, 1 }; CudaBuf* bufs[3] = { da, dx, dy };
        CudaCmd* cmd = cuda_begin_command(c); cuda_dispatch(cmd, p, grid, block, bufs, 3, &push, sizeof push, 0); cuda_commit_and_wait(cmd);
        cuda_download(c, dy, ygpu, (size_t)M * 4);
        float mr = 0; for (unsigned r = 0; r < M; r++) { float rr = fabsf(yref[r] - ygpu[r]) / (fabsf(yref[r]) + 1e-2f); if (rr > mr) mr = rr; }
        int ok = mr < 2e-3f; all_ok &= ok;
        printf("dmmv_q6k [M=%u K=%u]: max_rel_err=%.2e -> %s\n", M, K, mr, ok ? "PASS" : "FAIL");
        cuda_free_buffer(da); cuda_free_buffer(dx); cuda_free_buffer(dy); cuda_free_pipeline(p);
        free(a); free(x); free(yref); free(ygpu);
    }

    // ===== Test 10: softmax_topk =====
    {
        const unsigned NE = 128, K = 8;
        float* logits = malloc((size_t)NE * 4);
        for (unsigned i = 0; i < NE; i++) logits[i] = frand() * 4.0f;
        // CPU ref: top-k by logit, then renormalized softmax over the winners.
        float lc[128]; memcpy(lc, logits, (size_t)NE * 4);
        unsigned ref_id[8]; float ref_logit[8];
        for (unsigned ki = 0; ki < K; ki++) {
            float b = -1e30f; unsigned bi = 0;
            for (unsigned i = 0; i < NE; i++) if (lc[i] > b) { b = lc[i]; bi = i; }
            ref_id[ki] = bi; ref_logit[ki] = b; lc[bi] = -1e30f;
        }
        float maxl = -1e30f; for (unsigned i = 0; i < K; i++) maxl = fmaxf(maxl, ref_logit[i]);
        float ws = 0, ref_w[8];
        for (unsigned i = 0; i < K; i++) { ref_w[i] = expf(ref_logit[i] - maxl); ws += ref_w[i]; }
        for (unsigned i = 0; i < K; i++) ref_w[i] /= ws;
        CudaBuf* dl = cuda_create_buffer(c, (size_t)NE * 4); CudaBuf* dout = cuda_create_buffer(c, (size_t)2 * K * 4);
        cuda_upload(c, dl, logits, (size_t)NE * 4);
        CudaPipe* p = cuda_create_pipeline(c, src, "softmax_topk", NULL, 0);
        if (!p) { printf("FAIL softmax_topk compile: %s\n", cuda_last_error()); return 2; }
        struct { unsigned ne, k; } push = { NE, K };
        uint32_t grid[3] = { 1, 1, 1 }, block[3] = { 64, 1, 1 }; CudaBuf* bufs[2] = { dl, dout };
        CudaCmd* cmd = cuda_begin_command(c); cuda_dispatch(cmd, p, grid, block, bufs, 2, &push, sizeof push, 0); cuda_commit_and_wait(cmd);
        unsigned go[16]; cuda_download(c, dout, go, (size_t)2 * K * 4);
        int ids_ok = 1; float wmax = 0;
        for (unsigned i = 0; i < K; i++) {
            if (go[i] != ref_id[i]) ids_ok = 0;
            float gw; memcpy(&gw, &go[K + i], 4);
            float e = fabsf(gw - ref_w[i]); if (e > wmax) wmax = e;
        }
        int ok = ids_ok && wmax < 1e-5f; all_ok &= ok;
        printf("softmax_topk [NE=%u K=%u]: ids_match=%d w_max_err=%.2e -> %s\n", NE, K, ids_ok, wmax, ok ? "PASS" : "FAIL");
        cuda_free_buffer(dl); cuda_free_buffer(dout); cuda_free_pipeline(p); free(logits);
    }

    // ===== Test 11: rope (partial rotation) =====
    {
        const unsigned n_heads = 3, stride = 128, rope_dim = 64, position = 5;
        float freq_base = 1000000.0f; unsigned fbb; memcpy(&fbb, &freq_base, 4);
        unsigned total = n_heads * stride, half = rope_dim / 2;
        float* x = malloc((size_t)total * 4); for (unsigned i = 0; i < total; i++) x[i] = frand();
        float* yref = malloc((size_t)total * 4); float* ygpu = malloc((size_t)total * 4);
        for (unsigned h = 0; h < n_heads; h++) {
            unsigned base = h * stride;
            for (unsigned i = 0; i < half; i++) {
                float xi = x[base + i], xih = x[base + i + half];
                float freq = 1.0f / powf(freq_base, (float)(2 * i) / (float)rope_dim);
                float th = (float)position * freq, ct = cosf(th), st = sinf(th);
                yref[base + i] = xi * ct - xih * st;
                yref[base + i + half] = xi * st + xih * ct;
            }
            for (unsigned i = rope_dim; i < stride; i++) yref[base + i] = x[base + i];
        }
        CudaBuf* dx = cuda_create_buffer(c, (size_t)total * 4); CudaBuf* dy = cuda_create_buffer(c, (size_t)total * 4); CudaBuf* df = cuda_create_buffer(c, 4);
        cuda_upload(c, dx, x, (size_t)total * 4);
        CudaPipe* p = cuda_create_pipeline(c, src, "rope", NULL, 0);
        if (!p) { printf("FAIL rope compile: %s\n", cuda_last_error()); return 2; }
        struct { unsigned stride, rope_dim, n_heads, position, fbb, asb; } push = { stride, rope_dim, n_heads, position, fbb, 0 };
        uint32_t grid[3] = { n_heads, 1, 1 }, block[3] = { 64, 1, 1 }; CudaBuf* bufs[3] = { dx, dy, df };
        CudaCmd* cmd = cuda_begin_command(c); cuda_dispatch(cmd, p, grid, block, bufs, 3, &push, sizeof push, 0); cuda_commit_and_wait(cmd);
        cuda_download(c, dy, ygpu, (size_t)total * 4);
        float mr = 0; for (unsigned i = 0; i < total; i++) { float r = fabsf(yref[i] - ygpu[i]) / (fabsf(yref[i]) + 1e-4f); if (r > mr) mr = r; }
        int ok = mr < 1e-3f; all_ok &= ok;
        printf("rope [heads=%u stride=%u rope_dim=%u]: max_rel_err=%.2e -> %s\n", n_heads, stride, rope_dim, mr, ok ? "PASS" : "FAIL");
        cuda_free_buffer(dx); cuda_free_buffer(dy); cuda_free_buffer(df); cuda_free_pipeline(p);
        free(x); free(yref); free(ygpu);
    }

    // ===== Test 12: argmax =====
    {
        const unsigned N = 4096;
        float* logits = malloc((size_t)N * 4);
        for (unsigned i = 0; i < N; i++) logits[i] = frand() * 10.0f;
        unsigned ref = 0; float best = -1e30f;
        for (unsigned i = 0; i < N; i++) if (logits[i] > best) { best = logits[i]; ref = i; }
        CudaBuf* dl = cuda_create_buffer(c, (size_t)N * 4); CudaBuf* dt = cuda_create_buffer(c, 4);
        cuda_upload(c, dl, logits, (size_t)N * 4);
        CudaPipe* p = cuda_create_pipeline(c, src, "argmax", NULL, 0);
        if (!p) { printf("FAIL argmax compile: %s\n", cuda_last_error()); return 2; }
        struct { unsigned N; } push = { N };
        uint32_t grid[3] = { 1, 1, 1 }, block[3] = { 256, 1, 1 }; CudaBuf* bufs[2] = { dl, dt };
        CudaCmd* cmd = cuda_begin_command(c); cuda_dispatch(cmd, p, grid, block, bufs, 2, &push, sizeof push, 0); cuda_commit_and_wait(cmd);
        unsigned gt; cuda_download(c, dt, &gt, 4);
        int ok = (gt == ref); all_ok &= ok;
        printf("argmax [N=%u]: ref=%u gpu=%u -> %s\n", N, ref, gt, ok ? "PASS" : "FAIL");
        cuda_free_buffer(dl); cuda_free_buffer(dt); cuda_free_pipeline(p); free(logits);
    }

    // ===== Test 13: moe_weighted_acc =====
    {
        const unsigned N = 2048, n_used = 8;
        float* a0 = malloc((size_t)N * 4); float* b = malloc((size_t)n_used * N * 4);
        float* aref = malloc((size_t)N * 4); float* agpu = malloc((size_t)N * 4);
        unsigned routing[16]; float wts[8]; float wsum = 0;
        for (unsigned i = 0; i < N; i++) a0[i] = frand();
        for (unsigned j = 0; j < n_used * N; j++) b[j] = frand();
        for (unsigned j = 0; j < n_used; j++) { wts[j] = fabsf(frand()) + 0.1f; wsum += wts[j]; }
        for (unsigned j = 0; j < n_used; j++) { wts[j] /= wsum; routing[j] = j; memcpy(&routing[n_used + j], &wts[j], 4); }
        for (unsigned i = 0; i < N; i++) { float s = 0; for (unsigned j = 0; j < n_used; j++) s += wts[j] * b[(size_t)j * N + i]; aref[i] = a0[i] + s; }
        CudaBuf* da = cuda_create_buffer(c, (size_t)N * 4); CudaBuf* db = cuda_create_buffer(c, (size_t)n_used * N * 4); CudaBuf* dr = cuda_create_buffer(c, 16 * 4);
        cuda_upload(c, da, a0, (size_t)N * 4); cuda_upload(c, db, b, (size_t)n_used * N * 4); cuda_upload(c, dr, routing, 16 * 4);
        CudaPipe* p = cuda_create_pipeline(c, src, "moe_weighted_acc", NULL, 0);
        if (!p) { printf("FAIL moe_weighted_acc compile: %s\n", cuda_last_error()); return 2; }
        struct { unsigned N, nu, ss; } push = { N, n_used, N };
        uint32_t grid[3] = { (N + 255) / 256, 1, 1 }, block[3] = { 256, 1, 1 }; CudaBuf* bufs[3] = { da, db, dr };
        CudaCmd* cmd = cuda_begin_command(c); cuda_dispatch(cmd, p, grid, block, bufs, 3, &push, sizeof push, 0); cuda_commit_and_wait(cmd);
        cuda_download(c, da, agpu, (size_t)N * 4);
        float mr = 0; for (unsigned i = 0; i < N; i++) { float r = fabsf(aref[i] - agpu[i]) / (fabsf(aref[i]) + 1e-4f); if (r > mr) mr = r; }
        int ok = mr < 1e-3f; all_ok &= ok;
        printf("moe_weighted_acc [N=%u n_used=%u]: max_rel_err=%.2e -> %s\n", N, n_used, mr, ok ? "PASS" : "FAIL");
        cuda_free_buffer(da); cuda_free_buffer(db); cuda_free_buffer(dr); cuda_free_pipeline(p);
        free(a0); free(b); free(aref); free(agpu);
    }

    printf("RESULT: %s\n", all_ok ? "ALL PASS" : "FAIL");
    cuda_destroy(c);
    free(src);
    return all_ok ? 0 : 1;
}
