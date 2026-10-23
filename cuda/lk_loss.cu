// lk_loss.cu -- fused full-vocab elementwise kernels for the MTP LK losses.
//
// Pure CUDA (no torch headers -- see cuda/rwkv7_clampw.cu for the repo
// convention); torch binding and checks live in lk_loss.cpp.
//
// The eager PyTorch LK backward runs ~15 full-vocab fp32 elementwise passes
// per depth (log_softmax, exp, sign, mul/addcmul, div, ...), each streaming
// ~1 GiB at ctx 4096 -- that is what makes LK ~30% slower than the top-k
// distill path.  These kernels collapse the chains into one block-per-row
// sweep over bf16 logits (the raw cuBLAS GEMM output, no .float()
// materialisation); the row working set (V x 2 bytes, 128 KiB at V=65536)
// stays L2-resident between passes.
//
// Math (lk-loss.md, identical to the eager path in mtp.py):
//   q = softmax(draft logits),  p = softmax(teacher logits)
//   alpha_row = sum min(p, q) = 1 - TV(p, q),   TV = 0.5 * sum|q - p|
//   hybrid:   dL/dlogits = lam*(q - p) + (1-lam)*0.5*q*(s - E_q[s]),
//             s = sign(q - p), E_q[s] = sum(q * s)
//   lk_alpha: dL/dlogits = 0.5*q*(s - E_q[s]) / (alpha_row + eps)
//
// All reductions are per-row fp32 block reductions (256 threads); accurate
// expf/logf, no fast-math intrinsics, so results match the eager path to
// fp32 accumulation-order noise.

#include <cuda_bf16.h>
#include <math.h>

namespace {

constexpr int LK_BLOCK = 256;

__device__ __forceinline__ float block_reduce(float v, float* shared,
                                              bool is_max) {
    const int tid = threadIdx.x;
    const int T = blockDim.x;
    shared[tid] = v;
    __syncthreads();
    for (int off = T >> 1; off > 0; off >>= 1) {
        if (tid < off) {
            shared[tid] = is_max ? fmaxf(shared[tid], shared[tid + off])
                                 : (shared[tid] + shared[tid + off]);
        }
        __syncthreads();
    }
    float r = shared[0];
    __syncthreads();   // shared[] is reused by later reductions
    return r;
}

// Shared prologue for one row: block max and log-sum-exp of both logit rows.
__device__ __forceinline__ void row_logsumexp(
    const __nv_bfloat16* __restrict__ q,
    const __nv_bfloat16* __restrict__ p,
    int V, float* shared, float* logZq, float* logZp) {
    const int tid = threadIdx.x;
    const int T = blockDim.x;
    float maxq = -INFINITY, maxp = -INFINITY;
    for (int i = tid; i < V; i += T) {
        maxq = fmaxf(maxq, __bfloat162float(q[i]));
        maxp = fmaxf(maxp, __bfloat162float(p[i]));
    }
    maxq = block_reduce(maxq, shared, true);
    maxp = block_reduce(maxp, shared, true);

    float sumq = 0.f, sump = 0.f;
    for (int i = tid; i < V; i += T) {
        sumq += expf(__bfloat162float(q[i]) - maxq);
        sump += expf(__bfloat162float(p[i]) - maxp);
    }
    sumq = block_reduce(sumq, shared, false);
    sump = block_reduce(sump, shared, false);

    *logZq = maxq + logf(sumq);
    *logZp = maxp + logf(sump);
}

// Forward statistics: per-row alpha = 1 - TV and (hybrid only) KL(p||q).
__global__ void lk_forward_stats_kernel(
    const __nv_bfloat16* __restrict__ q_logits,   // (n, V)
    const __nv_bfloat16* __restrict__ p_logits,   // (n, V)
    float* __restrict__ alpha_out,                // (n,)
    float* __restrict__ kl_out,                   // (n,)  untouched if !hybrid
    int V, int hybrid) {
    extern __shared__ float shared[];
    const int row = blockIdx.x;
    const __nv_bfloat16* q = q_logits + (size_t)row * V;
    const __nv_bfloat16* p = p_logits + (size_t)row * V;
    const int tid = threadIdx.x;
    const int T = blockDim.x;

    float logZq, logZp;
    row_logsumexp(q, p, V, shared, &logZq, &logZp);

    float tv = 0.f, kl = 0.f;
    for (int i = tid; i < V; i += T) {
        const float qv = __bfloat162float(q[i]);
        const float pv = __bfloat162float(p[i]);
        const float qq = expf(qv - logZq);
        const float pp = expf(pv - logZp);
        const float d = qq - pp;
        tv += fmaxf(d, 0.f);                       // sum relu(q-p) == TV
        if (hybrid) {
            kl += pp * ((pv - logZp) - (qv - logZq));   // p*(logp - logq)
        }
    }
    tv = block_reduce(tv, shared, false);
    float kl_r = 0.f;
    if (hybrid) kl_r = block_reduce(kl, shared, false);

    if (tid == 0) {
        alpha_out[row] = 1.f - tv;
        if (hybrid) kl_out[row] = kl_r;
    }
}

// Backward stage 1: per-row scalars E_q[s], alpha, and the log-softmax
// normalisers (reused by the grad kernel so it needs no extra passes).
__global__ void lk_row_scalars_kernel(
    const __nv_bfloat16* __restrict__ q_logits,   // (n, V)
    const __nv_bfloat16* __restrict__ p_logits,   // (n, V)
    float* __restrict__ es_out,                   // (n,)  E_q[s]
    float* __restrict__ alpha_out,                // (n,)  1 - TV
    float* __restrict__ logZq_out,                // (n,)
    float* __restrict__ logZp_out,                // (n,)
    int V) {
    extern __shared__ float shared[];
    const int row = blockIdx.x;
    const __nv_bfloat16* q = q_logits + (size_t)row * V;
    const __nv_bfloat16* p = p_logits + (size_t)row * V;
    const int tid = threadIdx.x;
    const int T = blockDim.x;

    float logZq, logZp;
    row_logsumexp(q, p, V, shared, &logZq, &logZp);

    float es = 0.f, tv = 0.f;
    for (int i = tid; i < V; i += T) {
        const float qq = expf(__bfloat162float(q[i]) - logZq);
        const float pp = expf(__bfloat162float(p[i]) - logZp);
        const float d = qq - pp;
        const float s = copysignf(1.f, d);         // sign(q - p); 0 at d == 0
        es += qq * s;
        tv += fabsf(d);
    }
    es = block_reduce(es, shared, false);
    tv = block_reduce(tv, shared, false);

    if (tid == 0) {
        es_out[row] = es;
        alpha_out[row] = 1.f - 0.5f * tv;
        logZq_out[row] = logZq;
        logZp_out[row] = logZp;
    }
}

// Backward stage 2: the (n, V) logit gradient.  Rows whose target is
// ignore_index are written as zeros (excluded from the loss upstream).
__global__ void lk_grad_kernel(
    const __nv_bfloat16* __restrict__ q_logits,   // (n, V)
    const __nv_bfloat16* __restrict__ p_logits,   // (n, V)
    const float* __restrict__ es_in,              // (n,)
    const float* __restrict__ alpha_in,           // (n,)
    const long long* __restrict__ targets,        // (n,)
    __nv_bfloat16* __restrict__ g_out,            // (n, V)
    long long ignore_index, int V, int hybrid,
    float lam, float eps) {
    extern __shared__ float shared[];
    const int row = blockIdx.x;
    const __nv_bfloat16* q = q_logits + (size_t)row * V;
    const __nv_bfloat16* p = p_logits + (size_t)row * V;
    __nv_bfloat16* g = g_out + (size_t)row * V;
    const int tid = threadIdx.x;
    const int T = blockDim.x;

    if (targets[row] == ignore_index) {
        for (int i = tid; i < V; i += T) g[i] = __float2bfloat16_rn(0.f);
        return;
    }

    float logZq, logZp;
    row_logsumexp(q, p, V, shared, &logZq, &logZp);

    const float es = es_in[row];
    const float alpha = alpha_in[row];
    const float w_tv = hybrid ? (0.5f * (1.f - lam)) : 0.5f;
    const float denom = hybrid ? 1.f : (alpha + eps);

    for (int i = tid; i < V; i += T) {
        const float qv = __bfloat162float(q[i]);
        const float pv = __bfloat162float(p[i]);
        const float qq = expf(qv - logZq);
        const float pp = expf(pv - logZp);
        const float s = copysignf(1.f, qq - pp);
        float gval = w_tv * qq * (s - es);
        if (hybrid) gval += lam * (qq - pp);
        else        gval /= denom;
        g[i] = __float2bfloat16_rn(gval);
    }
}

}  // namespace

// ---------------------------------------------------------------------------
// Host launchers (bound by lk_loss.cpp)
// ---------------------------------------------------------------------------

void lk_forward_stats_cuda(const void* q_logits, const void* p_logits,
                           float* alpha_out, float* kl_out,
                           long long n, long long V, int hybrid) {
    if (n == 0) return;
    lk_forward_stats_kernel<<<(unsigned)n, LK_BLOCK, LK_BLOCK * sizeof(float)>>>(
        (const __nv_bfloat16*)q_logits, (const __nv_bfloat16*)p_logits,
        alpha_out, kl_out, (int)V, hybrid);
}

void lk_row_scalars_cuda(const void* q_logits, const void* p_logits,
                         float* es_out, float* alpha_out,
                         float* logZq_out, float* logZp_out,
                         long long n, long long V) {
    if (n == 0) return;
    lk_row_scalars_kernel<<<(unsigned)n, LK_BLOCK, LK_BLOCK * sizeof(float)>>>(
        (const __nv_bfloat16*)q_logits, (const __nv_bfloat16*)p_logits,
        es_out, alpha_out, logZq_out, logZp_out, (int)V);
}

void lk_grad_cuda(const void* q_logits, const void* p_logits,
                  const float* es, const float* alpha,
                  const long long* targets, void* g_out,
                  long long ignore_index, long long n, long long V,
                  int hybrid, float lam, float eps) {
    if (n == 0) return;
    lk_grad_kernel<<<(unsigned)n, LK_BLOCK, LK_BLOCK * sizeof(float)>>>(
        (const __nv_bfloat16*)q_logits, (const __nv_bfloat16*)p_logits,
        es, alpha, targets, (__nv_bfloat16*)g_out,
        ignore_index, (int)V, hybrid, lam, eps);
}
