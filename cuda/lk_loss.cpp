// lk_loss.cpp -- torch binding for the fused LK kernels (lk_loss.cu).
//
// Pure-CUDA .cu + torch-binding .cpp is the repo convention (see
// cuda/rwkv7_clampw.cpp): nvcc never sees the torch/CCCL headers, which keeps
// the MSVC/CUDA-13 preprocessor happy.  All tensors must be contiguous CUDA
// bf16 (the raw cuBLAS GEMM outputs, exactly what the eager path upcasts).

#include <torch/extension.h>
#include <vector>

void lk_forward_stats_cuda(const void* q_logits, const void* p_logits,
                           float* alpha_out, float* kl_out,
                           long long n, long long V, int hybrid);
void lk_row_scalars_cuda(const void* q_logits, const void* p_logits,
                         float* es_out, float* alpha_out,
                         float* logZq_out, float* logZp_out,
                         long long n, long long V);
void lk_grad_cuda(const void* q_logits, const void* p_logits,
                  const float* es, const float* alpha,
                  const long long* targets, void* g_out,
                  long long ignore_index, long long n, long long V,
                  int hybrid, float lam, float eps);

#define LK_CHECK_INPUT(x)                                       \
    TORCH_CHECK(x.is_cuda() && x.is_contiguous() &&             \
                x.scalar_type() == torch::kBFloat16,            \
                #x " must be a contiguous CUDA bf16 tensor (got cuda=" +   \
                std::to_string(x.is_cuda()) + " contig=" +                 \
                std::to_string(x.is_contiguous()) + " dtype=" +            \
                std::to_string((int)x.scalar_type()) + " shape=" +         \
                std::to_string(x.size(0)) + "x" +                          \
                std::to_string(x.dim() > 1 ? x.size(1) : 0) + ")")

static void check_pair(const torch::Tensor& q, const torch::Tensor& p) {
    LK_CHECK_INPUT(q);
    LK_CHECK_INPUT(p);
    TORCH_CHECK(q.size(0) == p.size(0) && q.size(1) == p.size(1),
                "q_logits and p_logits must have the same shape");
}

std::vector<torch::Tensor> lk_forward_stats(torch::Tensor q_logits,
                                            torch::Tensor p_logits,
                                            bool hybrid, double eps) {
    check_pair(q_logits, p_logits);
    const auto n = q_logits.size(0);
    const auto V = q_logits.size(1);
    auto alpha = torch::empty({n}, q_logits.options().dtype(torch::kFloat32));
    auto kl = torch::zeros({n}, q_logits.options().dtype(torch::kFloat32));
    lk_forward_stats_cuda(q_logits.data_ptr(), p_logits.data_ptr(),
                          alpha.data_ptr<float>(), kl.data_ptr<float>(),
                          (long long)n, (long long)V, hybrid ? 1 : 0);
    (void)eps;   // -log(alpha) is computed in Python (vectorised, (n,) only)
    return {alpha, kl};
}

std::vector<torch::Tensor> lk_row_scalars(torch::Tensor q_logits,
                                          torch::Tensor p_logits) {
    check_pair(q_logits, p_logits);
    const auto n = q_logits.size(0);
    const auto V = q_logits.size(1);
    auto opts = q_logits.options().dtype(torch::kFloat32);
    auto es = torch::empty({n}, opts);
    auto alpha = torch::empty({n}, opts);
    auto logZq = torch::empty({n}, opts);
    auto logZp = torch::empty({n}, opts);
    lk_row_scalars_cuda(q_logits.data_ptr(), p_logits.data_ptr(),
                        es.data_ptr<float>(), alpha.data_ptr<float>(),
                        logZq.data_ptr<float>(), logZp.data_ptr<float>(),
                        (long long)n, (long long)V);
    return {es, alpha, logZq, logZp};
}

torch::Tensor lk_grad(torch::Tensor q_logits, torch::Tensor p_logits,
                      torch::Tensor es, torch::Tensor alpha,
                      torch::Tensor targets, int64_t ignore_index,
                      bool hybrid, double lam, double eps) {
    check_pair(q_logits, p_logits);
    TORCH_CHECK(es.is_cuda() && es.is_contiguous() &&
                es.scalar_type() == torch::kFloat32, "es must be CUDA fp32");
    TORCH_CHECK(alpha.is_cuda() && alpha.is_contiguous() &&
                alpha.scalar_type() == torch::kFloat32,
                "alpha must be CUDA fp32");
    TORCH_CHECK(targets.is_cuda() && targets.is_contiguous() &&
                targets.scalar_type() == torch::kLong,
                "targets must be CUDA int64");
    TORCH_CHECK(targets.size(0) == q_logits.size(0),
                "targets rows must match logits rows");
    const auto n = q_logits.size(0);
    const auto V = q_logits.size(1);
    auto g = torch::empty_like(q_logits);
    lk_grad_cuda(q_logits.data_ptr(), p_logits.data_ptr(),
                 es.data_ptr<float>(), alpha.data_ptr<float>(),
                 (const long long*)targets.data_ptr<int64_t>(), g.data_ptr(),
                 (long long)ignore_index, (long long)n, (long long)V,
                 hybrid ? 1 : 0, (float)lam, (float)eps);
    return g;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("lk_forward_stats", &lk_forward_stats, "LK forward per-row stats");
    m.def("lk_row_scalars", &lk_row_scalars, "LK backward per-row scalars");
    m.def("lk_grad", &lk_grad, "LK backward logit gradient");
}
