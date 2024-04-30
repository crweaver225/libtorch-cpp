#pragma once

// @generated by torchgen/gen.py from NativeFunction.h

#include <c10/core/Scalar.h>
#include <c10/core/Storage.h>
#include <c10/core/TensorOptions.h>
#include <c10/util/Deprecated.h>
#include <c10/util/Optional.h>
#include <c10/core/QScheme.h>
#include <ATen/core/Reduction.h>
#include <ATen/core/Tensor.h>
#include <tuple>
#include <vector>


namespace at {
namespace native {
TORCH_API ::std::tuple<at::Tensor,at::Tensor,at::Tensor,at::Tensor> _efficient_attention_backward(const at::Tensor & grad_out_, const at::Tensor & query, const at::Tensor & key, const at::Tensor & value, const ::std::optional<at::Tensor> & bias, const at::Tensor & out, const ::std::optional<at::Tensor> & cu_seqlens_q, const ::std::optional<at::Tensor> & cu_seqlens_k, int64_t max_seqlen_q, int64_t max_seqlen_k, const at::Tensor & logsumexp, double dropout_p, const at::Tensor & philox_seed, const at::Tensor & philox_offset, int64_t custom_mask_type, bool bias_requires_grad, ::std::optional<double> scale=::std::nullopt, ::std::optional<int64_t> num_splits_key=::std::nullopt, ::std::optional<int64_t> window_size=::std::nullopt);
} // namespace native
} // namespace at
