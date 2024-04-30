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
TORCH_API ::std::tuple<at::Tensor,at::Tensor,at::Tensor,at::Tensor> _scaled_dot_product_efficient_attention_cuda(const at::Tensor & query, const at::Tensor & key, const at::Tensor & value, const ::std::optional<at::Tensor> & attn_bias, bool compute_log_sumexp, double dropout_p=0.0, bool is_causal=false, ::std::optional<double> scale=::std::nullopt);
TORCH_API ::std::tuple<at::Tensor,at::Tensor,at::Tensor,at::Tensor> _scaled_dot_product_efficient_attention_nestedtensor_cuda(const at::Tensor & query, const at::Tensor & key, const at::Tensor & value, const ::std::optional<at::Tensor> & attn_bias, bool compute_log_sumexp, double dropout_p=0.0, bool is_causal=false, ::std::optional<double> scale=::std::nullopt);
} // namespace native
} // namespace at
