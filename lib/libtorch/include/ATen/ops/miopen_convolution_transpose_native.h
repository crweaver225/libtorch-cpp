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
TORCH_API at::Tensor & miopen_convolution_transpose_out_symint(const at::Tensor & self, const at::Tensor & weight, const ::std::optional<at::Tensor> & bias, c10::SymIntArrayRef padding, c10::SymIntArrayRef output_padding, c10::SymIntArrayRef stride, c10::SymIntArrayRef dilation, c10::SymInt groups, bool benchmark, bool deterministic, at::Tensor & out);
TORCH_API at::Tensor miopen_convolution_transpose(const at::Tensor & self, const at::Tensor & weight, const ::std::optional<at::Tensor> & bias, at::IntArrayRef padding, at::IntArrayRef output_padding, at::IntArrayRef stride, at::IntArrayRef dilation, int64_t groups, bool benchmark, bool deterministic);
} // namespace native
} // namespace at
