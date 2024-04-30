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
TORCH_API at::Tensor fft_irfft_symint(const at::Tensor & self, ::std::optional<c10::SymInt> n=::std::nullopt, int64_t dim=-1, ::std::optional<c10::string_view> norm=::std::nullopt);
TORCH_API at::Tensor & fft_irfft_symint_out(const at::Tensor & self, ::std::optional<c10::SymInt> n, int64_t dim, ::std::optional<c10::string_view> norm, at::Tensor & out);
} // namespace native
} // namespace at
