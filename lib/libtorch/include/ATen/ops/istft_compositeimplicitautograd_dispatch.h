#pragma once
// @generated by torchgen/gen.py from DispatchKeyFunction.h

// NB: The implementing C++ file is RegisterDispatchKey.cpp

// The only #includes we need are for custom classes that have defaults in the C++ API
#include <c10/core/MemoryFormat.h>
#include <c10/core/Scalar.h>
#include <ATen/core/Reduction.h>

// Forward declarations of any types needed in the operator signatures.
// We can't directly include these classes because it will cause circular include dependencies.
// This file is included by TensorBody.h, which defines the Tensor class.
#include <ATen/core/ATen_fwd.h>

namespace at {

namespace compositeimplicitautograd {

TORCH_API at::Tensor istft(const at::Tensor & self, int64_t n_fft, ::std::optional<int64_t> hop_length=::std::nullopt, ::std::optional<int64_t> win_length=::std::nullopt, const ::std::optional<at::Tensor> & window={}, bool center=true, bool normalized=false, ::std::optional<bool> onesided=::std::nullopt, ::std::optional<int64_t> length=::std::nullopt, bool return_complex=false);

} // namespace compositeimplicitautograd
} // namespace at