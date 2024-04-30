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

namespace compositeexplicitautograd {

TORCH_API at::Tensor randperm(int64_t n, at::TensorOptions options=at::kLong);
TORCH_API at::Tensor randperm(int64_t n, ::std::optional<at::ScalarType> dtype, ::std::optional<at::Layout> layout, ::std::optional<at::Device> device, ::std::optional<bool> pin_memory);
TORCH_API at::Tensor randperm_symint(c10::SymInt n, at::TensorOptions options=at::kLong);
TORCH_API at::Tensor randperm_symint(c10::SymInt n, ::std::optional<at::ScalarType> dtype, ::std::optional<at::Layout> layout, ::std::optional<at::Device> device, ::std::optional<bool> pin_memory);
TORCH_API at::Tensor & randperm_out(at::Tensor & out, int64_t n);
TORCH_API at::Tensor & randperm_outf(int64_t n, at::Tensor & out);
TORCH_API at::Tensor & randperm_symint_out(at::Tensor & out, c10::SymInt n);
TORCH_API at::Tensor & randperm_symint_outf(c10::SymInt n, at::Tensor & out);
TORCH_API at::Tensor randperm(int64_t n, ::std::optional<at::Generator> generator, at::TensorOptions options=at::kLong);
TORCH_API at::Tensor randperm(int64_t n, ::std::optional<at::Generator> generator, ::std::optional<at::ScalarType> dtype, ::std::optional<at::Layout> layout, ::std::optional<at::Device> device, ::std::optional<bool> pin_memory);
TORCH_API at::Tensor randperm_symint(c10::SymInt n, ::std::optional<at::Generator> generator, at::TensorOptions options=at::kLong);
TORCH_API at::Tensor randperm_symint(c10::SymInt n, ::std::optional<at::Generator> generator, ::std::optional<at::ScalarType> dtype, ::std::optional<at::Layout> layout, ::std::optional<at::Device> device, ::std::optional<bool> pin_memory);

} // namespace compositeexplicitautograd
} // namespace at
