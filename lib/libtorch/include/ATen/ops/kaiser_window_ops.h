#pragma once

// @generated by torchgen/gen.py from Operator.h

#include <tuple>
#include <vector>

// Forward declarations of any types needed in the operator signatures.
// We can't directly include these classes because it will cause circular include dependencies.
// This file is included by TensorBody.h, which defines the Tensor class.
#include <ATen/core/ATen_fwd.h>

namespace at {
namespace _ops {


struct TORCH_API kaiser_window {
  using schema = at::Tensor (int64_t, ::std::optional<at::ScalarType>, ::std::optional<at::Layout>, ::std::optional<at::Device>, ::std::optional<bool>);
  using ptr_schema = schema*;
  // See Note [static constexpr char* members for windows NVCC]
  STATIC_CONSTEXPR_STR_INL_EXCEPT_WIN_CUDA(name, "aten::kaiser_window")
  STATIC_CONSTEXPR_STR_INL_EXCEPT_WIN_CUDA(overload_name, "")
  STATIC_CONSTEXPR_STR_INL_EXCEPT_WIN_CUDA(schema_str, "kaiser_window(int window_length, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None) -> Tensor")
  static at::Tensor call(int64_t window_length, ::std::optional<at::ScalarType> dtype, ::std::optional<at::Layout> layout, ::std::optional<at::Device> device, ::std::optional<bool> pin_memory);
  static at::Tensor redispatch(c10::DispatchKeySet dispatchKeySet, int64_t window_length, ::std::optional<at::ScalarType> dtype, ::std::optional<at::Layout> layout, ::std::optional<at::Device> device, ::std::optional<bool> pin_memory);
};

struct TORCH_API kaiser_window_periodic {
  using schema = at::Tensor (int64_t, bool, ::std::optional<at::ScalarType>, ::std::optional<at::Layout>, ::std::optional<at::Device>, ::std::optional<bool>);
  using ptr_schema = schema*;
  // See Note [static constexpr char* members for windows NVCC]
  STATIC_CONSTEXPR_STR_INL_EXCEPT_WIN_CUDA(name, "aten::kaiser_window")
  STATIC_CONSTEXPR_STR_INL_EXCEPT_WIN_CUDA(overload_name, "periodic")
  STATIC_CONSTEXPR_STR_INL_EXCEPT_WIN_CUDA(schema_str, "kaiser_window.periodic(int window_length, bool periodic, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None) -> Tensor")
  static at::Tensor call(int64_t window_length, bool periodic, ::std::optional<at::ScalarType> dtype, ::std::optional<at::Layout> layout, ::std::optional<at::Device> device, ::std::optional<bool> pin_memory);
  static at::Tensor redispatch(c10::DispatchKeySet dispatchKeySet, int64_t window_length, bool periodic, ::std::optional<at::ScalarType> dtype, ::std::optional<at::Layout> layout, ::std::optional<at::Device> device, ::std::optional<bool> pin_memory);
};

struct TORCH_API kaiser_window_beta {
  using schema = at::Tensor (int64_t, bool, double, ::std::optional<at::ScalarType>, ::std::optional<at::Layout>, ::std::optional<at::Device>, ::std::optional<bool>);
  using ptr_schema = schema*;
  // See Note [static constexpr char* members for windows NVCC]
  STATIC_CONSTEXPR_STR_INL_EXCEPT_WIN_CUDA(name, "aten::kaiser_window")
  STATIC_CONSTEXPR_STR_INL_EXCEPT_WIN_CUDA(overload_name, "beta")
  STATIC_CONSTEXPR_STR_INL_EXCEPT_WIN_CUDA(schema_str, "kaiser_window.beta(int window_length, bool periodic, float beta, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None) -> Tensor")
  static at::Tensor call(int64_t window_length, bool periodic, double beta, ::std::optional<at::ScalarType> dtype, ::std::optional<at::Layout> layout, ::std::optional<at::Device> device, ::std::optional<bool> pin_memory);
  static at::Tensor redispatch(c10::DispatchKeySet dispatchKeySet, int64_t window_length, bool periodic, double beta, ::std::optional<at::ScalarType> dtype, ::std::optional<at::Layout> layout, ::std::optional<at::Device> device, ::std::optional<bool> pin_memory);
};

struct TORCH_API kaiser_window_out {
  using schema = at::Tensor & (int64_t, at::Tensor &);
  using ptr_schema = schema*;
  // See Note [static constexpr char* members for windows NVCC]
  STATIC_CONSTEXPR_STR_INL_EXCEPT_WIN_CUDA(name, "aten::kaiser_window")
  STATIC_CONSTEXPR_STR_INL_EXCEPT_WIN_CUDA(overload_name, "out")
  STATIC_CONSTEXPR_STR_INL_EXCEPT_WIN_CUDA(schema_str, "kaiser_window.out(int window_length, *, Tensor(a!) out) -> Tensor(a!)")
  static at::Tensor & call(int64_t window_length, at::Tensor & out);
  static at::Tensor & redispatch(c10::DispatchKeySet dispatchKeySet, int64_t window_length, at::Tensor & out);
};

struct TORCH_API kaiser_window_periodic_out {
  using schema = at::Tensor & (int64_t, bool, at::Tensor &);
  using ptr_schema = schema*;
  // See Note [static constexpr char* members for windows NVCC]
  STATIC_CONSTEXPR_STR_INL_EXCEPT_WIN_CUDA(name, "aten::kaiser_window")
  STATIC_CONSTEXPR_STR_INL_EXCEPT_WIN_CUDA(overload_name, "periodic_out")
  STATIC_CONSTEXPR_STR_INL_EXCEPT_WIN_CUDA(schema_str, "kaiser_window.periodic_out(int window_length, bool periodic, *, Tensor(a!) out) -> Tensor(a!)")
  static at::Tensor & call(int64_t window_length, bool periodic, at::Tensor & out);
  static at::Tensor & redispatch(c10::DispatchKeySet dispatchKeySet, int64_t window_length, bool periodic, at::Tensor & out);
};

struct TORCH_API kaiser_window_beta_out {
  using schema = at::Tensor & (int64_t, bool, double, at::Tensor &);
  using ptr_schema = schema*;
  // See Note [static constexpr char* members for windows NVCC]
  STATIC_CONSTEXPR_STR_INL_EXCEPT_WIN_CUDA(name, "aten::kaiser_window")
  STATIC_CONSTEXPR_STR_INL_EXCEPT_WIN_CUDA(overload_name, "beta_out")
  STATIC_CONSTEXPR_STR_INL_EXCEPT_WIN_CUDA(schema_str, "kaiser_window.beta_out(int window_length, bool periodic, float beta, *, Tensor(a!) out) -> Tensor(a!)")
  static at::Tensor & call(int64_t window_length, bool periodic, double beta, at::Tensor & out);
  static at::Tensor & redispatch(c10::DispatchKeySet dispatchKeySet, int64_t window_length, bool periodic, double beta, at::Tensor & out);
};

}} // namespace at::_ops
