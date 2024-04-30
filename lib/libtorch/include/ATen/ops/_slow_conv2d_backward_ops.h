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


struct TORCH_API _slow_conv2d_backward_grad_input {
  using schema = ::std::tuple<at::Tensor &,at::Tensor &,at::Tensor &> (const at::Tensor &, const at::Tensor &, const at::Tensor &, c10::SymIntArrayRef, c10::SymIntArrayRef, c10::SymIntArrayRef, at::Tensor &, at::Tensor &, at::Tensor &);
  using ptr_schema = schema*;
  // See Note [static constexpr char* members for windows NVCC]
  STATIC_CONSTEXPR_STR_INL_EXCEPT_WIN_CUDA(name, "aten::_slow_conv2d_backward")
  STATIC_CONSTEXPR_STR_INL_EXCEPT_WIN_CUDA(overload_name, "grad_input")
  STATIC_CONSTEXPR_STR_INL_EXCEPT_WIN_CUDA(schema_str, "_slow_conv2d_backward.grad_input(Tensor grad_output, Tensor self, Tensor weight, SymInt[2] kernel_size, SymInt[2] stride, SymInt[2] padding, *, Tensor(a!) grad_input, Tensor(b!) grad_weight, Tensor(c!) grad_bias) -> (Tensor(a!), Tensor(b!), Tensor(c!))")
  static ::std::tuple<at::Tensor &,at::Tensor &,at::Tensor &> call(const at::Tensor & grad_output, const at::Tensor & self, const at::Tensor & weight, c10::SymIntArrayRef kernel_size, c10::SymIntArrayRef stride, c10::SymIntArrayRef padding, at::Tensor & grad_input, at::Tensor & grad_weight, at::Tensor & grad_bias);
  static ::std::tuple<at::Tensor &,at::Tensor &,at::Tensor &> redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & grad_output, const at::Tensor & self, const at::Tensor & weight, c10::SymIntArrayRef kernel_size, c10::SymIntArrayRef stride, c10::SymIntArrayRef padding, at::Tensor & grad_input, at::Tensor & grad_weight, at::Tensor & grad_bias);
};

struct TORCH_API _slow_conv2d_backward_output_mask {
  using schema = ::std::tuple<at::Tensor,at::Tensor,at::Tensor> (const at::Tensor &, const at::Tensor &, const at::Tensor &, c10::SymIntArrayRef, c10::SymIntArrayRef, c10::SymIntArrayRef, ::std::array<bool,3>);
  using ptr_schema = schema*;
  // See Note [static constexpr char* members for windows NVCC]
  STATIC_CONSTEXPR_STR_INL_EXCEPT_WIN_CUDA(name, "aten::_slow_conv2d_backward")
  STATIC_CONSTEXPR_STR_INL_EXCEPT_WIN_CUDA(overload_name, "output_mask")
  STATIC_CONSTEXPR_STR_INL_EXCEPT_WIN_CUDA(schema_str, "_slow_conv2d_backward.output_mask(Tensor grad_output, Tensor self, Tensor weight, SymInt[2] kernel_size, SymInt[2] stride, SymInt[2] padding, bool[3] output_mask) -> (Tensor grad_input, Tensor grad_weight, Tensor grad_bias)")
  static ::std::tuple<at::Tensor,at::Tensor,at::Tensor> call(const at::Tensor & grad_output, const at::Tensor & self, const at::Tensor & weight, c10::SymIntArrayRef kernel_size, c10::SymIntArrayRef stride, c10::SymIntArrayRef padding, ::std::array<bool,3> output_mask);
  static ::std::tuple<at::Tensor,at::Tensor,at::Tensor> redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & grad_output, const at::Tensor & self, const at::Tensor & weight, c10::SymIntArrayRef kernel_size, c10::SymIntArrayRef stride, c10::SymIntArrayRef padding, ::std::array<bool,3> output_mask);
};

struct TORCH_API _slow_conv2d_backward_output_mask_out {
  using schema = ::std::tuple<at::Tensor &,at::Tensor &,at::Tensor &> (const at::Tensor &, const at::Tensor &, const at::Tensor &, c10::SymIntArrayRef, c10::SymIntArrayRef, c10::SymIntArrayRef, ::std::array<bool,3>, at::Tensor &, at::Tensor &, at::Tensor &);
  using ptr_schema = schema*;
  // See Note [static constexpr char* members for windows NVCC]
  STATIC_CONSTEXPR_STR_INL_EXCEPT_WIN_CUDA(name, "aten::_slow_conv2d_backward")
  STATIC_CONSTEXPR_STR_INL_EXCEPT_WIN_CUDA(overload_name, "output_mask_out")
  STATIC_CONSTEXPR_STR_INL_EXCEPT_WIN_CUDA(schema_str, "_slow_conv2d_backward.output_mask_out(Tensor grad_output, Tensor self, Tensor weight, SymInt[2] kernel_size, SymInt[2] stride, SymInt[2] padding, bool[3] output_mask, *, Tensor(a!) out0, Tensor(b!) out1, Tensor(c!) out2) -> (Tensor(a!), Tensor(b!), Tensor(c!))")
  static ::std::tuple<at::Tensor &,at::Tensor &,at::Tensor &> call(const at::Tensor & grad_output, const at::Tensor & self, const at::Tensor & weight, c10::SymIntArrayRef kernel_size, c10::SymIntArrayRef stride, c10::SymIntArrayRef padding, ::std::array<bool,3> output_mask, at::Tensor & out0, at::Tensor & out1, at::Tensor & out2);
  static ::std::tuple<at::Tensor &,at::Tensor &,at::Tensor &> redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & grad_output, const at::Tensor & self, const at::Tensor & weight, c10::SymIntArrayRef kernel_size, c10::SymIntArrayRef stride, c10::SymIntArrayRef padding, ::std::array<bool,3> output_mask, at::Tensor & out0, at::Tensor & out1, at::Tensor & out2);
};

}} // namespace at::_ops
