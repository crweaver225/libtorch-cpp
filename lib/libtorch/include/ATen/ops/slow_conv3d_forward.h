#pragma once

// @generated by torchgen/gen.py from Function.h

#include <ATen/Context.h>
#include <ATen/DeviceGuard.h>
#include <ATen/TensorUtils.h>
#include <ATen/TracerMode.h>
#include <ATen/core/Generator.h>
#include <ATen/core/Reduction.h>
#include <ATen/core/Tensor.h>
#include <c10/core/Scalar.h>
#include <c10/core/Storage.h>
#include <c10/core/TensorOptions.h>
#include <c10/util/Deprecated.h>
#include <c10/util/Optional.h>



#include <ATen/ops/slow_conv3d_forward_ops.h>

namespace at {


// aten::slow_conv3d_forward.output(Tensor self, Tensor weight, SymInt[3] kernel_size, Tensor? bias, SymInt[3] stride, SymInt[3] padding, *, Tensor(a!) output) -> Tensor(a!)
inline at::Tensor & slow_conv3d_forward_out(at::Tensor & output, const at::Tensor & self, const at::Tensor & weight, at::IntArrayRef kernel_size, const ::std::optional<at::Tensor> & bias, at::IntArrayRef stride, at::IntArrayRef padding) {
    return at::_ops::slow_conv3d_forward_output::call(self, weight, c10::fromIntArrayRefSlow(kernel_size), bias, c10::fromIntArrayRefSlow(stride), c10::fromIntArrayRefSlow(padding), output);
}
namespace symint {
  template <typename T, typename = std::enable_if_t<std::is_same<T, int64_t>::value>>
  at::Tensor & slow_conv3d_forward_out(at::Tensor & output, const at::Tensor & self, const at::Tensor & weight, at::IntArrayRef kernel_size, const ::std::optional<at::Tensor> & bias, at::IntArrayRef stride, at::IntArrayRef padding) {
    return at::_ops::slow_conv3d_forward_output::call(self, weight, c10::fromIntArrayRefSlow(kernel_size), bias, c10::fromIntArrayRefSlow(stride), c10::fromIntArrayRefSlow(padding), output);
  }
}

// aten::slow_conv3d_forward.output(Tensor self, Tensor weight, SymInt[3] kernel_size, Tensor? bias, SymInt[3] stride, SymInt[3] padding, *, Tensor(a!) output) -> Tensor(a!)
inline at::Tensor & slow_conv3d_forward_outf(const at::Tensor & self, const at::Tensor & weight, at::IntArrayRef kernel_size, const ::std::optional<at::Tensor> & bias, at::IntArrayRef stride, at::IntArrayRef padding, at::Tensor & output) {
    return at::_ops::slow_conv3d_forward_output::call(self, weight, c10::fromIntArrayRefSlow(kernel_size), bias, c10::fromIntArrayRefSlow(stride), c10::fromIntArrayRefSlow(padding), output);
}
namespace symint {
  template <typename T, typename = std::enable_if_t<std::is_same<T, int64_t>::value>>
  at::Tensor & slow_conv3d_forward_outf(const at::Tensor & self, const at::Tensor & weight, at::IntArrayRef kernel_size, const ::std::optional<at::Tensor> & bias, at::IntArrayRef stride, at::IntArrayRef padding, at::Tensor & output) {
    return at::_ops::slow_conv3d_forward_output::call(self, weight, c10::fromIntArrayRefSlow(kernel_size), bias, c10::fromIntArrayRefSlow(stride), c10::fromIntArrayRefSlow(padding), output);
  }
}

// aten::slow_conv3d_forward.output(Tensor self, Tensor weight, SymInt[3] kernel_size, Tensor? bias, SymInt[3] stride, SymInt[3] padding, *, Tensor(a!) output) -> Tensor(a!)
inline at::Tensor & slow_conv3d_forward_symint_out(at::Tensor & output, const at::Tensor & self, const at::Tensor & weight, c10::SymIntArrayRef kernel_size, const ::std::optional<at::Tensor> & bias, c10::SymIntArrayRef stride, c10::SymIntArrayRef padding) {
    return at::_ops::slow_conv3d_forward_output::call(self, weight, kernel_size, bias, stride, padding, output);
}
namespace symint {
  template <typename T, typename = std::enable_if_t<std::is_same<T, c10::SymInt>::value>>
  at::Tensor & slow_conv3d_forward_out(at::Tensor & output, const at::Tensor & self, const at::Tensor & weight, c10::SymIntArrayRef kernel_size, const ::std::optional<at::Tensor> & bias, c10::SymIntArrayRef stride, c10::SymIntArrayRef padding) {
    return at::_ops::slow_conv3d_forward_output::call(self, weight, kernel_size, bias, stride, padding, output);
  }
}

// aten::slow_conv3d_forward.output(Tensor self, Tensor weight, SymInt[3] kernel_size, Tensor? bias, SymInt[3] stride, SymInt[3] padding, *, Tensor(a!) output) -> Tensor(a!)
inline at::Tensor & slow_conv3d_forward_symint_outf(const at::Tensor & self, const at::Tensor & weight, c10::SymIntArrayRef kernel_size, const ::std::optional<at::Tensor> & bias, c10::SymIntArrayRef stride, c10::SymIntArrayRef padding, at::Tensor & output) {
    return at::_ops::slow_conv3d_forward_output::call(self, weight, kernel_size, bias, stride, padding, output);
}
namespace symint {
  template <typename T, typename = std::enable_if_t<std::is_same<T, c10::SymInt>::value>>
  at::Tensor & slow_conv3d_forward_outf(const at::Tensor & self, const at::Tensor & weight, c10::SymIntArrayRef kernel_size, const ::std::optional<at::Tensor> & bias, c10::SymIntArrayRef stride, c10::SymIntArrayRef padding, at::Tensor & output) {
    return at::_ops::slow_conv3d_forward_output::call(self, weight, kernel_size, bias, stride, padding, output);
  }
}

// aten::slow_conv3d_forward(Tensor self, Tensor weight, SymInt[3] kernel_size, Tensor? bias, SymInt[3] stride, SymInt[3] padding) -> Tensor
inline at::Tensor slow_conv3d_forward(const at::Tensor & self, const at::Tensor & weight, at::IntArrayRef kernel_size, const ::std::optional<at::Tensor> & bias, at::IntArrayRef stride, at::IntArrayRef padding) {
    return at::_ops::slow_conv3d_forward::call(self, weight, c10::fromIntArrayRefSlow(kernel_size), bias, c10::fromIntArrayRefSlow(stride), c10::fromIntArrayRefSlow(padding));
}
namespace symint {
  template <typename T, typename = std::enable_if_t<std::is_same<T, int64_t>::value>>
  at::Tensor slow_conv3d_forward(const at::Tensor & self, const at::Tensor & weight, at::IntArrayRef kernel_size, const ::std::optional<at::Tensor> & bias, at::IntArrayRef stride, at::IntArrayRef padding) {
    return at::_ops::slow_conv3d_forward::call(self, weight, c10::fromIntArrayRefSlow(kernel_size), bias, c10::fromIntArrayRefSlow(stride), c10::fromIntArrayRefSlow(padding));
  }
}

// aten::slow_conv3d_forward(Tensor self, Tensor weight, SymInt[3] kernel_size, Tensor? bias, SymInt[3] stride, SymInt[3] padding) -> Tensor
inline at::Tensor slow_conv3d_forward_symint(const at::Tensor & self, const at::Tensor & weight, c10::SymIntArrayRef kernel_size, const ::std::optional<at::Tensor> & bias, c10::SymIntArrayRef stride, c10::SymIntArrayRef padding) {
    return at::_ops::slow_conv3d_forward::call(self, weight, kernel_size, bias, stride, padding);
}
namespace symint {
  template <typename T, typename = std::enable_if_t<std::is_same<T, c10::SymInt>::value>>
  at::Tensor slow_conv3d_forward(const at::Tensor & self, const at::Tensor & weight, c10::SymIntArrayRef kernel_size, const ::std::optional<at::Tensor> & bias, c10::SymIntArrayRef stride, c10::SymIntArrayRef padding) {
    return at::_ops::slow_conv3d_forward::call(self, weight, kernel_size, bias, stride, padding);
  }
}

}
