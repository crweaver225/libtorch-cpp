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



#include <ATen/ops/rand_like_ops.h>

namespace at {


// aten::rand_like(Tensor self, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None, MemoryFormat? memory_format=None) -> Tensor
inline at::Tensor rand_like(const at::Tensor & self, at::TensorOptions options={}, ::std::optional<at::MemoryFormat> memory_format=::std::nullopt) {
    return at::_ops::rand_like::call(self, c10::optTypeMetaToScalarType(options.dtype_opt()), options.layout_opt(), options.device_opt(), options.pinned_memory_opt(), c10::impl::check_tensor_options_and_extract_memory_format(options, memory_format));
}
// aten::rand_like(Tensor self, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None, MemoryFormat? memory_format=None) -> Tensor
inline at::Tensor rand_like(const at::Tensor & self, ::std::optional<at::ScalarType> dtype, ::std::optional<at::Layout> layout, ::std::optional<at::Device> device, ::std::optional<bool> pin_memory, ::std::optional<at::MemoryFormat> memory_format) {
    return at::_ops::rand_like::call(self, dtype, layout, device, pin_memory, memory_format);
}

// aten::rand_like.out(Tensor self, *, MemoryFormat? memory_format=None, Tensor(a!) out) -> Tensor(a!)
inline at::Tensor & rand_like_out(at::Tensor & out, const at::Tensor & self, ::std::optional<at::MemoryFormat> memory_format=::std::nullopt) {
    return at::_ops::rand_like_out::call(self, memory_format, out);
}
// aten::rand_like.out(Tensor self, *, MemoryFormat? memory_format=None, Tensor(a!) out) -> Tensor(a!)
inline at::Tensor & rand_like_outf(const at::Tensor & self, ::std::optional<at::MemoryFormat> memory_format, at::Tensor & out) {
    return at::_ops::rand_like_out::call(self, memory_format, out);
}

}