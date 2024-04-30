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



#include <ATen/ops/randint_like_ops.h>

namespace at {


// aten::randint_like(Tensor self, SymInt high, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None, MemoryFormat? memory_format=None) -> Tensor
inline at::Tensor randint_like(const at::Tensor & self, int64_t high, at::TensorOptions options={}, ::std::optional<at::MemoryFormat> memory_format=::std::nullopt) {
    return at::_ops::randint_like::call(self, high, c10::optTypeMetaToScalarType(options.dtype_opt()), options.layout_opt(), options.device_opt(), options.pinned_memory_opt(), c10::impl::check_tensor_options_and_extract_memory_format(options, memory_format));
}
namespace symint {
  template <typename T, typename = std::enable_if_t<std::is_same<T, int64_t>::value>>
  at::Tensor randint_like(const at::Tensor & self, int64_t high, at::TensorOptions options={}, ::std::optional<at::MemoryFormat> memory_format=::std::nullopt) {
    return at::_ops::randint_like::call(self, high, c10::optTypeMetaToScalarType(options.dtype_opt()), options.layout_opt(), options.device_opt(), options.pinned_memory_opt(), c10::impl::check_tensor_options_and_extract_memory_format(options, memory_format));
  }
}

// aten::randint_like(Tensor self, SymInt high, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None, MemoryFormat? memory_format=None) -> Tensor
inline at::Tensor randint_like(const at::Tensor & self, int64_t high, ::std::optional<at::ScalarType> dtype, ::std::optional<at::Layout> layout, ::std::optional<at::Device> device, ::std::optional<bool> pin_memory, ::std::optional<at::MemoryFormat> memory_format) {
    return at::_ops::randint_like::call(self, high, dtype, layout, device, pin_memory, memory_format);
}
namespace symint {
  template <typename T, typename = std::enable_if_t<std::is_same<T, int64_t>::value>>
  at::Tensor randint_like(const at::Tensor & self, int64_t high, ::std::optional<at::ScalarType> dtype, ::std::optional<at::Layout> layout, ::std::optional<at::Device> device, ::std::optional<bool> pin_memory, ::std::optional<at::MemoryFormat> memory_format) {
    return at::_ops::randint_like::call(self, high, dtype, layout, device, pin_memory, memory_format);
  }
}

// aten::randint_like(Tensor self, SymInt high, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None, MemoryFormat? memory_format=None) -> Tensor
inline at::Tensor randint_like_symint(const at::Tensor & self, c10::SymInt high, at::TensorOptions options={}, ::std::optional<at::MemoryFormat> memory_format=::std::nullopt) {
    return at::_ops::randint_like::call(self, high, c10::optTypeMetaToScalarType(options.dtype_opt()), options.layout_opt(), options.device_opt(), options.pinned_memory_opt(), c10::impl::check_tensor_options_and_extract_memory_format(options, memory_format));
}
namespace symint {
  template <typename T, typename = std::enable_if_t<std::is_same<T, c10::SymInt>::value>>
  at::Tensor randint_like(const at::Tensor & self, c10::SymInt high, at::TensorOptions options={}, ::std::optional<at::MemoryFormat> memory_format=::std::nullopt) {
    return at::_ops::randint_like::call(self, high, c10::optTypeMetaToScalarType(options.dtype_opt()), options.layout_opt(), options.device_opt(), options.pinned_memory_opt(), c10::impl::check_tensor_options_and_extract_memory_format(options, memory_format));
  }
}

// aten::randint_like(Tensor self, SymInt high, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None, MemoryFormat? memory_format=None) -> Tensor
inline at::Tensor randint_like_symint(const at::Tensor & self, c10::SymInt high, ::std::optional<at::ScalarType> dtype, ::std::optional<at::Layout> layout, ::std::optional<at::Device> device, ::std::optional<bool> pin_memory, ::std::optional<at::MemoryFormat> memory_format) {
    return at::_ops::randint_like::call(self, high, dtype, layout, device, pin_memory, memory_format);
}
namespace symint {
  template <typename T, typename = std::enable_if_t<std::is_same<T, c10::SymInt>::value>>
  at::Tensor randint_like(const at::Tensor & self, c10::SymInt high, ::std::optional<at::ScalarType> dtype, ::std::optional<at::Layout> layout, ::std::optional<at::Device> device, ::std::optional<bool> pin_memory, ::std::optional<at::MemoryFormat> memory_format) {
    return at::_ops::randint_like::call(self, high, dtype, layout, device, pin_memory, memory_format);
  }
}

// aten::randint_like.low_dtype(Tensor self, SymInt low, SymInt high, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None, MemoryFormat? memory_format=None) -> Tensor
inline at::Tensor randint_like(const at::Tensor & self, int64_t low, int64_t high, at::TensorOptions options={}, ::std::optional<at::MemoryFormat> memory_format=::std::nullopt) {
    return at::_ops::randint_like_low_dtype::call(self, low, high, c10::optTypeMetaToScalarType(options.dtype_opt()), options.layout_opt(), options.device_opt(), options.pinned_memory_opt(), c10::impl::check_tensor_options_and_extract_memory_format(options, memory_format));
}
namespace symint {
  template <typename T, typename = std::enable_if_t<std::is_same<T, int64_t>::value>>
  at::Tensor randint_like(const at::Tensor & self, int64_t low, int64_t high, at::TensorOptions options={}, ::std::optional<at::MemoryFormat> memory_format=::std::nullopt) {
    return at::_ops::randint_like_low_dtype::call(self, low, high, c10::optTypeMetaToScalarType(options.dtype_opt()), options.layout_opt(), options.device_opt(), options.pinned_memory_opt(), c10::impl::check_tensor_options_and_extract_memory_format(options, memory_format));
  }
}

// aten::randint_like.low_dtype(Tensor self, SymInt low, SymInt high, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None, MemoryFormat? memory_format=None) -> Tensor
inline at::Tensor randint_like(const at::Tensor & self, int64_t low, int64_t high, ::std::optional<at::ScalarType> dtype, ::std::optional<at::Layout> layout, ::std::optional<at::Device> device, ::std::optional<bool> pin_memory, ::std::optional<at::MemoryFormat> memory_format) {
    return at::_ops::randint_like_low_dtype::call(self, low, high, dtype, layout, device, pin_memory, memory_format);
}
namespace symint {
  template <typename T, typename = std::enable_if_t<std::is_same<T, int64_t>::value>>
  at::Tensor randint_like(const at::Tensor & self, int64_t low, int64_t high, ::std::optional<at::ScalarType> dtype, ::std::optional<at::Layout> layout, ::std::optional<at::Device> device, ::std::optional<bool> pin_memory, ::std::optional<at::MemoryFormat> memory_format) {
    return at::_ops::randint_like_low_dtype::call(self, low, high, dtype, layout, device, pin_memory, memory_format);
  }
}

// aten::randint_like.low_dtype(Tensor self, SymInt low, SymInt high, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None, MemoryFormat? memory_format=None) -> Tensor
inline at::Tensor randint_like_symint(const at::Tensor & self, c10::SymInt low, c10::SymInt high, at::TensorOptions options={}, ::std::optional<at::MemoryFormat> memory_format=::std::nullopt) {
    return at::_ops::randint_like_low_dtype::call(self, low, high, c10::optTypeMetaToScalarType(options.dtype_opt()), options.layout_opt(), options.device_opt(), options.pinned_memory_opt(), c10::impl::check_tensor_options_and_extract_memory_format(options, memory_format));
}
namespace symint {
  template <typename T, typename = std::enable_if_t<std::is_same<T, c10::SymInt>::value>>
  at::Tensor randint_like(const at::Tensor & self, c10::SymInt low, c10::SymInt high, at::TensorOptions options={}, ::std::optional<at::MemoryFormat> memory_format=::std::nullopt) {
    return at::_ops::randint_like_low_dtype::call(self, low, high, c10::optTypeMetaToScalarType(options.dtype_opt()), options.layout_opt(), options.device_opt(), options.pinned_memory_opt(), c10::impl::check_tensor_options_and_extract_memory_format(options, memory_format));
  }
}

// aten::randint_like.low_dtype(Tensor self, SymInt low, SymInt high, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None, MemoryFormat? memory_format=None) -> Tensor
inline at::Tensor randint_like_symint(const at::Tensor & self, c10::SymInt low, c10::SymInt high, ::std::optional<at::ScalarType> dtype, ::std::optional<at::Layout> layout, ::std::optional<at::Device> device, ::std::optional<bool> pin_memory, ::std::optional<at::MemoryFormat> memory_format) {
    return at::_ops::randint_like_low_dtype::call(self, low, high, dtype, layout, device, pin_memory, memory_format);
}
namespace symint {
  template <typename T, typename = std::enable_if_t<std::is_same<T, c10::SymInt>::value>>
  at::Tensor randint_like(const at::Tensor & self, c10::SymInt low, c10::SymInt high, ::std::optional<at::ScalarType> dtype, ::std::optional<at::Layout> layout, ::std::optional<at::Device> device, ::std::optional<bool> pin_memory, ::std::optional<at::MemoryFormat> memory_format) {
    return at::_ops::randint_like_low_dtype::call(self, low, high, dtype, layout, device, pin_memory, memory_format);
  }
}

// aten::randint_like.out(Tensor self, SymInt high, *, MemoryFormat? memory_format=None, Tensor(a!) out) -> Tensor(a!)
inline at::Tensor & randint_like_out(at::Tensor & out, const at::Tensor & self, int64_t high, ::std::optional<at::MemoryFormat> memory_format=::std::nullopt) {
    return at::_ops::randint_like_out::call(self, high, memory_format, out);
}
namespace symint {
  template <typename T, typename = std::enable_if_t<std::is_same<T, int64_t>::value>>
  at::Tensor & randint_like_out(at::Tensor & out, const at::Tensor & self, int64_t high, ::std::optional<at::MemoryFormat> memory_format=::std::nullopt) {
    return at::_ops::randint_like_out::call(self, high, memory_format, out);
  }
}

// aten::randint_like.out(Tensor self, SymInt high, *, MemoryFormat? memory_format=None, Tensor(a!) out) -> Tensor(a!)
inline at::Tensor & randint_like_outf(const at::Tensor & self, int64_t high, ::std::optional<at::MemoryFormat> memory_format, at::Tensor & out) {
    return at::_ops::randint_like_out::call(self, high, memory_format, out);
}
namespace symint {
  template <typename T, typename = std::enable_if_t<std::is_same<T, int64_t>::value>>
  at::Tensor & randint_like_outf(const at::Tensor & self, int64_t high, ::std::optional<at::MemoryFormat> memory_format, at::Tensor & out) {
    return at::_ops::randint_like_out::call(self, high, memory_format, out);
  }
}

// aten::randint_like.out(Tensor self, SymInt high, *, MemoryFormat? memory_format=None, Tensor(a!) out) -> Tensor(a!)
inline at::Tensor & randint_like_symint_out(at::Tensor & out, const at::Tensor & self, c10::SymInt high, ::std::optional<at::MemoryFormat> memory_format=::std::nullopt) {
    return at::_ops::randint_like_out::call(self, high, memory_format, out);
}
namespace symint {
  template <typename T, typename = std::enable_if_t<std::is_same<T, c10::SymInt>::value>>
  at::Tensor & randint_like_out(at::Tensor & out, const at::Tensor & self, c10::SymInt high, ::std::optional<at::MemoryFormat> memory_format=::std::nullopt) {
    return at::_ops::randint_like_out::call(self, high, memory_format, out);
  }
}

// aten::randint_like.out(Tensor self, SymInt high, *, MemoryFormat? memory_format=None, Tensor(a!) out) -> Tensor(a!)
inline at::Tensor & randint_like_symint_outf(const at::Tensor & self, c10::SymInt high, ::std::optional<at::MemoryFormat> memory_format, at::Tensor & out) {
    return at::_ops::randint_like_out::call(self, high, memory_format, out);
}
namespace symint {
  template <typename T, typename = std::enable_if_t<std::is_same<T, c10::SymInt>::value>>
  at::Tensor & randint_like_outf(const at::Tensor & self, c10::SymInt high, ::std::optional<at::MemoryFormat> memory_format, at::Tensor & out) {
    return at::_ops::randint_like_out::call(self, high, memory_format, out);
  }
}

// aten::randint_like.low_dtype_out(Tensor self, SymInt low, SymInt high, *, MemoryFormat? memory_format=None, Tensor(a!) out) -> Tensor(a!)
inline at::Tensor & randint_like_out(at::Tensor & out, const at::Tensor & self, int64_t low, int64_t high, ::std::optional<at::MemoryFormat> memory_format=::std::nullopt) {
    return at::_ops::randint_like_low_dtype_out::call(self, low, high, memory_format, out);
}
namespace symint {
  template <typename T, typename = std::enable_if_t<std::is_same<T, int64_t>::value>>
  at::Tensor & randint_like_out(at::Tensor & out, const at::Tensor & self, int64_t low, int64_t high, ::std::optional<at::MemoryFormat> memory_format=::std::nullopt) {
    return at::_ops::randint_like_low_dtype_out::call(self, low, high, memory_format, out);
  }
}

// aten::randint_like.low_dtype_out(Tensor self, SymInt low, SymInt high, *, MemoryFormat? memory_format=None, Tensor(a!) out) -> Tensor(a!)
inline at::Tensor & randint_like_outf(const at::Tensor & self, int64_t low, int64_t high, ::std::optional<at::MemoryFormat> memory_format, at::Tensor & out) {
    return at::_ops::randint_like_low_dtype_out::call(self, low, high, memory_format, out);
}
namespace symint {
  template <typename T, typename = std::enable_if_t<std::is_same<T, int64_t>::value>>
  at::Tensor & randint_like_outf(const at::Tensor & self, int64_t low, int64_t high, ::std::optional<at::MemoryFormat> memory_format, at::Tensor & out) {
    return at::_ops::randint_like_low_dtype_out::call(self, low, high, memory_format, out);
  }
}

// aten::randint_like.low_dtype_out(Tensor self, SymInt low, SymInt high, *, MemoryFormat? memory_format=None, Tensor(a!) out) -> Tensor(a!)
inline at::Tensor & randint_like_symint_out(at::Tensor & out, const at::Tensor & self, c10::SymInt low, c10::SymInt high, ::std::optional<at::MemoryFormat> memory_format=::std::nullopt) {
    return at::_ops::randint_like_low_dtype_out::call(self, low, high, memory_format, out);
}
namespace symint {
  template <typename T, typename = std::enable_if_t<std::is_same<T, c10::SymInt>::value>>
  at::Tensor & randint_like_out(at::Tensor & out, const at::Tensor & self, c10::SymInt low, c10::SymInt high, ::std::optional<at::MemoryFormat> memory_format=::std::nullopt) {
    return at::_ops::randint_like_low_dtype_out::call(self, low, high, memory_format, out);
  }
}

// aten::randint_like.low_dtype_out(Tensor self, SymInt low, SymInt high, *, MemoryFormat? memory_format=None, Tensor(a!) out) -> Tensor(a!)
inline at::Tensor & randint_like_symint_outf(const at::Tensor & self, c10::SymInt low, c10::SymInt high, ::std::optional<at::MemoryFormat> memory_format, at::Tensor & out) {
    return at::_ops::randint_like_low_dtype_out::call(self, low, high, memory_format, out);
}
namespace symint {
  template <typename T, typename = std::enable_if_t<std::is_same<T, c10::SymInt>::value>>
  at::Tensor & randint_like_outf(const at::Tensor & self, c10::SymInt low, c10::SymInt high, ::std::optional<at::MemoryFormat> memory_format, at::Tensor & out) {
    return at::_ops::randint_like_low_dtype_out::call(self, low, high, memory_format, out);
  }
}

}
