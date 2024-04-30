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



#include <ATen/ops/_standard_gamma_ops.h>

namespace at {


// aten::_standard_gamma(Tensor self, Generator? generator=None) -> Tensor
inline at::Tensor _standard_gamma(const at::Tensor & self, ::std::optional<at::Generator> generator=::std::nullopt) {
    return at::_ops::_standard_gamma::call(self, generator);
}

// aten::_standard_gamma.out(Tensor self, Generator? generator=None, *, Tensor(a!) out) -> Tensor(a!)
inline at::Tensor & _standard_gamma_out(at::Tensor & out, const at::Tensor & self, ::std::optional<at::Generator> generator=::std::nullopt) {
    return at::_ops::_standard_gamma_out::call(self, generator, out);
}
// aten::_standard_gamma.out(Tensor self, Generator? generator=None, *, Tensor(a!) out) -> Tensor(a!)
inline at::Tensor & _standard_gamma_outf(const at::Tensor & self, ::std::optional<at::Generator> generator, at::Tensor & out) {
    return at::_ops::_standard_gamma_out::call(self, generator, out);
}

}
