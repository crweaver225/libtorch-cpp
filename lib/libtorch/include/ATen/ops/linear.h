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



#include <ATen/ops/linear_ops.h>

namespace at {


// aten::linear(Tensor input, Tensor weight, Tensor? bias=None) -> Tensor
inline at::Tensor linear(const at::Tensor & input, const at::Tensor & weight, const ::std::optional<at::Tensor> & bias={}) {
    return at::_ops::linear::call(input, weight, bias);
}

// aten::linear.out(Tensor input, Tensor weight, Tensor? bias=None, *, Tensor(a!) out) -> Tensor(a!)
inline at::Tensor & linear_out(at::Tensor & out, const at::Tensor & input, const at::Tensor & weight, const ::std::optional<at::Tensor> & bias={}) {
    return at::_ops::linear_out::call(input, weight, bias, out);
}
// aten::linear.out(Tensor input, Tensor weight, Tensor? bias=None, *, Tensor(a!) out) -> Tensor(a!)
inline at::Tensor & linear_outf(const at::Tensor & input, const at::Tensor & weight, const ::std::optional<at::Tensor> & bias, at::Tensor & out) {
    return at::_ops::linear_out::call(input, weight, bias, out);
}

}