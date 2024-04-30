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



#include <ATen/ops/_functional_sym_constrain_range_ops.h>

namespace at {


// aten::_functional_sym_constrain_range(Scalar size, int? min, int? max, Tensor dep_token) -> Tensor
inline at::Tensor _functional_sym_constrain_range(const at::Scalar & size, ::std::optional<int64_t> min, ::std::optional<int64_t> max, const at::Tensor & dep_token) {
    return at::_ops::_functional_sym_constrain_range::call(size, min, max, dep_token);
}

}
