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



#include <ATen/ops/_batch_norm_impl_index_ops.h>

namespace at {


// aten::_batch_norm_impl_index(Tensor input, Tensor? weight, Tensor? bias, Tensor? running_mean, Tensor? running_var, bool training, float momentum, float eps, bool cudnn_enabled) -> (Tensor, Tensor, Tensor, Tensor, int)
inline ::std::tuple<at::Tensor,at::Tensor,at::Tensor,at::Tensor,int64_t> _batch_norm_impl_index(const at::Tensor & input, const ::std::optional<at::Tensor> & weight, const ::std::optional<at::Tensor> & bias, const ::std::optional<at::Tensor> & running_mean, const ::std::optional<at::Tensor> & running_var, bool training, double momentum, double eps, bool cudnn_enabled) {
    return at::_ops::_batch_norm_impl_index::call(input, weight, bias, running_mean, running_var, training, momentum, eps, cudnn_enabled);
}

}
