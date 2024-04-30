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



#include <ATen/ops/_batch_norm_no_update_ops.h>

namespace at {


// aten::_batch_norm_no_update(Tensor input, Tensor? weight, Tensor? bias, Tensor? running_mean, Tensor? running_var, float momentum, float eps) -> (Tensor, Tensor, Tensor, Tensor)
inline ::std::tuple<at::Tensor,at::Tensor,at::Tensor,at::Tensor> _batch_norm_no_update(const at::Tensor & input, const ::std::optional<at::Tensor> & weight, const ::std::optional<at::Tensor> & bias, const ::std::optional<at::Tensor> & running_mean, const ::std::optional<at::Tensor> & running_var, double momentum, double eps) {
    return at::_ops::_batch_norm_no_update::call(input, weight, bias, running_mean, running_var, momentum, eps);
}

// aten::_batch_norm_no_update.out(Tensor input, Tensor? weight, Tensor? bias, Tensor? running_mean, Tensor? running_var, float momentum, float eps, *, Tensor(a!) out0, Tensor(b!) out1, Tensor(c!) out2, Tensor(d!) out3) -> (Tensor(a!), Tensor(b!), Tensor(c!), Tensor(d!))
inline ::std::tuple<at::Tensor &,at::Tensor &,at::Tensor &,at::Tensor &> _batch_norm_no_update_out(at::Tensor & out0, at::Tensor & out1, at::Tensor & out2, at::Tensor & out3, const at::Tensor & input, const ::std::optional<at::Tensor> & weight, const ::std::optional<at::Tensor> & bias, const ::std::optional<at::Tensor> & running_mean, const ::std::optional<at::Tensor> & running_var, double momentum, double eps) {
    return at::_ops::_batch_norm_no_update_out::call(input, weight, bias, running_mean, running_var, momentum, eps, out0, out1, out2, out3);
}
// aten::_batch_norm_no_update.out(Tensor input, Tensor? weight, Tensor? bias, Tensor? running_mean, Tensor? running_var, float momentum, float eps, *, Tensor(a!) out0, Tensor(b!) out1, Tensor(c!) out2, Tensor(d!) out3) -> (Tensor(a!), Tensor(b!), Tensor(c!), Tensor(d!))
inline ::std::tuple<at::Tensor &,at::Tensor &,at::Tensor &,at::Tensor &> _batch_norm_no_update_outf(const at::Tensor & input, const ::std::optional<at::Tensor> & weight, const ::std::optional<at::Tensor> & bias, const ::std::optional<at::Tensor> & running_mean, const ::std::optional<at::Tensor> & running_var, double momentum, double eps, at::Tensor & out0, at::Tensor & out1, at::Tensor & out2, at::Tensor & out3) {
    return at::_ops::_batch_norm_no_update_out::call(input, weight, bias, running_mean, running_var, momentum, eps, out0, out1, out2, out3);
}

}
