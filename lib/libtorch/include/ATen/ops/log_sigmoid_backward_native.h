#pragma once

// @generated by torchgen/gen.py from NativeFunction.h

#include <c10/core/Scalar.h>
#include <c10/core/Storage.h>
#include <c10/core/TensorOptions.h>
#include <c10/util/Deprecated.h>
#include <c10/util/Optional.h>
#include <c10/core/QScheme.h>
#include <ATen/core/Reduction.h>
#include <ATen/core/Tensor.h>
#include <tuple>
#include <vector>


namespace at {
namespace native {
TORCH_API at::Tensor log_sigmoid_backward_cpu(const at::Tensor & grad_output, const at::Tensor & self, const at::Tensor & buffer);
TORCH_API at::Tensor & log_sigmoid_backward_cpu_out(const at::Tensor & grad_output, const at::Tensor & self, const at::Tensor & buffer, at::Tensor & grad_input);
TORCH_API at::Tensor log_sigmoid_backward_cuda(const at::Tensor & grad_output, const at::Tensor & self, const at::Tensor & buffer);
TORCH_API at::Tensor & log_sigmoid_backward_cuda_out(const at::Tensor & grad_output, const at::Tensor & self, const at::Tensor & buffer, at::Tensor & grad_input);
TORCH_API at::Tensor log_sigmoid_backward_mps(const at::Tensor & grad_output, const at::Tensor & self, const at::Tensor & buffer);
TORCH_API at::Tensor & log_sigmoid_backward_mps_out(const at::Tensor & grad_output, const at::Tensor & self, const at::Tensor & buffer, at::Tensor & grad_input);
} // namespace native
} // namespace at
