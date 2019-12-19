"""Resource functions for various resources (e.g. FLOPs, latency)."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from morph_net.framework import op_handler_util
from morph_net.network_regularizers import cost_calculator

import numpy as np
import tensorflow.compat.v1 as tf

# Data sheet for K80:
# https://www.nvidia.com/content/dam/en-zz/Solutions/Data-Center/tesla-product-literature/nvidia-tesla-k80-overview.pdf

# Data sheet for P4:
# https://images.nvidia.com/content/pdf/tesla/184457-Tesla-P4-Datasheet-NV-Final-Letter-Web.pdf

# Data sheet for T4:
# https://www.nvidia.com/content/dam/en-zz/Solutions/Data-Center/t4-tensor-core-datasheet.pdf

# Data sheet for P100:
# https://images.nvidia.com/content/tesla/pdf/nvidia-tesla-p100-PCIe-datasheet.pdf

# Data sheet for V100:
# https://images.nvidia.com/content/technologies/volta/pdf/tesla-volta-v100-datasheet-letter-fnl-web.pdf

# Data sheet for TPUv2:
# http://learningsys.org/nips17/assets/slides/dean-nips17.pdf
PEAK_COMPUTE = {  # GFLOP/s
    'K80': 8740,
    'P4': 5500,
    'T4': 8100,
    'P100': 9300,
    'V100': 125000,
    'TPUv2': 22500,
    'FLOP_LATENCY': 1,  # Simulate a device with infinite memory bandwidth.
    'MEMORY_LATENCY': 1e20,  # Simulate a device with infinite peak compute.
}
MEMORY_BANDWIDTH = {  # GB/s
    'K80': 480,
    'P4': 192,
    'T4': 300,
    'P100': 732,
    'V100': 900,
    'TPUv2': 300,
    'FLOP_LATENCY': 1e20,
    'MEMORY_LATENCY': 1,
}


def flop_coeff(op):
  """Computes the coefficient of number of flops associated with a convolution.

  The FLOPs cost of a convolution is given by C * output_depth * input_depth,
  where C = 2 * output_width * output_height * filter_size. The 2 is because we
  have one multiplication and one addition for each convolution weight and
  pixel. This function returns C.

  Supported operations names are listed in cost_calculator.FLOP_OPS.

  Args:
    op: A tf.Operation of supported types.

  Returns:
    A float, the coefficient that when multiplied by the input depth and by the
    output depth gives the number of flops needed to compute the convolution.

  Raises:
    ValueError: conv_op is not a supported tf.Operation.
  """
  if not is_flop_op(op):
    return 0.0
  if op.type == 'MatMul':
    # A MatMul is like a 1x1 conv with an output size of 1x1, so from the factor
    # below only the 2.0 remains.
    return 2.0
  # Looking at the output shape makes it easy to automatically take into
  # account strides and the type of padding.
  def kernel_num_elements(tensor):
    """Returns the number of elements of a kernel.

    Args:
      tensor: The weight tensor.

    Returns:
      Number of elements of the kernel (either float or tf.float).
    """
    num_elements = np.prod(tensor.shape.dims[1:-1]).value
    if num_elements:
      return num_elements
    return tf.to_float(tf.reduce_prod(tf.shape(tensor)[1:-1]))

  if op.type in ('Conv2D', 'DepthwiseConv2dNative', 'Conv3D'):
    num_elements = kernel_num_elements(op.outputs[0])
  elif op.type == 'Conv2DBackpropInput':
    # For a transposed convolution, the input and the output are swapped (as
    # far as shapes are concerned). In other words, for a given filter shape
    # and stride, if Conv2D maps from shapeX to shapeY, Conv2DBackpropInput
    # maps from shapeY to shapeX. Therefore wherever we use the output shape
    # for Conv2D, we use the input shape for Conv2DBackpropInput.
    num_elements = kernel_num_elements(cost_calculator.get_input_activation(op))
  else:
    # Can only happen if elements are added to FLOP_OPS and not taken care of.
    assert False, '%s in cost_calculator.FLOP_OPS but not handled' % op.type
  # Handle dynamic shaping while keeping old code path to not break
  # other clients.
  return 2.0 * num_elements * _get_conv_filter_size(op)


def num_weights_coeff(op):
  """The number of weights of a conv is C * output_depth * input_depth. Finds C.

  Args:
    op: A tf.Operation of type 'Conv2D' or 'MatMul'

  Returns:
    A float, the coefficient that when multiplied by the input depth and by the
    output depth gives the number of flops needed to compute the convolution.

  Raises:
    ValueError: conv_op is not a tf.Operation of type Conv2D.
  """
  if not is_flop_op(op):
    return 0.0
  return (_get_conv_filter_size(op) if op.type in cost_calculator.CONV_OPS
          else 1.0)


def flop_function(op, is_regularization, num_alive_inputs, num_alive_outputs,
                  reg_inputs, reg_outputs, batch_size=1):
  """Calculates FLOP cost or regularization loss for an op.

  Args:
    op: A tf.Operation.
    is_regularization: Boolean indicating whether to calculate regularization
      loss.  If False, calculate cost instead.
    num_alive_inputs: Scalar Tensor indicating how many input channels are
      considered alive.
    num_alive_outputs: Scalar Tensor indicating how many output channels are
      considered alive.
    reg_inputs: Scalar Tensor which is the sum over the input regularization
      vector.
    reg_outputs: Scalar Tensor which is the sum over the output regularization
      vector.
    batch_size: Integer batch size to calculate cost/loss for.

  Returns:
    Tensor with the cost or regularization loss of the op in terms of FLOPs.
  """
  coeff = flop_coeff(op)
  if is_regularization:
    return _calculate_bilinear_regularization(
        op, coeff, num_alive_inputs, num_alive_outputs, reg_inputs, reg_outputs,
        batch_size)
  return _calculate_bilinear_cost(
      op, coeff, num_alive_inputs, num_alive_outputs, batch_size)


def memory_function(op, is_regularization, num_alive_inputs, num_alive_outputs,
                    reg_inputs, reg_outputs, batch_size=1):
  """Calculates memory cost or regularization loss for an op.

  Args:
    op: A tf.Operation.
    is_regularization: Boolean indicating whether to calculate regularization
      loss.  If False, calculate cost instead.
    num_alive_inputs: Scalar Tensor indicating how many input channels are
      considered alive.
    num_alive_outputs: Scalar Tensor indicating how many output channels are
      considered alive.
    reg_inputs: Scalar Tensor which is the sum over the input regularization
      vector.
    reg_outputs: Scalar Tensor which is the sum over the output regularization
      vector.
    batch_size: Integer batch size to calculate cost/loss for.

  Returns:
    Tensor with the cost or regularization loss of the op in terms of memory.
  """
  # Separate tensors based on how their cost depends on input and output
  # channels.
  # 1. Input tensors: Memory size scales with input channels and batch size.
  # 2. Output tensors: Memory size scales with output channels and batch size.
  # 3. Weight tensors: Memory size scales with input and output channels
  #    (bilinear).
  input_tensors = []
  output_tensors = []
  bilinear_tensors = []
  weight_tensor_index = op_handler_util.WEIGHTS_INDEX_DICT.get(op.type)
  for i, tensor in enumerate(op.inputs):
    if weight_tensor_index is not None and i == weight_tensor_index:
      bilinear_tensors.append(tensor)
    else:
      if op.type == 'Conv2DBackpropInput' and i == 0:
        # This is tensor <scope>/stack:0 which just holds the output shape.
        continue
      if 'FusedBatchNorm' in op.type and i > 0:
        # Skip the gamma, beta, mean, and std.
        continue
      if op.type == 'Sum' and i == 1:
        # Skip the reduction_indices tensor for tf.reduction_sum op.
        continue
      input_tensors.append(tensor)
  for i, tensor in enumerate(op.outputs):
    if 'FusedBatchNorm' in op.type and i > 0:
      # Skip other batch norm outputs.
      continue
    output_tensors.append(tensor)

  if op.type == 'ConcatV2':
    # For concat, the alive/regularization of the input is the same as the
    # output, but split into multiple tensors.  For simplicity, treat the input
    # as the output rather than using the alive/regularization of individual
    # inputs.
    input_tensors = output_tensors

  # Normalize memory payload to batch size 1 and depth 1.  Rescale by target
  # batch size and input/output depth to calculate actual cost/loss.
  normalized_input_payloads = []
  for input_tensor in input_tensors:
    shape = _shape_with_dtype(input_tensor)
    # Divide by batch size and input channels.
    normalized_input_payloads.append(
        tf.reduce_prod(shape[1:-1]) * input_tensor.dtype.size)
  normalized_output_payloads = []
  for output_tensor in output_tensors:
    shape = _shape_with_dtype(output_tensor)
    # Divide by batch size and output channels.
    normalized_output_payloads.append(
        tf.reduce_prod(shape[1:-1]) * output_tensor.dtype.size)
  normalized_bilinear_payloads = []
  for bilinear_tensor in bilinear_tensors:
    shape = _shape_with_dtype(bilinear_tensor)
    # Divide by input and output channels.
    normalized_bilinear_payloads.append(
        tf.reduce_prod(shape[:-2]) * bilinear_tensor.dtype.size)

  # Rescale normalized payload to calculate cost/loss.
  input_payloads = []
  output_payloads = []
  bilinear_payloads = []
  if is_regularization:
    for input_payload in normalized_input_payloads:
      input_payloads.append(input_payload * batch_size * reg_inputs)
    for output_payload in normalized_output_payloads:
      output_payloads.append(output_payload * batch_size * reg_outputs)
    for bilinear_payload in normalized_bilinear_payloads:
      bilinear_payloads.append(
          bilinear_payload * (
              num_alive_inputs * reg_outputs + num_alive_outputs * reg_inputs))
  else:
    for input_payload in normalized_input_payloads:
      input_payloads.append(
          input_payload * batch_size * num_alive_inputs)
    for output_payload in normalized_output_payloads:
      output_payloads.append(
          output_payload * batch_size * num_alive_outputs)
    for bilinear_payload in normalized_bilinear_payloads:
      bilinear_payloads.append(
          bilinear_payload * num_alive_inputs * num_alive_outputs)

  return tf.reduce_sum(input_payloads + output_payloads + bilinear_payloads)


def latency_function(op, is_regularization, num_alive_inputs, num_alive_outputs,
                     reg_inputs, reg_outputs, peak_compute, memory_bandwidth,
                     batch_size=1):
  """Calculates latency cost or regularization loss for an op.

  Calculates the compute and memory cost of the op and returns the max.  This
  assumes ops can overlap compute and memory such that latency results from the
  slower of the 2 constraints.

  Args:
    op: A tf.Operation.
    is_regularization: Boolean indicating whether to calculate regularization
      loss.  If False, calculate cost instead.
    num_alive_inputs: Scalar Tensor indicating how many input channels are
      considered alive.
    num_alive_outputs: Scalar Tensor indicating how many output channels are
      considered alive.
    reg_inputs: Scalar Tensor which is the sum over the input regularization
      vector.
    reg_outputs: Scalar Tensor which is the sum over the output regularization
      vector.
    peak_compute: Integer peak compute of the target hardware in GFLOP/s.
    memory_bandwidth: Integer memory bandwidth of the target hardware in GB/s.
    batch_size: Integer batch size to calculate cost/loss for.

  Returns:
    Tensor with the cost or regularization loss of the op in terms of latency.
  """
  # Calculate compute cost for FLOP-expensive ops.
  flop_cost = 0
  if is_flop_op(op):
    # This op has non-trivial compute.
    flop_cost = flop_function(
        op, False, num_alive_inputs, num_alive_outputs, reg_inputs,
        reg_outputs, batch_size)

  # Convert FLOP cost to compute time cost.
  compute_cost = flop_cost / peak_compute

  memory_payload = memory_function(
      op, False, num_alive_inputs, num_alive_outputs, reg_inputs,
      reg_outputs, batch_size)
  # Convert memory payload cost to memory time cost.
  memory_cost = memory_payload / memory_bandwidth

  if is_regularization:
    compute_loss = flop_function(
        op, True, num_alive_inputs, num_alive_outputs, reg_inputs, reg_outputs,
        batch_size) / peak_compute
    memory_loss = memory_function(
        op, True, num_alive_inputs, num_alive_outputs, reg_inputs, reg_outputs,
        batch_size) / memory_bandwidth
    return tf.cond(memory_cost > compute_cost,
                   lambda: memory_loss,
                   lambda: compute_loss)
  else:
    return tf.maximum(compute_cost, memory_cost)


def latency_function_factory(hardware, batch_size):
  """Return latency_function with appropriate hardware platform specs.

  Args:
    hardware: String hardware platform to target for latency. Must be a key
      from PEAK_COMPUTE and MEMORY_BANDWIDTH.
    batch_size: Integer batch size to calculate cost/loss for.

  Returns:
    Function latency_function with target hardware specs.

  Raises:
    ValueError: If hardware not supported.
  """
  assert batch_size > 0
  if hardware not in PEAK_COMPUTE:
    raise ValueError(
        'Hardware %s must be in %s' % (hardware, PEAK_COMPUTE.keys()))
  # Create latency_function with hardware specifications.
  def latency_function_for_hardware(
      op, is_regularization, num_alive_inputs, num_alive_outputs, reg_inputs,
      reg_outputs):
    return latency_function(
        op, is_regularization, num_alive_inputs, num_alive_outputs, reg_inputs,
        reg_outputs, PEAK_COMPUTE[hardware], MEMORY_BANDWIDTH[hardware],
        batch_size)

  return latency_function_for_hardware


def model_size_function(op, is_regularization, num_alive_inputs,
                        num_alive_outputs, reg_inputs, reg_outputs,
                        batch_size=1):
  """Calculates model size cost or regularization loss for an op.

  Args:
    op: A tf.Operation.
    is_regularization: Boolean indicating whether to calculate regularization
      loss.  If False, calculate cost instead.
    num_alive_inputs: Scalar Tensor indicating how many input channels are
      considered alive.
    num_alive_outputs: Scalar Tensor indicating how many output channels are
      considered alive.
    reg_inputs: Scalar Tensor which is the sum over the input regularization
      vector.
    reg_outputs: Scalar Tensor which is the sum over the output regularization
      vector.
    batch_size: Integer batch size to calculate cost/loss for.  Unused.

  Returns:
    Tensor with the cost or regularization loss of the op in terms of FLOPs.
  """
  del batch_size  # Unused.
  coeff = num_weights_coeff(op)
  if is_regularization:
    return _calculate_bilinear_regularization(
        op, coeff, num_alive_inputs, num_alive_outputs, reg_inputs, reg_outputs,
        1)
  return _calculate_bilinear_cost(
      op, coeff, num_alive_inputs, num_alive_outputs, 1)


def activation_count_function(op, is_regularization, num_alive_inputs,
                              num_alive_outputs, reg_inputs, reg_outputs,
                              batch_size=1):
  """Calculates activation cost or regularization loss for an op.

  Args:
    op: A tf.Operation.
    is_regularization: Boolean indicating whether to calculate regularization
      loss.  If False, calculate cost instead.
    num_alive_inputs: Scalar Tensor indicating how many input channels are
      considered alive.
    num_alive_outputs: Scalar Tensor indicating how many output channels are
      considered alive.
    reg_inputs: Scalar Tensor which is the sum over the input regularization
      vector.
    reg_outputs: Scalar Tensor which is the sum over the output regularization
      vector.
    batch_size: Integer batch size to calculate cost/loss for.  Unused.

  Returns:
    Tensor with the cost or regularization loss of the op in terms of FLOPs.
  """
  del num_alive_inputs  # Unused.
  del reg_inputs  # Unused.
  del batch_size  # Unused.
  if not is_flop_op(op):
    return 0.0
  if is_regularization:
    return reg_outputs
  return num_alive_outputs


def _shape_with_dtype(tensor):
  """Returns the tensor shape with the same dtype as the tensor.

  Args:
    tensor: A tf.Tensor.

  Returns:
    A tf.Tensor of the tensor shape with the same dtype as tensor.
  """
  return tf.cast(tf.shape(tensor), tensor.dtype)


def is_flop_op(op):
  """Returns True if op consumes significant FLOPs to evaluate."""
  if not isinstance(op, tf.Operation):
    raise ValueError('conv_op must be a tf.Operation, not %s' % type(op))
  return op.type in cost_calculator.FLOP_OPS


def _get_conv_filter_size(conv_op):
  # Works for 2D and 3D convs where sizes of weight matrix are:
  # 4D or 5D tensors: [kernel_size[:], inputs, outputs]
  assert conv_op.type in cost_calculator.CONV_OPS
  conv_weights = conv_op.inputs[1]
  filter_shape = conv_weights.shape.as_list()[:-2]
  return np.prod(filter_shape)


def _calculate_bilinear_regularization(
    op, coeff, num_alive_inputs, num_alive_outputs, reg_inputs, reg_outputs,
    batch_size):
  """Calculates bilinear regularization term for an op.

  Args:
    op: A tf.Operation.
    coeff: A float coefficient for the bilinear function.
    num_alive_inputs: Scalar Tensor indicating how many input channels are
      considered alive.
    num_alive_outputs: Scalar Tensor indicating how many output channels are
      considered alive.
    reg_inputs: Scalar Tensor which is the sum over the input regularization
      vector.
    reg_outputs: Scalar Tensor which is the sum over the output regularization
      vector.
    batch_size: Integer batch size to calculate cost/loss for.

  Returns:
    Tensor with the regularization loss of the op.
  """
  if op.type == 'DepthwiseConv2dNative':
    # reg_inputs and reg_outputs are often identical since they should
    # come from the same regularizer. Duplicate them for symmetry.
    # When the input doesn't have a regularizer (e.g. input), only the
    # second term is used.
    # TODO(b1): revisit this expression after experiments.
    return batch_size * coeff * (reg_inputs + reg_outputs)
  else:
    # Handle normal ops.
    return batch_size * coeff * (
        num_alive_inputs * reg_outputs + num_alive_outputs * reg_inputs)


def _calculate_bilinear_cost(
    op, coeff, num_alive_inputs, num_alive_outputs, batch_size):
  """Calculates bilinear cost for an op.

  Args:
    op: A tf.Operation.
    coeff: A float coefficient for the bilinear function.
    num_alive_inputs: Scalar Tensor indicating how many input channels are
      considered alive.
    num_alive_outputs: Scalar Tensor indicating how many output channels are
      considered alive.
    batch_size: Integer batch size to calculate cost/loss for.

  Returns:
    Tensor with the cost of the op.
  """
  if op.type == 'DepthwiseConv2dNative':
    # num_alive_inputs may not always equals num_alive_outputs because the
    # input (e.g. the image) may not have a gamma regularizer. In this
    # case the computation is proportional only to num_alive_outputs.
    return batch_size * coeff * num_alive_outputs
  else:
    return batch_size * coeff * num_alive_inputs * num_alive_outputs
