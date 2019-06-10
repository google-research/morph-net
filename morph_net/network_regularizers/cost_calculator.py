"""CostCalculator that computes network cost or regularization loss."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf


CONV2D_OPS = ('Conv2D', 'Conv2DBackpropInput', 'DepthwiseConv2dNative')
CONV3D_OPS = ('Conv3D',)
CONV_OPS = CONV2D_OPS + CONV3D_OPS
FLOP_OPS = CONV_OPS + ('MatMul',)
SUPPORTED_OPS = FLOP_OPS + ('Add', 'AddN', 'ConcatV2', 'FusedBatchNorm',
                            'FusedBatchNormV2', 'FusedBatchNormV3', 'Mul',
                            'Relu', 'Relu6', 'Sum')


class CostCalculator(object):
  """CostCalculator that calculates resource cost/loss for a network."""

  def __init__(self, op_regularizer_manager, resource_function):
    """Creates an instance.

    Args:
      op_regularizer_manager: OpRegularizerManager that contains the
        OpRegularizer for each op in the network.
      resource_function: Callable that returns the resource (e.g. FLOP) cost or
        loss for an op.  The function signature is:
          op; A tf.Operation.
          is_regularization; Boolean indicating whether to calculate
            regularization loss.  If False, calculate cost instead.
          num_alive_inputs; Scalar Tensor indicating how many input channels are
            considered alive.
          num_alive_outputs; Scalar Tensor indicating how many output channels
            are considered alive.
          reg_inputs; Scalar Tensor which is the sum over the input
            regularization vector.
          reg_outputs; Scalar Tensor which is the sum over the output
            regularization vector.
          batch_size; Integer batch size to calculate cost/loss for.
    """
    self._manager = op_regularizer_manager
    self._resource_function = resource_function

  def _get_cost_or_regularization_term(self, is_regularization, ops=None):
    """Returns cost or regularization term for ops.

    Args:
      is_regularization: Boolean indicating whether to calculate regularization
        loss.  If False, calculate cost instead.
      ops: List of tf.Operation.  If None, calculates cost/regularization for
        all ops found by OpRegularizerManager.

    Returns:
      Cost or regularization term for ops as a tensor or float.
    """
    total = 0.0
    if not ops:
      ops = self._manager.ops
    for op in ops:
      if op.type not in SUPPORTED_OPS:
        continue

      # Get regularization and alive terms for input and output.
      input_tensor = get_input_activation(op)
      if op.type == 'ConcatV2':
        # For concat, the input and output regularization are identical but the
        # input is composed of multiple concatenated regularizers.  Thus, just
        # use the output regularizer as the input regularizer for simpler cost
        # calculation.
        input_tensor = op.outputs[0]
      input_op_reg = self._manager.get_regularizer(input_tensor.op)
      output_op_reg = self._manager.get_regularizer(op)
      num_alive_inputs = _count_alive(input_tensor, input_op_reg)
      num_alive_outputs = _count_alive(op.outputs[0], output_op_reg)
      reg_inputs = _sum_of_reg_vector(input_op_reg)
      reg_outputs = _sum_of_reg_vector(output_op_reg)

      total += self._resource_function(
          op, is_regularization, num_alive_inputs, num_alive_outputs,
          reg_inputs, reg_outputs)

    # If at least one supported op is present, type would be tensor, not float.
    if isinstance(total, float):
      # Tests rely on this function not raising exception in this case.
      tf.logging.warning('No supported ops found.')
    return total

  def get_cost(self, ops=None):
    """Returns cost for ops.

    Args:
      ops: List of tf.Operation.  If None, calculates cost/regularization for
        all ops found by OpRegularizerManager.

    Returns:
      Cost of ops as a tensor for float.
    """

    return self._get_cost_or_regularization_term(False, ops)

  def get_regularization_term(self, ops=None):
    """Returns regularization for ops.

    Args:
      ops: List of tf.Operation.  If None, calculates cost/regularization for
        all ops found by OpRegularizerManager.

    Returns:
      Regularization term of ops as a tensor or float.
    """
    return self._get_cost_or_regularization_term(True, ops)


def get_input_activation(op):
  """Returns the input to `op` that represents the activations.

  (as opposed to e.g. weights.)

  Args:
    op: A tf.Operation object with type in SUPPORTED_OPS.

  Returns:
    A tf.Tensor representing the input activations.

  Raises:
    ValueError: op type not supported.).
    ValueError: MatMul is used with transposition (unsupported).
  """
  if op.type not in SUPPORTED_OPS:
    raise ValueError('Op type %s is not supported.' % op.type)
  if op.type in ('Conv3D', 'Conv2D', 'DepthwiseConv2dNative'):
    return op.inputs[0]
  if op.type == 'Conv2DBackpropInput':
    return op.inputs[2]
  if op.type == 'MatMul':
    if op.get_attr('transpose_a') or op.get_attr('transpose_b'):
      raise ValueError('MatMul with transposition is not yet supported.')
    return op.inputs[0]
  return op.inputs[0]


def _count_alive(tensor, opreg):
  if opreg:
    return tf.reduce_sum(tf.cast(opreg.alive_vector, tf.float32))
  shape = tensor.shape.as_list()
  if shape:
    num_outputs = tensor.shape.as_list()[-1]
    if num_outputs is not None:
      return tf.constant(num_outputs, tf.float32)
  tf.logging.info('Unknown channel count in tensor %s', tensor)
  return tf.constant(0, tf.float32)


def _sum_of_reg_vector(opreg):
  if opreg:
    return tf.reduce_sum(opreg.regularization_vector)
  else:
    return tf.constant(0.0, tf.float32)
