"""OpHandler implementation for batch norm ops that are regularizer sources.

This OpHandler is used when batch norm gammas are considered regularizers.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from morph_net.framework import grouping_op_handler
from morph_net.framework import tpu_util
from morph_net.op_regularizers import gamma_l1_regularizer


class BatchNormSourceOpHandler(grouping_op_handler.GroupingOpHandler):
  """OpHandler implementation for batch norm source operations."""

  def __init__(self, gamma_threshold):
    """Instantiate an instance.

    Args:
      gamma_threshold: Float scalar, the threshold above which a gamma is
        considered alive.
    """
    super(BatchNormSourceOpHandler, self).__init__()
    self._gamma_threshold = gamma_threshold

  @property
  def is_source_op(self):
    return True

  def assign_grouping(self, op, op_reg_manager):
    """Assign grouping to the given op and updates the manager.

    Args:
      op: tf.Operation to assign grouping to.
      op_reg_manager: OpRegularizerManager to keep track of the grouping.
    """
    # This is a source op so begin by getting the OpGroup or creating one.
    op_slices = op_reg_manager.get_op_slices(op)
    for op_slice in op_slices:
      op_group = op_reg_manager.get_op_group(op_slice)
      if op_group is None:
        op_reg_manager.create_op_group_for_op_slice(op_slice)

    super(BatchNormSourceOpHandler, self).assign_grouping(op, op_reg_manager)

  def create_regularizer(self, op_slice):
    """Create a regularizer for this batch norm OpSlice.

    Args:
      op_slice: op_regularizer_manager.OpSlice that is a batch norm OpSlice.

    Returns:
      OpRegularizer for this batch norm op.
    """
    start_index = op_slice.slice.start_index
    size = op_slice.slice.size
    gamma = op_slice.op.inputs[1]  # Input 1 is gamma.

    # If OpSlice size matches tensor size, use the entire tensor.  Otherwise,
    # slice the tensor accordingly.
    if start_index == 0 and size == gamma.shape.as_list()[-1]:
      return gamma_l1_regularizer.GammaL1Regularizer(
          gamma, self._gamma_threshold)
    else:
      # Note: this conversion is also attempted inside GammaL1Regularizer
      # because it may be invoked from another call site.
      gamma = tpu_util.maybe_convert_to_variable(gamma)
      return gamma_l1_regularizer.GammaL1Regularizer(
          gamma[start_index:start_index + size], self._gamma_threshold)
