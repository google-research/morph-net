"""OpHandler for logistic-sigmoid gating ops that are regularizer sources.

This OpHandler is used when logistic-sigmoid gating probabilities or masks
are considered regularizers.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from morph_net.framework import grouping_op_handler
from morph_net.framework import tpu_util
from morph_net.op_regularizers import prob_gating_regularizer


class LogisticSigmoidSourceOpHandler(grouping_op_handler.GroupingOpHandler):
  """OpHandler implementation for logistic-sigmoid gating source operations."""

  def __init__(self, regularize_on_mask=True,
               alive_threshold=0.1, mask_as_alive_vector=True):
    """Instantiate an instance.

    Args:
      regularize_on_mask: Bool. If True uses the binary mask as the
          regularization vector. Else uses the probability vector.
      alive_threshold: Float. Threshold below which values are considered dead.
        This can be used both when mask_as_alive_vector is True and then the
        threshold is used to binarize the sampled values and
        when mask_as_alive_vector is False, and then the threshold is on the
        channel probability.
      mask_as_alive_vector: Bool. If True use the thresholded sampled mask
        as the alive vector. Else, use thresholded probabilities from the
        logits.
    """
    super(LogisticSigmoidSourceOpHandler, self).__init__()
    self._regularize_on_mask = regularize_on_mask
    self._alive_threshold = alive_threshold
    self._mask_as_alive_vector = mask_as_alive_vector

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

    super(LogisticSigmoidSourceOpHandler, self).assign_grouping(
        op, op_reg_manager)

  def create_regularizer(self, op_slice):
    """Create a regularizer for this logistic-sigmoid gating OpSlice.

    Args:
      op_slice: op_regularizer_manager.OpSlice that is a batch norm OpSlice.

    Returns:
      OpRegularizer for this batch norm op.
    """
    start_index = op_slice.slice.start_index
    size = op_slice.slice.size
    logits = op_slice.op.inputs[0]  # Input 0 are the logits.
    mask = op_slice.op.outputs[0]

    # If OpSlice size matches tensor size, use the entire tensor.  Otherwise,
    # slice the tensor accordingly.
    mask = tpu_util.read_from_variable(mask)

    if start_index == 0 and size == logits.shape.as_list()[-1]:
      return prob_gating_regularizer.ProbGatingRegularizer(
          logits, mask,
          regularize_on_mask=self._regularize_on_mask,
          alive_threshold=self._alive_threshold,
          mask_as_alive_vector=self._mask_as_alive_vector)
    else:
      logits = tpu_util.maybe_convert_to_variable(logits)
      return prob_gating_regularizer.ProbGatingRegularizer(
          logits[start_index:start_index + size],
          mask[start_index:start_index + size],
          regularize_on_mask=self._regularize_on_mask,
          alive_threshold=self._alive_threshold,
          mask_as_alive_vector=self._mask_as_alive_vector)
