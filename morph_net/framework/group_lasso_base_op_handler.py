"""Base OpHandler for ops that use group lasso regularizer.

This OpHandler should not be called directly. It is a virtual base class
for regularization source OpHandlers that use Group Lasso as their regularizer.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import abc

from morph_net.framework import op_handler
from morph_net.framework import op_handler_util
from morph_net.framework import tpu_util
from morph_net.op_regularizers import group_lasso_regularizer


class GroupLassoBaseSourceOpHandler(op_handler.OpHandler):
  """Base OpHandler for source ops that use Group Lasso."""

  __metaclass__ = abc.ABCMeta

  def __init__(self, threshold, l1_fraction=0.0):
    """Instantiate an instance.

    Args:
      threshold: Float scalar used as threshold for GroupLassoRegularizer.
      l1_fraction: Float scalar used as l1_fraction for GroupLassoRegularizer.
    """
    self._threshold = threshold
    self._l1_fraction = l1_fraction

  @abc.abstractmethod
  def _reduce_dims(self, op):
    # Reduction dimensions for Group Lasso.
    pass

  @property
  def is_source_op(self):
    return True

  @property
  def is_passthrough(self):
    return False

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

    # Check if all input ops have groups, or tell the manager to process them.
    input_ops = op_handler_util.get_input_ops(op, op_reg_manager)
    input_ops_without_group = op_handler_util.get_ops_without_groups(
        input_ops, op_reg_manager)

    # Check if all output ops have groups, or tell the manager to process them.
    output_ops = op_handler_util.get_output_ops(op, op_reg_manager)
    output_ops_without_group = op_handler_util.get_ops_without_groups(
        output_ops, op_reg_manager)

    # Remove non-passthrough ops from outputs ops to group with.
    output_ops = op_handler_util.remove_non_passthrough_ops(
        output_ops, op_reg_manager)

    # Only group with ops that have the same size.  Process the ops that have
    # mismatched size.
    output_ops_to_group, output_ops_to_process = (
        op_handler_util.separate_same_size_ops(op, output_ops))

    # Also process ungrouped ops.
    input_ops_to_process = input_ops_without_group
    output_ops_to_process.extend(output_ops_without_group)

    # Align op slice sizes if needed.
    output_op_slices = op_handler_util.get_op_slices(
        output_ops_to_group, op_reg_manager)
    aligned_op_slice_sizes = op_handler_util.get_aligned_op_slice_sizes(
        op_slices, [], output_op_slices)
    op_handler_util.reslice_ops([op] + output_ops_to_group,
                                aligned_op_slice_sizes, op_reg_manager)

    # Repopulate OpSlice data, as ops may have been resliced.
    output_op_slices = op_handler_util.get_op_slices(
        output_ops_to_group, op_reg_manager)

    # Group with outputs.
    op_handler_util.group_op_with_inputs_and_outputs(
        op, [], output_op_slices, aligned_op_slice_sizes,
        op_reg_manager)

    # Reprocess ops.
    op_reg_manager.process_ops(output_ops_to_process + input_ops_to_process)

  def create_regularizer(self, op_slice):
    """Create a regularizer for this conv2d OpSlice.

    Args:
      op_slice: op_regularizer_manager.OpSlice that is a conv2d OpSlice.

    Returns:
      OpRegularizer for this conv2d op.
    """
    start_index = op_slice.slice.start_index
    size = op_slice.slice.size
    weights = op_slice.op.inputs[1]  # Input 1 are the weights.
    weights = tpu_util.maybe_convert_to_variable(weights)
    reduce_dims = self._reduce_dims(op_slice.op)
    rank = len(weights.shape.as_list())
    assert rank == len(reduce_dims) + 1

    def _slice_weights():
      """Slices the weight tensor according to op_slice information."""
      if rank == 2:
        if reduce_dims[0] == 0:
          return weights[:, start_index:start_index + size]
        else:
          return weights[start_index:start_index + size, :]
      if rank == 3:
        if 2 not in reduce_dims:
          return weights[:, :, start_index:start_index + size]
        if 1 not in reduce_dims:
          return weights[:, start_index:start_index + size, :]
        if 0 not in reduce_dims:
          return weights[start_index:start_index + size, :, :]
      if rank == 4:
        if 3 not in reduce_dims:
          return weights[:, :, :, start_index:start_index + size]
        if 2 not in reduce_dims:
          return weights[:, :, start_index:start_index + size, :]
        if 1 not in reduce_dims:
          return weights[:, start_index:start_index + size, :, :]
        if 0 not in reduce_dims:
          return weights[start_index:start_index + size, :, :, :]
      raise ValueError('Unsupported rankd or bad reduce_dim')

    weight_tensor = _slice_weights()

    # If OpSlice size matches tensor size, use the entire tensor.  Otherwise,
    # slice the tensor accordingly.
    return group_lasso_regularizer.GroupLassoRegularizer(
        weight_tensor=weight_tensor,
        reduce_dims=self._reduce_dims(op_slice.op),
        threshold=self._threshold,
        l1_fraction=self._l1_fraction)
