"""OpHandler implementation for grouping operations.

This is the default OpHandler for ops without a specifically assigned OpHandler.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from morph_net.framework import op_handler
from morph_net.framework import op_handler_util


class GroupingOpHandler(op_handler.OpHandler):
  """OpHandler implementation for grouping operations."""

  def __init__(self, grouping_indices=None):
    """Creates a GroupingOpHandler.

    Args:
      grouping_indices: A list of indices which define which of the inputs the
        handler should group. The goal is to allow the handler to ignore indices
        which hold tensors that should not be grouped, e.g. The kernel size of a
        convolution should not be grouped with the input tensor.
    """
    self._grouping_indices = grouping_indices

  @property
  def is_source_op(self):
    return False

  @property
  def is_passthrough(self):
    return True

  def assign_grouping(self, op, op_reg_manager):
    """Assign grouping to the given op and updates the manager.

    Args:
      op: tf.Operation to assign grouping to.
      op_reg_manager: OpRegularizerManager to keep track of the grouping.
    """
    # Check if all input ops have groups, or tell the manager to process them.
    input_ops = op_handler_util.get_input_ops(op, op_reg_manager,
                                              self._grouping_indices)
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
    input_ops_to_group, input_ops_to_process = (
        op_handler_util.separate_same_size_ops(op, input_ops))
    output_ops_to_group, output_ops_to_process = (
        op_handler_util.separate_same_size_ops(op, output_ops))

    # Remove broadcast ops.
    input_ops_to_process = [input_op for input_op in input_ops_to_process
                            if not self._is_broadcast(input_op, op_reg_manager)]

    # Also process ungrouped ops.
    for input_op_without_group in input_ops_without_group:
      if input_op_without_group not in input_ops_to_process:
        input_ops_to_process.append(input_op_without_group)
    for output_op_without_group in output_ops_without_group:
      if output_op_without_group not in output_ops_to_process:
        output_ops_to_process.append(output_op_without_group)

    # Align op slice sizes if needed.
    op_slices = op_reg_manager.get_op_slices(op)
    input_op_slices = op_handler_util.get_op_slices(
        input_ops_to_group, op_reg_manager)
    output_op_slices = op_handler_util.get_op_slices(
        output_ops_to_group, op_reg_manager)
    aligned_op_slice_sizes = op_handler_util.get_aligned_op_slice_sizes(
        op_slices, input_op_slices, output_op_slices)
    op_handler_util.reslice_ops(input_ops_to_group + [op] + output_ops_to_group,
                                aligned_op_slice_sizes, op_reg_manager)

    # Repopulate OpSlice data, as ops may have been resliced.
    input_op_slices, output_op_slices = self._get_input_output_op_slices(
        input_ops_to_group, output_ops_to_group, op_reg_manager)

    # Group with inputs and outputs.
    op_handler_util.group_aligned_input_output_slices(
        op, input_ops_to_process, output_ops_to_process, input_op_slices,
        output_op_slices, aligned_op_slice_sizes, op_reg_manager)

  def create_regularizer(self, _):
    raise NotImplementedError('Not a source op.')

  def _get_input_output_op_slices(self, input_ops, output_ops, op_reg_manager):
    """Returns op slices for inputs and outputs.

    Args:
      input_ops: List of tf.Operation.
      output_ops: List of tf.Operation.
      op_reg_manager: OpRegularizerManager to keep track of the grouping.

    Returns:
      Tuple of (input_op_slices, output_op_slices), where each element is a list
      of list of OpSlice with a list per op.
    """
    input_op_slices = op_handler_util.get_op_slices(input_ops, op_reg_manager)
    output_op_slices = op_handler_util.get_op_slices(output_ops, op_reg_manager)
    return (input_op_slices, output_op_slices)

  def _is_broadcast(self, op, op_reg_manager):
    """Returns True if op is broadcast.

    Args:
      op: A tf.Operation.
      op_reg_manager: OpRegularizerManager to keep track of the grouping.

    Returns:
      A boolean indicating if op is broadcast.
    """
    op_slices = op_reg_manager.get_op_slices(op)
    op_groups = [op_reg_manager.get_op_group(op_slice)
                 for op_slice in op_slices]
    return op_handler_util.get_op_size(op) == 1 and all(op_groups)
