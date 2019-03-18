"""OpHandler implementation for DepthToSpace operations."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from morph_net.framework import op_handler
from morph_net.framework import op_handler_util


class DepthToSpaceOpHandler(op_handler.OpHandler):
  """OpHandler implementation for DepthToSpace operations."""

  @property
  def is_source_op(self):
    return False

  @property
  def is_passthrough(self):
    return False

  def assign_grouping(self, op, op_reg_manager):
    """Assign grouping to the given op and updates the manager.

    Args:
      op: tf.Operation to assign grouping to.
      op_reg_manager: OpRegularizerManager to keep track of the grouping.
    """
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

    # Only group with ops that have the same size.  Defer the ops that have
    # mismatched size.
    input_ops_to_group = input_ops
    output_ops_to_group, output_ops_to_defer = (
        op_handler_util.separate_same_size_ops(op, output_ops))

    # Also defer ungrouped ops.
    input_ops_to_defer = input_ops_without_group
    for output_op_without_group in output_ops_without_group:
      if output_op_without_group not in output_ops_to_defer:
        output_ops_to_defer.append(output_op_without_group)

    # Only slice and merge if all inputs are grouped.
    if input_ops_to_defer:
      op_reg_manager.process_ops(input_ops_to_defer)
      return

    block_size = op.get_attr('block_size')
    block_group = block_size * block_size

    # For DepthToSpace, slice ops into individual channels before mapping.  For
    # example, this op might reshape a tensor [N, H, W, 4] -> [N, 2H, 2W, 1]
    # where the 4 input channels are mapped to 1 output channel.  Thus, slice
    # the input into individual OpSlice in order to group.
    assert len(input_ops_to_group) == 1
    input_op = input_ops_to_group[0]
    op_handler_util.reslice_ops(
        input_ops, [1] * op_handler_util.get_op_size(input_op),
        op_reg_manager)
    op_handler_util.reslice_ops(
        [op] + output_ops_to_group, [1] * op_handler_util.get_op_size(op),
        op_reg_manager)

    # Repopulate OpSlice data.
    op_slices = op_reg_manager.get_op_slices(op)
    input_op_slices = op_handler_util.get_op_slices(input_ops, op_reg_manager)

    # Group blocks of input channels with output channels based on block group.
    # For block_size B, the block group is B * B.  For example, if the input
    # tensor is [N, H, W, 18] with block_size 3, the output tensor is
    # [N, 3H, 3W, 2] where block_size * block_size number of channels are mapped
    # to space values (i.e. 3H and 3W).  See Tensorflow documentation for
    # additional details.
    for i, op_slice in enumerate(op_slices):
      for input_op_slice in input_op_slices:
        op_reg_manager.group_op_slices(
            input_op_slice[i * block_group:(i + 1) * block_group] + [op_slice])

    # Process deferred ops.
    if input_ops_to_defer or output_ops_to_defer:
      op_reg_manager.process_ops(output_ops_to_defer + input_ops_to_defer)

  def create_regularizer(self, _):
    raise NotImplementedError('Not a source op.')
