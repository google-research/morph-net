"""OpHandler implementation for leaf operations.

A leaf operation terminates the OpRegularizerManager graph traversal.  This is
typically network inputs, constants, variables, etc.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from morph_net.framework import op_handler
from morph_net.framework import op_handler_util


class LeafOpHandler(op_handler.OpHandler):
  """OpHandler implementation for leaf operations."""

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
    for output_op_without_group in output_ops_without_group:
      if output_op_without_group not in output_ops_to_process:
        output_ops_to_process.append(output_op_without_group)

    # Align op slice sizes if needed.
    op_slices = op_reg_manager.get_op_slices(op)
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
    op_handler_util.group_aligned_input_output_slices(
        op, [], output_ops_to_process, [], output_op_slices,
        aligned_op_slice_sizes, op_reg_manager)

  def create_regularizer(self, _):
    raise NotImplementedError('Not a source op.')
