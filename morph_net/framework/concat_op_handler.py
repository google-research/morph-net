"""OpHandler implementation for concat operations."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from morph_net.framework import grouping_op_handler
from morph_net.framework import op_handler
from morph_net.framework import op_handler_util


# The axis arg of tf.concat is a constant tensor stored in the last element of
# op.inputs. This function access the value of that tensor.
def _get_concat_op_axis(op):
  return op.inputs[-1].op.get_attr('value').int_val[0]


class ConcatOpHandler(op_handler.OpHandler):
  """OpHandler implementation for concat operations."""

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
    concat_axis = _get_concat_op_axis(op)
    # Need to figure out the rank to know if axis is last.
    rank = len(op.inputs[0].shape)  # Rank of the first input.

    if concat_axis != -1 and concat_axis != rank - 1:
      # Concat is actually grouping inputs!
      handler = grouping_op_handler.GroupingOpHandler()
      handler.assign_grouping(op, op_reg_manager)
      return

    # If concat is of the last dimension, this is a `standard` concat.
    # TODO(a1): Consider refactoring this duplicated logic.
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

    # Only group with output ops that have the same size.  Process the ops that
    # have mismatched size.
    input_ops_to_group = input_ops
    input_ops_to_process = input_ops_without_group
    output_ops_to_group, output_ops_to_process = (
        op_handler_util.separate_same_size_ops(op, output_ops))

    # Also process ungrouped ops.
    output_ops_to_process.extend(output_ops_without_group)

    # Populate OpSlice data for all relevant ops.
    concat_op_slices = op_reg_manager.get_op_slices(op)
    input_op_slices, output_op_slices = self._get_input_output_op_slices(
        input_ops_to_group, output_ops_to_group, op_reg_manager)

    # Align op slices sizes if needed.
    aligned_op_slice_sizes = op_handler_util.get_aligned_op_slice_sizes(
        concat_op_slices, input_op_slices, output_op_slices)
    op_handler_util.reslice_concat_ops(
        input_ops_to_group, aligned_op_slice_sizes, op_reg_manager)
    op_handler_util.reslice_ops(
        output_ops_to_group + [op], aligned_op_slice_sizes, op_reg_manager)

    # Repopulate OpSlice data, as ops may have been resliced.
    input_op_slices, output_op_slices = self._get_input_output_op_slices(
        input_ops_to_group, output_ops_to_group, op_reg_manager)

    # Group aligned OpSlice.
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
    input_op_slices = op_handler_util.get_concat_input_op_slices(
        input_ops, op_reg_manager)
    output_op_slices = op_handler_util.get_op_slices(
        [output_op for output_op in output_ops
         if op_reg_manager.is_passthrough(output_op)], op_reg_manager)
    return (input_op_slices, output_op_slices)
