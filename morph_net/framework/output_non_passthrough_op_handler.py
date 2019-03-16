"""OpHandler for OutputNonPassthrough ops.

OutputNonPassthrough ops take their regularizer from the output and do not
passthrough the regularizer to their input. This is the default OpHandler for
ops like Conv2D and MatMul when L1-gamma regularization is used.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from morph_net.framework import op_handler
from morph_net.framework import op_handler_util


class OutputNonPassthroughOpHandler(op_handler.OpHandler):
  """OpHandler implementation for OutputNonPassthrough operations.

   These ops take their regularizer from the output and do not
   passthrough the regularizer to their input.
  """

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

    # Only group with ops that have the same size.  Process the ops that have
    # mismatched size.
    output_ops_to_group, output_ops_to_process = (
        op_handler_util.separate_same_size_ops(op, output_ops))

    # Also process ungrouped ops.
    input_ops_to_process = input_ops_without_group
    output_ops_to_process.extend(output_ops_without_group)

    # Align op slice sizes if needed.
    op_slices = op_reg_manager.get_op_slices(op)
    output_op_slices = op_handler_util.get_op_slices(
        output_ops_to_group, op_reg_manager)
    aligned_op_slice_sizes = op_handler_util.get_aligned_op_slice_sizes(
        op_slices, [], output_op_slices)
    op_handler_util.reslice_ops([op] + output_ops_to_group,
                                aligned_op_slice_sizes, op_reg_manager)

    # TODO(a1): Consider refactoring this method.
    # Repopulate OpSlice data, as ops may have been resliced.
    output_op_slices = self._get_output_op_slices(
        output_ops_to_group, op_reg_manager)

    # Group with inputs and outputs.
    op_handler_util.group_op_with_inputs_and_outputs(
        op, [], output_op_slices, aligned_op_slice_sizes,
        op_reg_manager)

    # Reprocess ops.
    op_reg_manager.process_ops(output_ops_to_process + input_ops_to_process)

  def _group_with_output_slices(
      self, op, output_op_slices, op_slices, op_reg_manager):
    """Groups OpSlice of current op with output ops.

    Assuming OpSlice of op have been aligned with output, groups the
    corresponding OpSlice.

    Args:
      op: tf.Operation to determine grouping for.
      output_op_slices: List of list of OpSlice, with a list per output op.
      op_slices: List of OpSlice for current op.
      op_reg_manager: OpRegularizerManager to keep track of grouping.

    Raises:
      ValueError: If sizes for current and output op slices are not the same.
    """
    # Assert that op slices for output and current op are aligned.
    output_op_slices_sizes = op_handler_util.get_op_slice_sizes(
        output_op_slices)
    op_slice_sizes = op_handler_util.get_op_slice_sizes([op_slices])

    if op_slice_sizes != output_op_slices_sizes:
      raise ValueError('Current op and output op have differing slice '
                       'sizes: {}, {}'.format(
                           op_slice_sizes, output_op_slices_sizes))

    op_handler_util.group_op_with_inputs_and_outputs(
        op, [], output_op_slices, op_slice_sizes, op_reg_manager)

  def _get_output_op_slices(self, output_ops, op_reg_manager):
    """Returns op slices for outputs.

    Args:
      output_ops: List of tf.Operation.
      op_reg_manager: OpRegularizerManager to keep track of the grouping.

    Returns:
      A list of list of OpSlice with a list per output op.
    """
    return op_handler_util.get_op_slices(output_ops, op_reg_manager)

  def create_regularizer(self, _):
    raise NotImplementedError('Not a source op.')
