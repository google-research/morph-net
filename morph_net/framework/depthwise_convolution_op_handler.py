"""OpHandler implementation for depthwise convolution."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from morph_net.framework import grouping_op_handler
from morph_net.framework import op_handler_util


class DepthwiseConvolutionOpHandler(grouping_op_handler.GroupingOpHandler):
  """OpHandler implementation for depthwise convolution."""

  def assign_grouping(self, op, op_reg_manager):
    """Assign grouping to the given op and updates the manager.

    Args:
      op: tf.Operation to assign grouping to.
      op_reg_manager: OpRegularizerManager to keep track of the grouping.
    """
    assert op.type == 'DepthwiseConv2dNative'

    # Get output size.
    output_size = op_handler_util.get_op_size(op)

    # Get input size.
    input_size = op_handler_util.get_op_size(op.inputs[0].op)

    # Take depth_multiplier from size of weight tensor.
    depth_multiplier = op.inputs[1].shape.as_list()[-1]

    if depth_multiplier == 1:
      super(DepthwiseConvolutionOpHandler, self).assign_grouping(
          op, op_reg_manager)
      return

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
    # mismatched size.  For the input, we hardcode that inputs[0] is a normal
    # input while inputs[1] is the depthwise filter.
    input_ops_to_group = [input_ops[0]]
    input_ops_to_process = input_ops_without_group
    output_ops_to_group, output_ops_to_process = (
        op_handler_util.separate_same_size_ops(op, output_ops))

    # Also process ungrouped ops.
    for output_op_without_group in output_ops_without_group:
      if output_op_without_group not in output_ops_to_process:
        output_ops_to_process.append(output_op_without_group)

    # Slice ops into individual channels.  For example, consider 3 input
    # channels and depth_multiplier = 2.  Let the input channels be [0, 1, 2]
    # and the output channels be [3, 4, 5, 6, 7, 8].  The channels should be
    # individually sliced and grouped with consecutive groups of size
    # depth_multiplier.  Thus, this would end up grouping [0, 0, 1, 1, 2, 2] and
    # [3, 4, 5, 6, 7, 8] into groups (0, 3, 4), (1, 5, 6), and (2, 7, 8).
    aligned_op_slice_sizes = [1] * output_size
    op_handler_util.reslice_ops(
        input_ops_to_group, [1] * input_size, op_reg_manager)
    op_handler_util.reslice_ops(
        [op] + output_ops_to_group, aligned_op_slice_sizes, op_reg_manager)

    # Rearrange OpSlice to align input and output.
    input_op_slices, output_op_slices = (
        self._get_depth_multiplier_input_output_op_slices(
            input_ops_to_group, input_size, output_ops_to_group,
            op_reg_manager, depth_multiplier))

    # Group with inputs and outputs.
    op_handler_util.group_aligned_input_output_slices(
        op, input_ops_to_process, output_ops_to_process, input_op_slices,
        output_op_slices, aligned_op_slice_sizes, op_reg_manager)

  def _get_depth_multiplier_input_output_op_slices(
      self, input_ops, input_size, output_ops, op_reg_manager,
      depth_multiplier):
    """Returns op slices for inputs and outputs.

    Args:
      input_ops: List of tf.Operation.
      input_size: Integer number of input channels.
      output_ops: List of tf.Operation.
      op_reg_manager: OpRegularizerManager to keep track of the grouping.
      depth_multiplier: Integer indicating how many times each input channel
        should be replicated.  Must be positive.

    Returns:
      Tuple of (input_op_slices, output_op_slices), where each element is a list
      of list of OpSlice with a list per op.
    """
    input_op_slices = op_handler_util.get_op_slices(input_ops, op_reg_manager)

    # Each input OpSlice needs to be replicated N times where N is
    # depth_multiplier.
    depth_multiplier_input_op_slices = []
    for input_op in input_op_slices:
      slices = []
      for op_slice in input_op:
        slices.extend([op_slice] * depth_multiplier)
      depth_multiplier_input_op_slices.append(slices)

    output_op_slices = op_handler_util.get_op_slices(output_ops, op_reg_manager)

    return (depth_multiplier_input_op_slices, output_op_slices)
