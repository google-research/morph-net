"""Utility methods for working with OpHandler and tf.Operation."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import re

OP_TYPES_WITH_MULTIPLE_OUTPUTS = ('SplitV',)

# Dictionary mapping op type to input index of weights.
WEIGHTS_INDEX_DICT = {
    'Conv2D': 1,
    'Conv2DBackpropInput': 1,
    'DepthwiseConv2dNative': 1,
    'MatMul': 1
}


def get_input_ops(op, op_reg_manager, whitelist_indices=None):
  """Returns input ops for a given op.

  Filters constants and weight tensors.

  Args:
    op: tf.Operation to get inputs of.
    op_reg_manager: OpRegularizerManager to keep track of the grouping.
    whitelist_indices: Optional, indices of op.inputs that should be considered.

  Returns:
    List of tf.Operation that are the inputs to op.
  """
  # Ignore scalar or 1-D constant inputs.
  def is_const(tensor):
    return tensor.op.type == 'Const'
  def is_weight_tensor(i, op_type):
    return i == WEIGHTS_INDEX_DICT.get(op_type, -666)  # If op_type not in dict.

  # If op has a weight tensor as an input, remove it.
  inputs = list(op.inputs)

  whitelist_indices = whitelist_indices or range(len(inputs))
  filted_input_ops = []
  for i, tensor in enumerate(inputs):
    if (i not in whitelist_indices
        or is_weight_tensor(i, op.type)
        or is_const(tensor)
        or tensor.op not in op_reg_manager.ops):
      continue
    filted_input_ops.append(tensor.op)
  return filted_input_ops


def get_output_ops(op, op_reg_manager):
  """Returns output ops for a given op.

  Args:
    op: tf.Operation to get outputs of.
    op_reg_manager: OpRegularizerManager to keep track of the grouping.

  Returns:
    List of tf.Operation that are the outputs of op.
  """
  output_ops = []
  for output_tensor in op.outputs:
    for output_op in output_tensor.consumers():
      if output_op not in output_ops and output_op in op_reg_manager.ops:
        output_ops.append(output_op)
  return output_ops


def get_ops_without_groups(ops, op_reg_manager):
  """Returns ops without OpGroup.

  Args:
    ops: List of tf.Operation.
    op_reg_manager: OpRegularizerManager to keep track of the grouping.

  Returns:
    List of tf.Operation that do not have OpGroup assigned.
  """
  ops_without_groups = []
  for op in ops:
    op_slices = op_reg_manager.get_op_slices(op)
    for op_slice in op_slices:
      op_group = op_reg_manager.get_op_group(op_slice)
      if op_group is None:
        ops_without_groups.append(op)
        break

  return ops_without_groups


def remove_non_passthrough_ops(ops, op_reg_manager):
  """Removes non-passthrough ops from ops.

  Args:
    ops: List of tf.Operation.
    op_reg_manager: OpRegularizerManager to keep track of the grouping.

  Returns:
    List of tf.Operation of only passthrough ops in ops.
  """
  return [op for op in ops if op_reg_manager.is_passthrough(op)]


def group_op_with_inputs_and_outputs(op, input_op_slices, output_op_slices,
                                     aligned_op_slice_sizes, op_reg_manager):
  """Groups op with inputs and outputs if grouping is inconsistent.

  Args:
    op: tf.Operation.
    input_op_slices: List of list of OpSlice, with a list per input op.
    output_op_slices: List of list of OpSlice, with a list per output op.
    aligned_op_slice_sizes: List of integer OpSlice sizes.
    op_reg_manager: OpRegularizerManager to keep track of the grouping.

  Returns:
    Boolean indicating if grouping was inconsistent.
  """
  op_slices = op_reg_manager.get_op_slices(op)

  inconsistent_grouping = False
  # Group aligned OpSlice by iterating along each slice.
  for slice_index in range(len(aligned_op_slice_sizes)):
    op_group = op_reg_manager.get_op_group(op_slices[slice_index])
    output_op_slices_at_index = [output_op_slice[slice_index]
                                 for output_op_slice in output_op_slices]
    input_op_slices_at_index = [input_op_slice[slice_index]
                                for input_op_slice in input_op_slices]

    if op_group is None:
      # The current op does not have a group.  Group with inputs and outputs.
      op_reg_manager.group_op_slices(
          [op_slices[slice_index]] + input_op_slices_at_index
          + output_op_slices_at_index)
      continue

    if any([op_group != op_reg_manager.get_op_group(output_op_slice)
            for output_op_slice in output_op_slices_at_index]):
      # Some output OpSlice have different grouping.
      op_reg_manager.group_op_slices(
          [op_slices[slice_index]] + output_op_slices_at_index)
      # Refesh OpGroup before comparing with input groups.
      op_group = op_reg_manager.get_op_group(op_slices[slice_index])
      inconsistent_grouping = True

    if any([op_group != op_reg_manager.get_op_group(input_op_slice)
            for input_op_slice in input_op_slices_at_index]):
      # Some input OpSlice have different grouping.
      op_reg_manager.group_op_slices(
          [op_slices[slice_index]] + input_op_slices_at_index,
          omit_source_op_slices=_get_input_source_ops_to_omit(
              input_op_slices_at_index,
              op_slices[slice_index],
              op_reg_manager))
      inconsistent_grouping = True

  return inconsistent_grouping


def get_concat_input_op_slices(concat_ops, op_reg_manager):
  """Returns OpSlice for concat input ops to concatenate.

  For concat, all input OpSlice should be stacked to align with the concat
  OpSlice.  Also, the last input is the axis which should be omitted.

  Args:
    concat_ops: List of tf.Operation which provide inputs to the concat op.
    op_reg_manager: OpRegularizerManager that tracks the slicing.

  Returns:
    List of list of OpSlice, where the outer list only has 1 element, and the
    inner list is the concatenation of input OpSlice.
  """
  concat_input_op_slices = []
  for concat_op in concat_ops:
    concat_input_op_slices.extend(op_reg_manager.get_op_slices(concat_op))

  return [concat_input_op_slices]


def get_op_slices(ops, op_reg_manager):
  """Returns list of OpSlice per op in a list of ops.

  Args:
    ops: List of tf.Operation.
    op_reg_manager: OpRegularizerManager that tracks the slicing.

  Returns:
    List of list of OpSlice, where the outer list has a list per op, and the
    inner list is a list of OpSlice that compose the op.
  """
  op_slices = []
  for op in ops:
    op_slices.append(op_reg_manager.get_op_slices(op))
  return list(filter(None, op_slices))


def get_op_slice_sizes(op_slices):
  """Returns OpSlice sizes for a list of list of OpSlice.

  The outer list has an element per op, while the inner list is the list of
  OpSlice that compose the op.

  Args:
    op_slices: List of list of OpSlice.

  Returns:
    List of list of OpSlice sizes where the outer list has an entry per op.
  """
  op_slice_sizes = []
  for op in op_slices:
    op_slice_sizes.append([op_slice.slice.size for op_slice in op])

  return op_slice_sizes


def get_aligned_op_slice_sizes(op_slices, input_op_slices, output_op_slices):
  """Returns list of OpSlice sizes with aligned sizes.

  Given lists of OpSlice for an op and its inputs and outputs, returns the
  smallest list of slice sizes that aligns the slices.  For example, given an
  input of [[1, 2], [3]] representing a first op with slice sizes [1, 2] and a
  second op with op slice size [3], then the aligned slice sizes is [1, 2] to be
  compatible.  This means the second op would need to be sliced to match the
  aligned slice sizes.  As another example, given an input of [[2, 5], [3, 4]],
  both ops would need to be resliced.  The smallest list of slice sizes that
  aligns the 2 ops is [2, 1, 4].  Finally, consider the example
  [[5, 6, 7], [9, 4, 5], [18]], which returns [5, 4, 2, 2, 5].  Once the slice
  sizes are aligned, the corresponding slices are of matching size and can be
  grouped for the purpose of regularization.

  Given lists of OpSlice for an op and its inputs and outputs, returns the
  smallest list of slice sizes that aligns the slices.

  Args:
    op_slices: List of OpSlice for an op.
    input_op_slices: List of list of OpSlice, with a list per input op.
    output_op_slices: List of list of OpSlice, with a list per output op.

  Returns:
    List of integer slice sizes which is the smallest list of aligned sizes.
  """
  # TODO(a1): Create a helper class to manage list of list of OpSlice.
  input_op_slice_sizes = get_op_slice_sizes(input_op_slices)
  output_op_slices_sizes = get_op_slice_sizes(output_op_slices)
  op_slice_sizes = get_op_slice_sizes([op_slices])

  all_op_slice_sizes = (input_op_slice_sizes + output_op_slices_sizes
                        + op_slice_sizes)
  return get_aligned_sizes(all_op_slice_sizes)


def get_aligned_sizes(op_slice_sizes):
  """Returns list of OpSlice sizes with aligned sizes.

  Given a list of OpSlice sizes, returns the smallest list of slice sizes that
  aligns the slices.

  Args:
    op_slice_sizes: List of list of slice sizes, where the outer list has a list
      per op and the inner list is the integer slice sizes of the op.

  Returns:
    List of integer slice sizes which is the smallest list of aligned sizes.

  Raises:
    ValueError: If op_slice_sizes is empty.
    ValueError: If slice size lists do not have the same total size.
  """
  # Check for empty list.
  if not op_slice_sizes:
    raise ValueError('Cannot align empty op slice lists.')

  # Check that all ops have the same total size.
  total_slice_sizes = [
      get_total_slice_size(op_slice_size, 0, len(op_slice_size))
      for op_slice_size in op_slice_sizes]
  if total_slice_sizes.count(total_slice_sizes[0]) != len(total_slice_sizes):
    raise ValueError(
        'Total size for op slices do not match: %s' % op_slice_sizes)

  # Make local copy of op_slice_sizes for destruction.
  aligned_op_slice_sizes = [list(op_slice_size)
                            for op_slice_size in op_slice_sizes]
  slice_index = 0
  num_slices = _get_num_slices(op_slice_sizes)
  # Iterate slice by slice to check if slice sizes match across ops, or if they
  # need to be split further.
  while slice_index < num_slices:
    op_slices_at_index = [slice_size[slice_index]
                          for slice_size in aligned_op_slice_sizes]
    min_slice_size = min(op_slices_at_index)
    for op_index in range(len(aligned_op_slice_sizes)):
      old_size = aligned_op_slice_sizes[op_index][slice_index]
      if old_size != min_slice_size:
        # This OpSlice is bigger than the minimum, meaning this op needs to be
        # sliced again to match sizes.
        aligned_op_slice_sizes[op_index][slice_index] = min_slice_size
        aligned_op_slice_sizes[op_index].insert(
            slice_index + 1, old_size - min_slice_size)
    num_slices = _get_num_slices(aligned_op_slice_sizes)
    slice_index += 1
  return aligned_op_slice_sizes[0]


def _get_num_slices(op_slice_sizes):
  """Returns the number of slices in a list of OpSlice sizes.

  Args:
    op_slice_sizes: List of list of slice sizes, where the outer list has a list
      per op and the inner list is the slice sizes of the op.

  Returns:
    Integer max number of slices in the list of ops.
  """
  return max([len(slices) for slices in op_slice_sizes])


def reslice_concat_ops(concat_ops, aligned_op_slice_sizes, op_reg_manager):
  """Reslices concat ops according to aligned sizes.

  For concat, the input ops are concatenated which means the input op slice
  sizes must be concatenated when comparing to aligned slice sizes.  This is
  different from the output, where the output op slices can be directly compared
  to the aligned sizes.

  For example, consider a concatenation of OpA (size 3) and OpB (size 5) which
  is input into OpC (size 8, but slices of size [3, 3, 2] perhaps due to another
  downstream concat).  To group these ops, the input op slices need to be
  concatenated before aligning with the output op slices, which requires
  aligning ops slice sizes [[3, 5], [3, 3, 2]] which results in [3, 3, 2].
  Thus, OpB needs to be sliced into sizes [3, 2] in order to makes slice sizes
  compatible for grouping.

  Args:
    concat_ops: List of tf.Operation to slice.
    aligned_op_slice_sizes: List of integer slice sizes.
    op_reg_manager: OpRegularizerManager to keep track of slicing.

  Raises:
    ValueError: If concat op slice sizes do not match aligned op slice sizes.
  """
  concat_slice_index = 0
  for concat_op in concat_ops:
    concat_op_slices = op_reg_manager.get_op_slices(concat_op)
    concat_op_slice_sizes = get_op_slice_sizes([concat_op_slices])[0]
    if concat_op_slice_sizes == aligned_op_slice_sizes[
        concat_slice_index:concat_slice_index + len(concat_op_slice_sizes)]:
      # Slice sizes match so move on to the next op.
      concat_slice_index += len(concat_op_slice_sizes)
      continue
    else:
      # Slice sizes do not match.  The concat op needs to be resliced to match
      # the aligned sizes.
      slice_count = 1
      concat_op_size = sum(concat_op_slice_sizes)
      slice_size = get_total_slice_size(
          aligned_op_slice_sizes, concat_slice_index, slice_count)

      # Accumulate aligned slices until the sizes match the input op size.
      while concat_op_size > slice_size:
        slice_count += 1
        slice_size = get_total_slice_size(
            aligned_op_slice_sizes, concat_slice_index, slice_count)

      if concat_op_size != slice_size:
        raise ValueError('Could not properly slice op: %s' % concat_op)
      else:
        # Now concat_slice_index and slice_count specify the sublist of aligned
        # op slice sizes that match the current concat op.  Reslice the concat
        # op using the aligned sizes.
        op_reg_manager.slice_op(
            concat_op,
            aligned_op_slice_sizes[
                concat_slice_index:concat_slice_index + slice_count])
        concat_slice_index += slice_count


def get_total_slice_size(op_slice_sizes, index, slice_count):
  """Returns total size of a sublist of slices.

  Args:
    op_slice_sizes: List of integer slice sizes.
    index: Integer index specifying the start of the sublist.
    slice_count: Integer number of slices to include in the total size.

  Returns:
    Integer total size of the sublist of slices.
  """
  return sum(op_slice_sizes[index:index + slice_count])


def reslice_ops(ops, aligned_op_slice_sizes, op_reg_manager):
  """Reslices ops according to aligned sizes.

  Args:
    ops: List of tf.Operation to slice.
    aligned_op_slice_sizes: List of integer slice sizes.
    op_reg_manager: OpRegularizerManager to keep track of slicing.
  """
  for op_to_slice in ops:
    op_slice_sizes = [
        op_slice.slice.size
        for op_slice in op_reg_manager.get_op_slices(op_to_slice)]
    if op_slice_sizes and op_slice_sizes != aligned_op_slice_sizes:
      op_reg_manager.slice_op(op_to_slice, aligned_op_slice_sizes)


def _get_source_op_slices(op_slices, op_reg_manager):
  """Returns list of OpSlice that are sources.

  Args:
    op_slices: List of OpSlice.
    op_reg_manager: OpRegularizerManager to keep track of slicing.

  Returns:
    List of OpSlice that are sources.
  """
  op_groups = [op_reg_manager.get_op_group(op_slice)
               for op_slice in op_slices
               if op_reg_manager.get_op_group(op_slice) is not None]
  # pylint: disable=g-complex-comprehension
  return list(set([source_op_slice for op_group in op_groups
                   for source_op_slice in op_group.source_op_slices]))
  # pylint: enable=g-complex-comprehension


def _get_input_source_ops_to_omit(input_op_slices, op_slice,
                                  op_reg_manager):
  """Returns list of input OpSlice to omit as sources in a new group.

  If op_slice contains a source, the new group should ignore sources from
  the input OpSlice (e.g. Conv2D input to batch norm).  Otherwise, return an
  empty list so that the new group propagates the sources from the input.

  Args:
    input_op_slices: List of OpSlice that are inputs to a grouping op.
    op_slice: OpSlice that is being assigned grouping.
    op_reg_manager: OpRegularizerManager to keep track of slicing.

  Returns:
    List of source input OpSlice to omit as sources in a new group.
  """
  source_op_slices = _get_source_op_slices([op_slice], op_reg_manager)
  # If op_slice has a source, it overrides the input sources.  Otherwise,
  # return an empty list so input sources are propagated to the new group.
  if source_op_slices:
    input_source_op_slices = _get_source_op_slices(
        input_op_slices, op_reg_manager)
    return [op_slice for op_slice in input_source_op_slices
            if op_slice not in source_op_slices]
  else:
    return []


def group_aligned_input_output_slices(
    op, input_ops_to_process, output_ops_to_process, input_op_slices,
    output_op_slices, aligned_op_slice_sizes, op_reg_manager):
  """Groups aligned OpSlice and reprocesses ungrouped ops.

  Assuming OpSlice of op have been aligned with input and output, groups the
  corresponding OpSlice based on whether all inputs or all outputs are grouped.
  Ungrouped ops are put into the queue for processing.

  1. If all inputs and outputs have groups, op is also grouped with them for
     consistency.
  2. If all inputs are grouped, op is grouped with inputs while ungrouped
     outputs are queued for processing.
  3. If all outputs are grouped and there is only 1 input, op is grouped with
     outputs while ungrouped inputs are queued for processing.
  4. If neither inputs or outputs are grouped, then all ungrouped ops are queued
     for processing as grouping for op currently cannot be resolved.

  Args:
    op: tf.Operation to determine grouping for.
    input_ops_to_process: List of tf.Operation of ungrouped input ops.
    output_ops_to_process: List of tf.Operation of ungrouped output ops.
    input_op_slices: List of list of OpSlice, with a list per input op.
    output_op_slices: List of list of OpSlice, with a list per output op.
    aligned_op_slice_sizes: List of integer slice sizes.
    op_reg_manager: OpRegularizerManager to keep track of grouping.
  """
  # If all inputs and outputs have groups, group slices with op for consistency.
  if not input_ops_to_process and not output_ops_to_process:
    group_op_with_inputs_and_outputs(
        op, input_op_slices, output_op_slices, aligned_op_slice_sizes,
        op_reg_manager)
  elif not input_ops_to_process:
    # All inputs are grouped, so group with inputs and process outputs.
    group_op_with_inputs_and_outputs(
        op, input_op_slices, [], aligned_op_slice_sizes, op_reg_manager)
    op_reg_manager.process_ops(output_ops_to_process)
  else:
    # Both inputs and outputs need to be grouped first.
    op_reg_manager.process_ops(output_ops_to_process + input_ops_to_process)
    op_reg_manager.process_ops_last([op])


def get_op_size(op):
  """Returns output size of an op.

  The output size of an op is typically the last dimension of the output tensor.
  For example, this is the number of output channels of a convolution.  If the
  op has no shape (i.e. a constant), then return 0.

  Args:
    op: A tf.Operation.

  Returns:
    Integer output size of the op.
  """
  if op.type in OP_TYPES_WITH_MULTIPLE_OUTPUTS:
    return sum([output_tensor.shape.as_list()[-1]
                for output_tensor in op.outputs])
  # For regular ops, return the size of the first output tensor.
  shape = op.outputs[0].shape.as_list()
  if shape:
    return shape[-1]
  return 0


def separate_same_size_ops(reference_op, ops):
  """Separate ops by comparing to size of op.

  Ops of size 0 are dropped.

  Args:
    reference_op: tf.Operation which is the reference size.
    ops: List of tf.Operation to compare to the reference op size.

  Returns:
    A 2-tuple of lists of tf.Operation.  The first element is a list of
    tf.Operation which match the size of the reference op.  The second element
    is a list of tf.Operation that do not match the size of the reference op.
  """
  same_size_ops = []
  different_size_ops = []
  reference_op_size = get_op_size(reference_op)
  for op in ops:
    op_size = get_op_size(op)
    if op_size == reference_op_size:
      same_size_ops.append(op)
    elif op_size > 0:
      different_size_ops.append(op)

  return (same_size_ops, different_size_ops)


def group_match(regex, op_slices):
  """Returns True if the regex is found in the op name of any Opslice.

  Args:
    regex: A string regex.
    op_slices: List of OpRegularizerManager.OpSlice.

  Returns:
    True if the regex is found in the op name of any op in op_slices.
  """
  # If no regex, then group does not match.
  if not regex:
    return False

  # Check if any OpSlice in the group matches the regex.
  matches = [re.search(regex, op_slice.op.name) for op_slice in op_slices]
  return any(matches)
