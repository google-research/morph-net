"""A class for managing OpRegularizers."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections

from morph_net.framework import concat_and_slice_regularizers
from morph_net.framework import constant_op_regularizer
from morph_net.framework import grouping_regularizers
from morph_net.framework import op_handler_util
import tensorflow as tf

# Hardcoded limit for OpRegularizerManager to finish analyzing the network.
ITERATION_LIMIT = 1000000


# OpSlice represents a slice of a tf.Operation.
# op: A tf.Operation.
# slice: A Slice tuple containing the index and size of the slice.  If None, or
#   part of the tuple is None, then the OpSlice represents the entire op.
class OpSlice(collections.namedtuple('OpSlice', ['op', 'slice'])):

  def __str__(self):
    return '{} {}'.format(self.op.name, self.slice)

  __repr__ = __str__


# Slice represents the index and size of a slice.
# start_index: Integer specifying start index of the slice.
# size: Integer specifying number of elements in the slice.
class Slice(collections.namedtuple('Slice', ['start_index', 'size'])):

  def __str__(self):
    return '({}, {})'.format(self.start_index, self.size)

  __repr__ = __str__


class OpRegularizerManager(object):
  """A class for managing OpRegularizers."""

  def __init__(
      self,
      ops,
      op_handler_dict=None,
      create_grouping_regularizer=grouping_regularizers.MaxGroupingRegularizer,
      force_group=None,
      regularizer_blacklist=None,
      input_boundary=None,
      iteration_limit=ITERATION_LIMIT):
    """Creates an instance of OpRegularizerManager.

    Several internal data structures are initialized which are used to track ops
    and their grouping.  A DFS is performed on ops to find all source ops which
    are placed into a queue for processing.  The OpRegularizerManager then loops
    over ops in the queue, using the associated OpHandler to determine the
    grouping of the op.  Once all ops have been grouped, regularizers for the
    groups can be created.

    If a group has multiple sources of regularization, the
    create_grouping_regularizer function is used to create an OpRegularizer that
    combines the multiple sources.

    If force_group is specified, ops that would not normally be grouped are
    force-grouped.  Ops matching the regex will be grouped together, along with
    all ops that were grouped with the matching ops.  Basically, the groups
    would be merged.  Each regex specifies a separate force-grouping.

    If regularizer_blacklist is specified, then ops matching any of the regex
    (and ops in the same group) do not get regularized.  The
    OpRegularizerManager will instead create a None regularizer for the group.

    Args:
      ops: List of tf.Operation.  An OpRegularizer will be created for all
        operations in ops, as well as all operations which are dependencies.
        Typically, ops would contain a single tf.Operation, which is the output
        of the network.
      op_handler_dict: Dictionary mapping tf.Operation type to OpHandler.
      create_grouping_regularizer: Function that creates an OpRegularizer given
        a list of OpRegularizer.
      force_group: List of regex for ops that should be force-grouped.  Each
        regex corresponds to a separate group.  Use '|' operator to specify
        multiple patterns in a single regex.
      regularizer_blacklist: List of regex for ops that should not be
        regularized.
      input_boundary: A list of ops that represent the input boundary of the
        subgraph being regularized (input boundary is not regularized).
      iteration_limit: Integer iteration limit for OpRegularizerManager to
        finish analyzing the network.  If the limit is reached, it is assumed
        that OpRegularizerManager got stuck in a loop.

    Raises:
      RuntimeError: If OpRegularizerManager cannot analyze the entire network
        within ITERATION_LIMIT.
      TypeError: If force_group argument is not a list.
      TypeError: If regularizer_blacklist argument is not a list.
    """
    # Dictionary mapping op to list of OpSlice.  The op is the concatenation of
    # its OpSlice list.
    self._op_slice_dict = {}

    # Dictionary mapping OpSlice to OpGroup.
    self._op_group_dict = {}

    # Dictionary mapping op type to OpHandler class.
    self._op_handler_dict = op_handler_dict or {}

    # Dictionary mapping OpSlice to OpRegularizer.
    self._op_regularizer_dict = {}

    # Queue of ops to process.
    self._op_deque = collections.deque()

    # Set of all ops to regularize.
    self._all_ops = set()

    # Start DFS from outputs to find all source ops.
    tf.logging.info('OpRegularizerManager starting analysis from: %s.', ops)
    self._dfs_for_source_ops(ops, input_boundary)
    tf.logging.info('OpRegularizerManager found %d ops and %d sources.',
                    len(self._all_ops), len(self._op_deque))

    # Process grouping for all ops.
    iteration_count = 0
    while self._op_deque and iteration_count < iteration_limit:
      op = self._op_deque.pop()
      self._op_handler_dict[op.type].assign_grouping(op, self)
      iteration_count += 1
    if iteration_count >= iteration_limit:
      # OpRegularizerManager got stuck in a loop.  Report the ops still in the
      # processing queue.
      raise RuntimeError('OpRegularizerManager could not handle ops: %s' %
                         ['%s (%s)' % (o.name, o.type) for o in self._op_deque])

    # Force-group ops.
    force_group = force_group or []
    if not isinstance(force_group, list):
      raise TypeError('force_group must be a list of regex.')
    self._force_group_ops(force_group)

    # Create blacklist regex.
    blacklist_regex = ''
    if regularizer_blacklist:
      if not isinstance(regularizer_blacklist, list):
        raise TypeError('regularizer_blacklist must be a list of regex.')
      blacklist_regex = '|'.join(regularizer_blacklist)

    # Instantiate regularizers for all groups that have sources.
    groups = set(self._op_group_dict.values())
    blacklist_used = False
    for group in groups:
      # Collect regularizer for every source OpSlice in the OpGroup.
      source_op_slices = []
      regularizers = []

      # If group is blacklisted, then no regularizers are created and all
      # OpSlice will be assigned a None regularizer.
      if op_handler_util.group_match(blacklist_regex, group.op_slices):
        tf.logging.info('OpGroup not regularized due to blacklist: %s.',
                        group.op_slices)
        blacklist_used = True
      else:
        for source_op_slice in group.source_op_slices:
          handler = self._op_handler_dict[source_op_slice.op.type]
          source_op_slices.append(source_op_slice)
          regularizers.append(handler.create_regularizer(source_op_slice))

      # Create a group regularizer and assign to all OpSlice in the OpGroup.  If
      # there are no regularizers, assign None.
      if regularizers:
        if len(regularizers) > 1:
          group_regularizer = create_grouping_regularizer(regularizers)
        else:
          group_regularizer = regularizers[0]
      else:
        group_regularizer = None
      for op_slice in group.op_slices:
        self._op_regularizer_dict[op_slice] = group_regularizer
      tf.logging.info('Source OpSlice %s for OpGroup: %s.', source_op_slices,
                      group.op_slices)

    if blacklist_regex and not blacklist_used:
      raise ValueError('Blacklist regex never used: \'%s\'.' % blacklist_regex)

    tf.logging.info('OpRegularizerManager regularizing %d groups.',
                    len(set(self._op_group_dict.values())))

    # Set scope of all ops to be ops that were analyzed.
    self._all_ops = set(self._op_slice_dict.keys())

  @property
  def ops(self):
    """Returns all ops discovered by OpRegularizerManager."""
    return self._all_ops

  def get_regularizer(self, op):
    """Returns an OpRegularizer for the specified op.

    If no OpRegularizer exists for any slices in the op, returns None.
    Otherwise, create a ConstantOpRegularizer for any slices that are missing a
    regularizer.

    Args:
      op: A tf.Operation.

    Returns:
      An OpRegularizer for op, or None if no OpRegularizer exists.
    """
    op_slices = self.get_op_slices(op)
    regularizers = [
        self._op_regularizer_dict.get(op_slice) for op_slice in op_slices
    ]
    # If all OpSlice have None regularizer, return None.
    if not any(regularizers):
      return None

    regularizers = []
    for op_slice in op_slices:
      regularizer = self._op_regularizer_dict.get(op_slice)
      if regularizer is None:
        regularizer = constant_op_regularizer.ConstantOpRegularizer(
            op_slice.slice.size)
        self._op_regularizer_dict[op_slice] = regularizer
      regularizers.append(regularizer)

    # If op only has 1 OpSlice, return the regularizer for that OpSlice.
    # Otherwise, return the concatenation of regularizers for the constituent
    # OpSlice.
    if len(regularizers) == 1:
      return regularizers[0]
    else:
      return concat_and_slice_regularizers.ConcatRegularizer(regularizers)

  def create_op_group_for_op_slice(self, op_slice, is_source=True):
    """Creates an OpGroup for an OpSlice.

    Args:
      op_slice: OpSlice to create an OpGroup for.
      is_source: Boolean indicating if the OpSlice is a source.

    Returns:
      OpGroup for the OpSlice.
    """
    # If OpSlice is not a source, then omit it from list of source OpSlice.
    omit_source_op_slices = [] if is_source else [op_slice]

    # Create OpGroup for the OpSlice.
    op_group = OpGroup(op_slice, omit_source_op_slices=omit_source_op_slices)

    # Update mapping of OpSlice to new OpGroup.
    self._op_group_dict[op_slice] = op_group

    return self.get_op_group(op_slice)

  def group_op_slices(self, op_slices, omit_source_op_slices=None):
    """Group op slices.

    Each OpSlice in op_slices gets mapped to the same group.  Additionally, the
    new group is also mapped to the list of OpSlice.  Note that this is
    transitive, meaning that if group_op_slices([A, B]) is called when B is
    grouped with C, then all 3 OpSlice [A, B, C] will be grouped together.

    Args:
      op_slices: List of OpSlice to group.
      omit_source_op_slices: List of OpSlice to not track as sources in the new
        OpGroup.
    """
    # Find groups that op slices are already a part of.
    existing_op_groups = []
    for op_slice in op_slices:
      op_group = self.get_op_group(op_slice)
      if op_group and op_group not in existing_op_groups:
        existing_op_groups.append(op_group)

    # Find OpSlice that will change group.
    # pylint: disable=g-complex-comprehension
    op_slices_to_update = [
        os for og in existing_op_groups for os in og.op_slices
    ]
    for op_slice in op_slices:
      if op_slice not in op_slices_to_update:
        # This OpSlice does not have an OpGroup, so create a temporary one.
        temp_op_group = self.create_op_group_for_op_slice(
            op_slice, is_source=self.is_source_op(op_slice.op))
        existing_op_groups.append(temp_op_group)
        op_slices_to_update.append(op_slice)

    # Create new OpGroup.
    new_op_group = OpGroup(
        op_groups=existing_op_groups,
        omit_source_op_slices=omit_source_op_slices)

    # Update mapping.
    for op_slice in op_slices_to_update:
      self._op_group_dict[op_slice] = new_op_group

  def slice_op(self, op, sizes):
    """Slice an op into specified sizes.

    Creates OpSlice objects to represent slices of op.  The op is mapped to its
    constituent OpSlice and reformed by concatenating the OpSlice.  For example,
    if op has 10 channels and sizes is [3, 7], then this method returns
    [OpSlice(op, (0, 3)), OpSlice(op, (3, 7))].

    Note that sizes must be able to be aligned with the original op slice sizes.
    An original slice can be partitioned into smaller slices, but the original
    slice boundaries cannot be changed.  For example, if the original sizes are
    [3, 7], the op cannot be sliced into sizes [2, 8].  However, slicing into
    sizes [1, 2, 3, 4] is okay because the original slices are being sliced
    (3 -> [1, 2] and 7 -> [3, 4]).

    Also note that ops that are grouped with op will also be sliced accordingly,
    with respective slices grouped.  For example, if OpA is grouped with OpB and
    OpC, and OpA is sliced into OpA1 and OpA2, then the result will be groups
    (OpA1, OpB1, OpC1) and (OpA2, OpB2, OpC2).

    Args:
      op: A tf.Operation to slice for the purpose of grouping.
      sizes: List of Integer sizes to slice op into.  Sizes must sum up to the
        number of output channels for op.

    Raises:
      ValueError: If sizes cannot be aligned with original op slice sizes.
    """
    old_op_slices = self.get_op_slices(op)
    old_op_slice_sizes = op_handler_util.get_op_slice_sizes([old_op_slices])[0]

    # If sizes already match, then nothing happens.
    if old_op_slice_sizes == sizes:
      return

    # If sizes cannot be aligned with original sizes, raise exception.
    try:
      aligned_op_slice_sizes = op_handler_util.get_aligned_sizes(
          [old_op_slice_sizes, sizes])
    except ValueError as e:
      raise ValueError('Error with op: %s: %s' % (op.name, e.args[0]))

    if sizes != aligned_op_slice_sizes:
      raise ValueError('Cannot slice op %s from sizes %s to %s' %
                       (op.name, old_op_slice_sizes, sizes))

    # Iterate through slices to find old slices that need to be resliced.
    old_slice_index = 0
    new_slice_index = 0
    new_slice_count = 1
    while (new_slice_index + new_slice_count <= len(aligned_op_slice_sizes) and
           old_slice_index < len(old_op_slice_sizes)):
      old_size = old_op_slice_sizes[old_slice_index]
      new_size = op_handler_util.get_total_slice_size(sizes, new_slice_index,
                                                      new_slice_count)
      if old_size == new_size:
        if new_slice_count > 1:
          # If sizes match then this old slice is sliced into new_slice_count
          # smaller slices.  Find the group of the old slice because all OpSlice
          # in the group will need to be sliced similarly.
          op_group = self.get_op_group(old_op_slices[old_slice_index])
          if op_group:
            group_op_slices = op_group.op_slices
          else:
            # If OpSlice has no group, just use the OpSlice itself.
            group_op_slices = [old_op_slices[old_slice_index]]
          new_op_slice_group = [list() for _ in range(new_slice_count)]
          for group_op_slice in group_op_slices:
            self._slice_op_slice(group_op_slice, sizes, new_slice_index,
                                 new_slice_count, new_op_slice_group)

          if op_group:
            # Group all new OpSlice along each index.
            for i in range(new_slice_count):
              self.group_op_slices(new_op_slice_group[i])

        # Update indices for the next slice.
        old_slice_index += 1
        new_slice_index += new_slice_count
        new_slice_count = 1
      else:
        # If sizes do not match, then more new slices are needed to match the
        # old slice.
        new_slice_count += 1

  def process_ops(self, ops):
    """Add ops to processing queue.

    Args:
      ops: List of tf.Operation to put into the processing queue.
    """
    new_ops = [
        op for op in ops if op not in self._op_deque and op in self._all_ops
    ]
    self._op_deque.extend(new_ops)

  def process_ops_last(self, ops):
    """Add ops to the end of the processing queue.

    Used to avoid infinite looping if an OpHandler decides to defer processing
    of itself.

    Args:
      ops: List of tf.Operation to put at the end of the processing queue.
    """
    new_ops = [op for op in ops if op not in self._op_deque]
    self._op_deque.extendleft(new_ops)

  def is_source_op(self, op):
    """Returns True if op is a source op.

    Args:
      op: tf.Operation to check whether it is a source op.

    Returns:
      Boolean indicating if op is a source op.
    """
    op_handler = self._op_handler_dict[op.type]
    return op_handler.is_source_op

  def is_passthrough(self, op):
    """Returns True if op is passthrough.

    Args:
      op: tf.Operation to check whether it is passthrough.

    Returns:
      Boolean indicating if op is passthrough.
    """
    op_handler = self._op_handler_dict[op.type]
    return op_handler.is_passthrough

  def get_op_slices(self, op):
    """Returns OpSlice objects that are mapped to op.

    If no mapping exists, a new OpSlice object will be created and mapped to op.

    Args:
      op: A tf.Operation to get OpSlice for.

    Returns:
      List of OpSlice that constitute op.
    """
    if op not in self._op_slice_dict:
      # No OpSlice exists for op so create a new one.
      size = op_handler_util.get_op_size(op)
      if size > 0:
        new_op_slice = OpSlice(op, Slice(0, size))
        self._op_slice_dict[op] = [new_op_slice]
      else:
        self._op_slice_dict[op] = []
    return self._op_slice_dict[op]

  def get_op_group(self, op_slice):
    """Returns the OpGroup that contains op_slice.

    Returns None if no mapping exists.

    Args:
      op_slice: An OpSlice to find OpGroup for.

    Returns:
      OpGroup that contains op_slice, or None if no mapping exists.
    """
    return self._op_group_dict.get(op_slice)

  def _slice_op_slice(self, op_slice, sizes, size_index, size_count,
                      new_op_slice_group):
    """Slices an OpSlice according to new sizes.

    During reslicing, any OpSlice of an op could be resliced.  Given the new
    sizes, this method finds the index where the old OpSlice matches, and
    reslices the OpSlice according to the new sizes.  The new OpSlice are added
    to new_op_slice_group by index, so that matching OpSlice can be grouped
    together later.

    Args:
      op_slice: OpSlice that should be sliced.
      sizes: List of integers specifying the new slice sizes.
      size_index: Integer specifying which index in sizes corresponds to
        op_slice.
      size_count: Integer specifying how many slices op_slice will be sliced
        into.
      new_op_slice_group: List of list of new OpSlice that should be grouped
        together.
    """
    op = op_slice.op
    op_slices = self.get_op_slices(op)

    # Get slice sizes for op.
    op_slice_sizes = op_handler_util.get_op_slice_sizes([op_slices])[0]

    # Find the slice index that needs to be resliced.
    op_slice_index = op_slices.index(op_slice)

    # Clear old OpSlice to OpGroup mapping.
    if op_slice in self._op_group_dict:
      del self._op_group_dict[op_slice]

    # Calculate the new op slice sizes for the op.
    op_slice_sizes.pop(op_slice_index)
    # Keep track of which OpSlice were resliced.
    is_resliced = [False] * len(op_slice_sizes)
    for i in range(size_count):
      op_slice_sizes.insert(op_slice_index + i, sizes[size_index + i])
      is_resliced.insert(op_slice_index + i, True)

    # Find source slices and slice the op.
    is_source = self._get_source_slices(op_slice_sizes, op_slices)
    slices = self._slice_op_with_sizes(op, op_slice_sizes, is_source,
                                       is_resliced)

    # Accumulate new OpSlice at the corresonding index.
    for i in range(size_count):
      new_op_slice_group[i].append(slices[op_slice_index + i])

  def _slice_op_with_sizes(self, op, sizes, is_source, is_resliced):
    """Slices the op according to sizes.

    Args:
      op: tf.Operation to slice.
      sizes: List of integers of slice sizes.
      is_source: List of booleans indicating which new slices are sources.
      is_resliced: List of booleans indicating which slices are new.

    Returns:
      List of OpSlice for the newly sliced op.
    """
    old_slices = self.get_op_slices(op)
    slices = []
    for i, size in enumerate(sizes):
      if is_resliced[i]:
        # Sum up previous slice sizes to find start index of next slice.
        index = sum(sizes[:i])
        # Create new OpSlice for new slices.
        new_slice = OpSlice(op, Slice(index, size))
        # Create new OpGroup for OpSlice that should be sources.
        if is_source[i]:
          self.create_op_group_for_op_slice(new_slice)
      else:
        # If OpSlice is not new, reuse existing OpSlice.  Calculate the index of
        # the old OpSlice by subtracting the count of new slices.
        offset = max(is_resliced[:i].count(True) - 1, 0)
        new_slice = old_slices[i - offset]
      slices.append(new_slice)

    # Update OpSlice for the op.
    self._op_slice_dict[op] = slices

    return slices

  def _get_source_slices(self, sizes, op_slices):
    """Returns list of booleans indicating which slices are sources.

    If an OpSlice is a source, then its slices are also sources.  For example,
    if an op consists of slices size [3, 7] where only the first slice is a
    source, but is resliced into sizes [1, 2, 3, 4], then only the first 2
    slices are sources.  Then this method would return
    [True, True, False, False].

    Args:
      sizes: List of integers indicating new slice sizes.
      op_slices: List of OpSlice before slicing.

    Returns:
      List of booleans indicating which slices are sources.
    """
    size_index = 0
    slice_index = 0
    is_source = []
    while size_index < len(sizes):
      # Get the OpGroup for the OpSlice to see if it is a source.
      op_slice = op_slices[slice_index]
      op_group = self.get_op_group(op_slice)
      if op_group and op_slice in op_group.source_op_slices:
        is_source.append(True)
      else:
        is_source.append(False)

      # Check end indices of op_slices and sizes.  If they match, then current
      # slice is done and slice index should be incremented.
      end_index = sum(sizes[:size_index + 1])
      slice_end_index = op_slice.slice.start_index + op_slice.slice.size
      if end_index == slice_end_index:
        slice_index += 1
      size_index += 1
    return is_source

  def _dfs_for_source_ops(self, ops, input_boundary=None):
    """Performs DFS from ops and finds source ops to process.

    Args:
      ops: A list of tf.Operation's.
      input_boundary: A list of ops where traversal should terminate.
    """
    if input_boundary:
      input_boundary = set(input_boundary)
    else:
      input_boundary = set()
    to_visit = list(ops)
    visited = set()
    while to_visit:
      # Get next op and mark as visited.
      op = to_visit.pop()
      visited.add(op)
      if op in input_boundary:
        continue
      self._all_ops.add(op)

      # Check if op is a source by querying OpHandler.
      if self._op_handler_dict[op.type].is_source_op:
        self.process_ops([op])

      # Add op inputs to to_visit.
      for tensor in op.inputs:
        next_op = tensor.op
        if next_op not in visited:
          to_visit.append(next_op)

  def _force_group_ops(self, force_group):
    """Force-groups ops that match a regex.

    Args:
      force_group: List of regex.  For each regex, all matching ops will have
        their groups merged.
    """
    for regex in force_group:
      force_group_ops = []
      for op, op_slices in self._op_slice_dict.items():
        if op_handler_util.group_match(regex, op_slices):
          force_group_ops.append(op)

      # If no ops match, continue to the next force-group.
      if not force_group_ops:
        raise ValueError('Regex \'%s\' did not match any ops.')

      # Assert all ops to force-group have only 1 OpSlice.
      if ([len(self._op_slice_dict[op]) for op in force_group_ops] !=
          [1] * len(force_group_ops)):
        multiple_slice_ops = []
        for op in force_group_ops:
          if len(self._op_slice_dict[op]) != 1:
            multiple_slice_ops.append(op.name)
        raise ValueError('Cannot force-group ops with more than 1 OpSlice: %s' %
                         multiple_slice_ops)

      # Assert all ops to force-group have the same size.
      target_op_size = self._op_slice_dict[force_group_ops[0]][0].slice.size
      if ([self._op_slice_dict[op][0].slice.size for op in force_group_ops] !=
          [target_op_size] * len(force_group_ops)):
        op_names = [op.name for op in force_group_ops]
        raise ValueError(
            'Cannot force-group ops with different sizes: %s' % op_names)

      # Group the ops.
      self.group_op_slices(
          [self._op_slice_dict[op][0] for op in force_group_ops])


class OpGroup(object):
  """Helper class to keep track of OpSlice grouping."""

  _static_index = 0

  def __init__(self, op_slice=None, op_groups=None, omit_source_op_slices=None):
    """Create OpGroup with self-incrementing index.

    The OpGroup keeps a list of OpSlice that belong to the group.  The OpGroup
    also keeps a separate list of source OpSlice.  If op_slice is specified, it
    is assumed to be a source.  All OpGroup in op_groups will be merged together
    to form a new OpGroup.  OpSlice listed in omit_source_op_slices will not
    be tracked as sources in the new OpGroup.

    Args:
      op_slice: OpSlice to include in the group and track as a source.
      op_groups: List of OpGroup to merge together into a new OpGroup.
      omit_source_op_slices: List of OpSlice to not track as sources in the new
        OpGroup.
    """
    omit_source_op_slices = omit_source_op_slices or []

    # Add op_slice to the OpGroup.
    self._op_slices = []
    if op_slice:
      self._op_slices.append(op_slice)
    self._source_op_slices = []
    if op_slice is not None and op_slice not in omit_source_op_slices:
      self._source_op_slices.append(op_slice)

    # Merge op_groups into a combined OpGroup.
    if op_groups:
      for op_group in op_groups:
        # Collect OpSlice from each OpGroup.
        for op_slice in op_group.op_slices:
          if op_slice not in self._op_slices:
            self._op_slices.append(op_slice)
        # Collect source OpSlice from each OpGroup.
        for source_op_slice in op_group.source_op_slices:
          if (source_op_slice not in omit_source_op_slices and
              source_op_slice not in self._source_op_slices):
            self._source_op_slices.append(source_op_slice)

    # Increment OpGroup index.
    self._index = OpGroup._static_index
    OpGroup._static_index += 1

  @property
  def op_slices(self):
    """Return a list of OpSlice belonging to the OpGroup."""
    return self._op_slices

  @property
  def source_op_slices(self):
    """Return a list of OpSlice that are regularizer sources."""
    return self._source_op_slices
