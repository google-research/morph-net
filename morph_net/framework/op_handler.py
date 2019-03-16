"""Interface for OpHandler classes.

An OpHandler contains the logic for assigning a regularizer to an op type.  The
OpRegularizerManager uses a dictionary of {op type: OpHandler} to assign
regularizers to ops in the network.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import abc


class OpHandler(object):
  """An interface for OpHandler classes.

  An OpHandler contains the logic for assigning a regularizer to an op type.
  The OpHandler denotes if the op type is considered a source of regularization,
  or how the regularization is derived from neighboring ops.  Ops are first
  assigned grouping if they share a regularizer.  Then, regularizers are created
  for the groups.
  """
  __metaclass__ = abc.ABCMeta

  @abc.abstractproperty
  def is_source_op(self):
    """Returns True if this op type is a regularizer source."""

  @abc.abstractproperty
  def is_passthrough(self):
    """Returns True if this op type is considered passthrough.

    Neighboring ops may query this property before deciding to group with this
    op.  For example, consider OpX followed by convolution followed by batch
    norm (source op).  The batch norm creates its own group and adds the
    convolution to the processing queue.  Then the convolution groups itself
    with the output op (batch norm) and adds OpX to the processing queue.  OpX
    would examine its output (the convolution) but see that
    is_passthrough=False.  Thus, OpX would NOT group itself with the convolution
    and would group with the input instead.
    """

  @abc.abstractmethod
  def assign_grouping(self, op, op_reg_manager):
    """Assign grouping to the given op and updates the manager.

    Each OpHandler includes custom logic for how to assign op grouping.  This
    logic could consider: if this op type is a source, if neighboring ops have
    groupings and are passthrough, if all inputs or outputs have groupings, if
    channels need to be concat/sliced into groups, etc.

    For example, generic passthrough ops can group with input or output
    neighbors, but Conv2D ops would not group with input ops.

    Once the OpHandler determines how to group the op, OpRegularizerManager
    should be updated with the resulting grouping.  Finally, OpHandler should
    also decide which neighboring ops should be put into the queue for
    OpRegularizerManager to process.

    Args:
      op: tf.Operation to assign grouping to.
      op_reg_manager: OpRegularizerManager to keep track of the grouping.
    """
    pass

  @abc.abstractmethod
  def create_regularizer(self, op_slice):
    """Create a regularizer for a source OpSlice.

    Args:
      op_slice: op_regularizer_manager.OpSlice that is a source OpSlice.

    Returns:
      OpRegularizer for the source OpSlice.
    """
    pass
