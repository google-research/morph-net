"""Support for overriding op regularization penalty."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from morph_net.framework import op_handler


class OpHandlerDecorator(op_handler.OpHandler):
  """A decorator for OpHandler implementations.

  This decorator overrides the create_regularizer method, allowing customization
  of the regularization penalty used.  Other members of the original OpHandler
  are unchanged.
  """

  def __init__(self, handler, regularizer_decorator=None,
               decorator_parameters=None):
    """Creates an instance.

    Args:
      handler: OpHandler to be decorated.
      regularizer_decorator: OpRegularizer decorator to apply to OpRegularizer
        returned by create_regularizer method of handler.  If None, the
        OpRegularizer is unchanged.
      decorator_parameters: Dictionary of regularizer decorator parameters.
        None or {} will pass no parameters.
    """
    self._op_handler = handler
    self._regularization_decorator = regularizer_decorator
    self._decorator_parameters = decorator_parameters or {}

  @property
  def is_source_op(self):
    return self._op_handler.is_source_op

  @property
  def is_passthrough(self):
    return self._op_handler.is_passthrough

  def assign_grouping(self, op, op_reg_manager):
    self._op_handler.assign_grouping(op, op_reg_manager)

  def create_regularizer(self, op_slice):
    """Creates a decorated OpRegularizer for the given OpSlice.

    Args:
      op_slice: op_regularizer_manager.OpSlice to create a regularizer for.

    Returns:
      A decorated OpRegularizer for the given OpSlice.
    """
    regularizer = self._op_handler.create_regularizer(op_slice)
    if regularizer and self._regularization_decorator:
      regularizer = self._regularization_decorator(
          regularizer, **(self._decorator_parameters))

    return regularizer

