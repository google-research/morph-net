"""Utility functions for handling TPU graphs."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf


def get_variable_name(read_variable_op):
  assert read_variable_op.type == 'ReadVariableOp'
  op = read_variable_op
  while op.type != 'VarHandleOp':
    assert len(op.inputs) == 1
    op = op.inputs[0].op
  return op.name


def maybe_convert_to_variable(tensor):
  """Convert TPU variable to be usable outside a while loop.

  Args:
    tensor: A tf.Tensor.

  Returns:
    If tensor is the output of reading a ResourceVariable, replace it with an
    equivalent tensor produced outside the while loop. Otherwise, return the
    original tensor.
  """
  op = tensor.op
  if op.type != 'ReadVariableOp':
    # No need to convert.
    return tensor
  with tf.variable_scope(
      # We don't want to add any new scope, just reuses the outside scope.
      # The only reason we are using variable_scope is because there's no other
      # way to request reuse.
      name_or_scope=tf.get_variable_scope(),
      # We are looking for a reference to an existing variable, so we want to
      # raise an exception if variable is not found.
      reuse=True,
  ):
    # The name already contains (as slashes) all the scope we need.
    # However, we cannot reset scope to root because TF has no such API;
    # see https://yaqs.googleplex.com/eng/q/5549333873426432, and
    # https://github.com/tensorflow/tensorflow/issues/7731#issuecomment-329694973
    # Therefore, we have to hack the variable name.

    variable_name = get_variable_name(op)
    current_scope = tf.get_variable_scope().name
    tf.logging.info('Converting. variable_name %s current_scope %s tensor %s',
                    variable_name, current_scope, tensor)
    if current_scope:
      current_scope += '/'
      # Strip current_scope from the variable name.
      assert variable_name.startswith(current_scope)
      relative_name = variable_name[len(current_scope):]
    else:
      # current_scope == '' if subplan variable scope is not set, or if it's set
      # but w are creating the regularizer for summaries rather than losses.
      relative_name = variable_name
    return tf.get_variable(relative_name)
