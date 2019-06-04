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
    A tf.Tensor. If input tensor is an output of reading a ResourceVariable, we
    return an equivalent tensor produced outside the while loop. Otherwise, we
    return the original input tensor.
  """
  op = tensor.op
  if op.type != 'ReadVariableOp':
    # No need to convert.
    return tensor
  with tf.variable_scope(
      # Reset the scope because variable_name contains all the scopes we need.
      name_or_scope=tf.VariableScope(''),
      # We are looking for a reference to an existing variable, so we want to
      # raise an exception if variable is not found.
      reuse=True,
  ):
    variable_name = get_variable_name(op)
    tf.logging.info('Converting tensor %s --> tf.get_variable(%s)',
                    tensor, variable_name)
    return tf.get_variable(variable_name)


class VariableStore(object):
  """Implements storing tensors in variables and lookup by tensor.

  An instance of this class is defined as a global object. There should
  no need to create more instances of this class, but it is ok to do so.

  Example of original code:
    layer1 = build_layer(inputs)
    logits = build_logits(layer1)
    sess.run(tf.global_variable_initializer())
    logits = sess.run(logits)

  Example of rewritten code that uses VariableStore:
    var_store = tpu_util.VariableStore()
    layer1 = replace_with_variable(build_layer(inputs))
    logits = (build_logits(layer1)
    sess.run(tf.global_variable_initializer())
    logits = sess.run(logits)
    variable = var_store.lookup_tensor(layer)
    # The value of layer1 tensor last time it was computed.
    old_layer1_value = sess.run(variable)
  """

  def __init__(self):
    self._registry = {}

  def _get_tensor_signature(self, tensor):
    return tensor.graph, tensor.name

  def replace_with_variable(self, tensor):
    """Create if necessary, and update the variable with the tensor value.

    The returned value must be used in the computation of the model
    outputs or added to it via tf.control_dependencies (otherwise
    the variable update op will be skipped in the session.run()). The
    easiest way to achieve this is simply to use the returned value
    instead of the original tensor while building the model.

    Args:
      tensor: The tensor to replace.

    Returns:
      The tensor that is read from the newly updated variable.
    """

    if not isinstance(tensor, tf.Tensor):
      raise ValueError('Argument tensor must be a tf.Tensor: %s' % tensor)
    if not tensor.shape.is_fully_defined:
      raise ValueError('Tensor shape must be fully defined: %s' % tensor)
    tensor_signature = self._get_tensor_signature(tensor)
    registry_entry = self._registry.get(tensor_signature)
    if registry_entry is None:
      with tf.variable_scope(tf.VariableScope(''), reuse=False):
        # The variable scope and name matters only for debugging;
        # we use self._registry to map tensor names to variables.
        name = tensor.name.replace(':', '__')
        self._registry[tensor_signature] = name, tensor.shape, tensor.dtype
        v = tf.get_variable(
            name, dtype=tensor.dtype, shape=tensor.shape, use_resource=True)
        tf.logging.info('Added variable %s for %s, %s', v, tensor, tensor.graph)
        assert name + ':0' == v.name
        return tf.assign(v, tensor)
    else:
      name, shape, dtype = registry_entry
      with tf.variable_scope(tf.VariableScope(''), reuse=True):
        # TF requires specifying matching shape and dtype even when reuse=True.
        v = tf.get_variable(name, dtype=dtype, shape=shape, use_resource=True)
      return tf.assign(v, tensor)

  def lookup_tensor(self, tensor):
    registry_entry = self._registry.get(self._get_tensor_signature(tensor))
    if registry_entry is None:
      raise ValueError('Tensor %s was never replaced with variable' % tensor)
    name, shape, dtype = registry_entry
    with tf.variable_scope(tf.VariableScope(''), reuse=True):
      return tf.get_variable(name, dtype=dtype, shape=shape, use_resource=True)


# Global instance of VariableStore.
variable_store = VariableStore()
