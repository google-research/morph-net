"""Utility functions for handling TPU graphs."""

from __future__ import absolute_import
from __future__ import division
# [internal] enable type annotations
from __future__ import print_function

import tensorflow as tf
from typing import Dict, Tuple, Text

TensorSignature = Tuple[tf.Graph, Text]
VariableSignature = Tuple[Text, tf.TensorShape, tf.DType]


class VariableStore(object):
  """Implements storing tensors in variables and lookup by tensor.

  This class is intended to be used when computing summaries on CPU using
  tensors from TPU. An instance of this class is defined as a global object.
  There should be no need to create more instances of this class, but it is ok
  to do so.

  To make tensor `x` available on CPU, at least one occurrence of x on TPU must
  be replaced with `cross_device(x, is_tpu=True)`; the result must be among the
  dependencies of session.run fetches. Then, to look up `x` on CPU, call
  `cross_device(x, is_tpu=False)`.

  Example of original code:
    layer1 = build_layer(inputs)
    logits = build_logits(layer1)
    sess.run(tf.global_variable_initializer())
    logits = sess.run(logits)

  Example of rewritten code that uses VariableStore:
    layer1 = tpu_util.cross_device(build_layer(inputs), is_tpu=True)
    logits = (build_logits(layer1)
    sess.run(tf.global_variable_initializer())
    logits = sess.run(logits)
    variable = tpu_util.cross_device(layer, is_tpu=False)
    # The value of layer1 tensor last time it was computed.
    old_layer1_value = sess.run(variable)
  """

  def __init__(self) -> None:
    # Keys: (graph, tensor_name)
    # Value: (var_name, var_shape, var_dtype)
    self._registry = {
    }  # type: Dict[TensorSignature, VariableSignature]

  def _get_tensor_signature(self, tensor: tf.Tensor) -> TensorSignature:
    return tensor.graph, tensor.name

  def copy_to_variable(self, tensor: tf.Tensor) -> tf.Tensor:
    """Create if necessary, and update the variable with the tensor value.

    The returned value must be used in the computation of the model
    outputs or added to it via tf.control_dependencies (otherwise
    the variable update op will be skipped in the session.run()). The
    easiest way to achieve this is simply to use the returned value
    instead of the original tensor while building the model.

    Args:
      tensor: The tensor to replace.

    Returns:
      The tf.Identity of `tensor`, which control-depends on the variable assign.
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
    else:
      name, shape, dtype = registry_entry
      with tf.variable_scope(tf.VariableScope(''), reuse=True):
        # TF requires specifying matching shape and dtype even when reuse=True.
        v = tf.get_variable(name, dtype=dtype, shape=shape, use_resource=True)

    with tf.control_dependencies([tf.assign(v, tensor)]):
      return tf.identity(tensor)

  def read_from_variable(self, tensor: tf.Tensor) -> tf.Tensor:
    """Lookup the desired tensors in variable store.

    This function should only be used on CPU, where a separate session ensures
    order relative to assign ops. Using it on TPU would require adding control
    dependencies, and it's simpler just not to use on TPU at all.

    Args:
      tensor: The tensor to look up.

    Returns:
      The tensor that reads the matching variable.
    """
    registry_entry = self._registry.get(self._get_tensor_signature(tensor))
    if registry_entry is None:
      raise ValueError('Tensor %s was never replaced with variable' % tensor)
    name, shape, dtype = registry_entry
    with tf.variable_scope(tf.VariableScope(''), reuse=True):
      return tf.get_variable(name, dtype=dtype, shape=shape, use_resource=True)


# Global instance of VariableStore.
variable_store = VariableStore()


@tf.contrib.framework.add_arg_scope
def cross_device(tensor: tf.Tensor, is_tpu: bool = True) -> tf.Tensor:
  if is_tpu:
    return variable_store.copy_to_variable(tensor)
  else:
    return variable_store.read_from_variable(tensor)
