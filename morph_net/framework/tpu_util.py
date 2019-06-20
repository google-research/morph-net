"""Utility functions for handling TPU graphs."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import contextlib2
import tensorflow as tf

_run_on_cpu = False


@contextlib2.contextmanager
def run_on_cpu():
  """Provide a context for the code that needs to run on CPU.

  Not thread-safe.

  Yields:
    None.
  """
  global _run_on_cpu
  original_run_on_cpu = _run_on_cpu
  _run_on_cpu = True
  try:
    yield
  finally:
    _run_on_cpu = original_run_on_cpu


def is_on_cpu():
  return _run_on_cpu


def get_variable_name(read_variable_op):
  assert read_variable_op.type == 'ReadVariableOp'
  op = read_variable_op
  while op.type != 'VarHandleOp':
    assert len(op.inputs) == 1
    op = op.inputs[0].op
  return op.name


def maybe_convert_to_variable(tensor):
  """Read value of a tensor from a variable when possible.

  This function is intended to make tensors from inside the TPU while loop
  available on the CPU by reading it from the variable to which the tensor was
  written earlier. Note that the read may not reflect any writes that happened
  in the same session.run(), unless control dependencies are added.

  Args:
    tensor: A tf.Tensor.

  Returns:
    A tf.Tensor. If input tensor is an output of reading a ResourceVariable, we
    return an equivalent tensor produced in the current context. Otherwise, we
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
    tf.logging.info('Converting tensor %s --> variable %s',
                    tensor, variable_name)
    try:
      return tf.get_variable(variable_name)
    except ValueError:
      tf.logging.info(
          'Variable %s was not created with tf.get_variable(). '
          'Attempting to find it in GLOBAL_VARIABLES collection.',
          variable_name)
    global_vars = tensor.graph.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
    matched_vars = [v for v in global_vars if v.name == variable_name + ':0']
    if not matched_vars:
      raise ValueError('Variable %s is in GraphDef but not in the live graph.')
    assert len(matched_vars) == 1
    return matched_vars[0]


var_store = {}


def write_to_variable(tensor):
  """Saves a tensor for later retrieval on CPU."""
  # Only relevant for debugging.
  debug_name = 'tpu_util__' + tensor.name.split(':')[0].split('/')[-1]
  with tf.variable_scope('', reuse=False):
    variable = tf.get_variable(
        name=debug_name,
        shape=tensor.shape,
        dtype=tensor.dtype,
        trainable=False,
        use_resource=True)
  var_store[tensor] = variable
  with tf.control_dependencies([variable.assign(tensor)]):
    return tf.identity(tensor)


def read_from_variable(tensor):
  """Retrieves (a possibly stale copy of) the previously stored tensor."""
  if is_on_cpu():
    # Stale read, but on CPU that's all we can do without adding to loop vars.
    return var_store[tensor]
  else:
    # Current read, but only works on TPU.
    return tensor
