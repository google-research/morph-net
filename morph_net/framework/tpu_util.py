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
  """Convert TPU tensor to the ResourceVariable if possible."""
  op = tensor.op
  if op.type != 'ReadVariableOp':
    # Cannot convert.
    return tensor
  with tf.variable_scope(tf.get_variable_scope(), reuse=tf.AUTO_REUSE):
    shape = tensor.shape
    return tf.get_variable(get_variable_name(op), shape=shape)
