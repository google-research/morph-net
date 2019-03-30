"""Utility functions for handling TPU graphs."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl import logging
import tensorflow as tf


def maybe_convert_to_cpu(gamma):
  """Convert TPU gammas to the equivalent variables available on CPU."""
  if gamma.op.type != 'ReadVariableOp' or 'BatchNorm' not in gamma.op.name:
    # We are looking for resource variables containing gammas. This isn't one.
    logging.info('Not replacing %s.', gamma.name)
    return gamma

  with tf.variable_scope(tf.get_variable_scope(), reuse=tf.AUTO_REUSE):
    base_name = gamma.name.rsplit('/', 1)[0]
    shape = gamma.shape
    cpu_variable = tf.get_variable(base_name + '/gamma', shape=shape)
    logging.info('Replacing %s with %s.', gamma.name, cpu_variable.name)
    return tf.convert_to_tensor(cpu_variable)
