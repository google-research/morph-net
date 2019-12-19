"""An OpRegularizer with 0 regularization and always alive."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from morph_net.framework import generic_regularizers
import tensorflow.compat.v1 as tf


class ConstantOpRegularizer(generic_regularizers.OpRegularizer):
  """An OpRegularizer with 0 regularization and always alive."""

  def __init__(self, size):
    """Creates an instance.

    Args:
      size: Integer size of the regularizer.
    """
    self._regularization_vector = tf.zeros(size)
    self._alive_vector = tf.ones(size, dtype=tf.bool)

  @property
  def regularization_vector(self):
    return self._regularization_vector

  @property
  def alive_vector(self):
    return self._alive_vector
