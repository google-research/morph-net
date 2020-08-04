"""Regularizer that groups other regularizers based on channel probability.

See morph_net/framework/grouping_regularizers.py for overview grouping
regularizers.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from morph_net.framework import generic_regularizers
import tensorflow.compat.v1 as tf


class ProbabilisticGroupingRegularizer(generic_regularizers.OpRegularizer):
  """A regularizer that groups others by taking their activation probability."""

  def __init__(self, regularizers_to_group):
    """Creates an instance.

    Args:
      regularizers_to_group: A list of generic_regularizers.OpRegularizer
        objects. Their regularization_vector (alive_vector) are expected to be
        of the same length. These are expected to be probabilistic regularizers.
        Currently, the only supported OpRegularizer is ProbGatingRegularizer.

    Raises:
      ValueError: regularizers_to_group is not of length at least 2.
    """
    if len(regularizers_to_group) < 2:
      raise ValueError('Groups must be of at least size 2.')

    regularization_vectors = []
    alive_vectors = []
    for reg in regularizers_to_group:
      if not hasattr(reg, 'is_probabilistic') or not reg.is_probabilistic:
        raise ValueError('Regularizer is not probabilistic.')
      regularization_vectors.append(
          tf.reshape(reg.regularization_vector, [1, -1]))
      alive_vectors.append(
          tf.reshape(reg.alive_vector, [1, -1]))

    regularization_vectors = tf.concat(regularization_vectors, axis=0)
    alive_vectors = tf.concat(alive_vectors, axis=0)

    # The probability that at least one of the channels is alive:
    # 1 - \prod_i{1 - p_i}.
    self._regularization_vector = 1.0 - tf.reduce_prod(
        1.0 - regularization_vectors, axis=0)
    self._alive_vector = tf.reduce_any(alive_vectors, axis=0)

  @property
  def regularization_vector(self):
    return self._regularization_vector

  @property
  def alive_vector(self):
    return self._alive_vector
