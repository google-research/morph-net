"""An OpRegularizer that applies L1 regularization on batch-norm gammas."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from morph_net.framework import generic_regularizers
from morph_net.framework import tpu_util
import tensorflow.compat.v1 as tf


class GammaL1Regularizer(generic_regularizers.OpRegularizer):
  """An OpRegularizer that L1-regularizes batch-norm gamma."""

  def __init__(self, gamma, gamma_threshold):
    """Creates an instance.

    Args:
      gamma: A tf.Tensor of shape (n_channels,) with the gammas.
      gamma_threshold: A float scalar, the threshold above which a gamma is
        considered 'alive'.
    """
    self._gamma = tpu_util.maybe_convert_to_variable(gamma)
    self._gamma_threshold = gamma_threshold
    abs_gamma = tf.abs(self._gamma)
    self._alive_vector = abs_gamma > gamma_threshold
    self._regularization_vector = abs_gamma

  @property
  def regularization_vector(self):
    return self._regularization_vector

  @property
  def alive_vector(self):
    """Returns a tf.Tensor of shape (n_channels,) with alive bits."""
    return self._alive_vector
