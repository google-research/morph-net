"""An OpRegularizer that applies regularization on Gating probabilities.

This regularizer targets the gating probability of a LogisticSigmoidGating OP.
It can do so directly by minimizing the log odds ratio of the probability
log(p/1-p), or by minimizing the L1 of the sampled mask.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from morph_net.framework import generic_regularizers
from morph_net.framework import tpu_util

import tensorflow.compat.v1 as tf


class ProbGatingRegularizer(generic_regularizers.OpRegularizer):
  """An OpRegularizer that regularizes gating probabilities."""

  def __init__(self, logits, mask, regularize_on_mask=True,
               alive_threshold=0.1, mask_as_alive_vector=True):
    """Creates an instance.

    Args:
      logits: A tf.Tensor of shape (n_channels,) with the
        log odds ratio of the channel being on: log(p / 1-p).
      mask: A tf.Tensor of the same shape as `logits`.
        The sampled mask/gating vector.
      regularize_on_mask: Bool. If True uses the mask as the
        regularization vector. Else uses probabilities. Default True.
      alive_threshold: Float. Threshold below which values are considered dead.
        This can be used both when mask_as_alive_vector is True and then the
        threshold is used to binarize the sampled values and
        when mask_as_alive_vector is False, and then the threshold is on the
        channel probability.
      mask_as_alive_vector: Bool. If True use the thresholded sampled mask
        as the alive vector. Else, use thresholded probabilities from the
        logits.
    """
    if len(logits.shape.as_list()) != 1:
      raise ValueError('logits tensor should be 1D.')
    if len(mask.shape.as_list()) != 1:
      raise ValueError('mask tensor should be 1D.')

    self._logits = tpu_util.maybe_convert_to_variable(logits)
    self._mask = mask
    self._probs = tf.sigmoid(self._logits)

    alive_vector = self._mask if mask_as_alive_vector else self._probs
    self._alive_vector = alive_vector >= alive_threshold

    self._regularization_vector = (
        self._mask if regularize_on_mask else self._probs)

  @property
  def regularization_vector(self):
    return self._regularization_vector

  @property
  def alive_vector(self):
    """Returns a tf.Tensor of shape (n_channels,) with alive bits."""
    return self._alive_vector

  @property
  def is_probabilistic(self):
    return True
