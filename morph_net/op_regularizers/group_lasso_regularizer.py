"""A regularizer based on group-lasso.

All the weights that are related to a single output are grouped into one LASSO
group (https://arxiv.org/pdf/1611.06321.pdf).
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from morph_net.framework import generic_regularizers
from morph_net.framework import tpu_util
import tensorflow.compat.v1 as tf


class GroupLassoRegularizer(generic_regularizers.OpRegularizer):
  """A regularizer for convolutions and matmul operations, based on group-lasso.

  Supported ops: Conv2D, Conv2DBackpropInput (transposed Conv2D), and MatMul
  are supported. The grouping is done according to the formula:

  (1 - l1_fraction) * L2(weights) / sqrt(dim) + l1_fraction * L1(weights) / dim,

  where `dim` is the number of weights associated with an activation, L2 and L1
  are the respective norms, and l1_fraction controls the balance between L1 and
  L2 grouping. The paper cited above experiments with 0.0 and 0.5 for
  l1_fraction.
  """

  def __init__(self, weight_tensor, reduce_dims, threshold, l1_fraction=0.0):
    """Creates an instance.

    Args:
      weight_tensor: A tensor with the weights of the op (potentially sliced).
      reduce_dims: A tuple indictaing the dimensions of `weight_tensor`
        to reduce over. Most often it will include all dimensions except
        the output size.
      threshold: A float. When the norm of the group associated with an
        activation is below the threshold, it will be considered dead.
      l1_fraction: A float, controls the balance between L1 and L2 grouping (see
        above).
    """
    weight_tensor = tpu_util.maybe_convert_to_variable(weight_tensor)
    if l1_fraction < 0.0 or l1_fraction > 1.0:
      raise ValueError(
          'l1_fraction should be in [0.0, 1.0], not %e.' % l1_fraction)

    self._threshold = threshold
    l2_norm = tf.sqrt(
        tf.reduce_mean(tf.square(weight_tensor), axis=reduce_dims))
    if l1_fraction > 0.0:
      l1_norm = tf.reduce_mean(tf.abs(weight_tensor), axis=reduce_dims)
      norm = l1_fraction * l1_norm + (1.0 - l1_fraction) * l2_norm
    else:
      norm = l2_norm

    self._regularization_vector = norm
    self._alive_vector = norm > threshold

  @property
  def regularization_vector(self):
    return self._regularization_vector

  @property
  def alive_vector(self):
    return self._alive_vector
