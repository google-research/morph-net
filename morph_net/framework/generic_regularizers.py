# Copyright 2018 The TensorFlow Authors All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Interface for MorphNet regularizers framework."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import abc


class NetworkRegularizer(object):  # pytype: disable=ignored-metaclass
  """An interface for Network Regularizers."""
  __metaclass__ = abc.ABCMeta

  @abc.abstractmethod
  def get_regularization_term(self, ops=None):
    """Compute the regularization term.

    Args:
      ops: A list of tf.Operation. If not specified, all ops that the
        NetworkRegularizer is aware of are implied.

    Returns:
      A tf.Tensor scalar of floating point type that evaluates to the
        regularization term that should be added to the total loss, with a
        suitable coefficient.
    """
    pass

  @abc.abstractmethod
  def get_cost(self, ops=None):
    """Calculates the cost targeted by the Regularizer.

    Args:
      ops: A list of tf.Operation objects. Same as get_regularization_term, but
        returns total cost implied by the regularization term.

    Returns:
      A tf.Tensor scalar that evaluates to the cost.
    """
    pass

  @property
  def op_regularizer_manager(self):
    """Returns the OpRegularizerManager managing the graph's OpRegularizers.

    If the NetworkRegularizer subclass is not using an OpRegularizerManager,
    None is returned.
    """
    return None

  @property
  def name(self):
    """Name of network regularizer.."""
    return ''

  @property
  def cost_name(self):
    """Name of the cost targeted by network regularizer."""
    return ''


class OpRegularizer(object):  # pytype: disable=ignored-metaclass
  """An interface for Op Regularizers.

  An OpRegularizer object corresponds to a tf.Operation, and provides
  a regularizer for the output of the op (we assume that the op has one output
  of interest in the context of MorphNet).
  """
  __metaclass__ = abc.ABCMeta

  @abc.abstractproperty
  def regularization_vector(self):
    """Returns a vector of floats, with regularizers.

    The length of the vector is the number of "output activations" (call them
    neurons, nodes, filters, etc.) of the op. For a convolutional network, it's
    the number of filters (aka "depth"). For a fully-connected layer, it's
    usually the second (and last) dimension - assuming the first one is the
    batch size.
    """
    pass

  @abc.abstractproperty
  def alive_vector(self):
    """Returns a vector of booleans, indicating which activations are alive.

    Call them activations, neurons, nodes, filters, etc. This vector is of the
    same length as the regularization_vector.
    """
    pass


def dimensions_are_compatible(op_regularizer):
  """Checks if op_regularizer's alive_vector matches regularization_vector."""
  return op_regularizer.alive_vector.shape.with_rank(1).dims[
      0].is_compatible_with(
          op_regularizer.regularization_vector.shape.with_rank(1).dims[0])
