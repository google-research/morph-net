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
r"""Helpers for testing the regularizers framework.

Contains logic for creating Stubs for OpRegularizers for the
convolutions in a model.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from morph_net.framework import generic_regularizers
import tensorflow.compat.v1 as tf
from tensorflow.contrib import layers as contrib_layers

layers = contrib_layers


class OpRegularizerStub(generic_regularizers.OpRegularizer):
  """A stub that exponses a constant regularization_vector and alive_vector."""

  def __init__(self, regularization_vector, alive_vector):
    self._regularization_vector = tf.constant(
        regularization_vector, dtype=tf.float32)
    self._alive_vector = tf.constant(alive_vector, dtype=tf.bool)

  @property
  def regularization_vector(self):
    return self._regularization_vector

  @property
  def alive_vector(self):
    return self._alive_vector


class OpRegularizerStubFactory(object):
  """OpRegularizerStub factory to be used in testing."""

  def __init__(self, alive_stub_dict, reg_stub_dict):
    self._alive_stub = alive_stub_dict
    self._reg_stub = reg_stub_dict

  def _create_stub(self, key):
    return OpRegularizerStub(self._reg_stub[key], self._alive_stub[key])

  def create_conv2d_regularizer(self, conv_op, manager=None):
    del manager  # unused
    for key in self._reg_stub:
      if conv_op.name.startswith(key):
        return self._create_stub(key)
    raise ValueError('No regularizer for %s' % conv_op.name)


def image_stub():
  return tf.constant(0.0, shape=[1, 17, 19, 3])
