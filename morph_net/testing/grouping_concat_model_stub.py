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
r"""A model stub with convolutions and non-channal concats residual connections.

Model:

             -> conv1 -\                      conv4 --
            /           |                     /        \
      image          [concat axis=1]  -> conv3   -->   [concat axis=2]
            \           |
             -> conv2 -/
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from morph_net.testing import op_regularizer_stub
import tensorflow.compat.v1 as tf
from tensorflow.contrib import layers as contrib_layers

layers = contrib_layers

ALIVE_STUB = {
    'conv1': [False, True, True],
    'conv2': [True, False, True],
    'conv3': [False, False, True, True],
    'conv4': [False, True, False, True],
}

REG_STUB = {
    'conv1': [0.1, 0.3, 0.6],
    'conv2': [0.15, 0.25, 0.05],
    'conv3': [0.07, 0.17, 0.57, 0.37],
    'conv4': [0.06, 100.87, 0.17, 1.57],
}


MOCK_REG_DICT = {
    'Conv2D':
        op_regularizer_stub.OpRegularizerStubFactory(ALIVE_STUB, REG_STUB)
        .create_conv2d_regularizer
}


def image_stub():
  return op_regularizer_stub.image_stub()


def build_model(image=None):
  """Builds the model network described at the top of the file.

  Args:
    image: A 4D tensor to be used as image. If None, image_stub will be used.

  Returns:
    The output op of the network.
  """
  if image is None:
    image = image_stub()
  conv1 = layers.conv2d(image, 3, [7, 5], padding='SAME', scope='conv1')
  conv2 = layers.conv2d(image, 3, [1, 1], padding='SAME', scope='conv2')
  concat = tf.concat([conv1, conv2], 1, 'concat1')
  conv3 = layers.conv2d(concat, 4, [1, 1], padding='SAME', scope='conv3')
  conv4 = layers.conv2d(conv3, 4, [3, 3], padding='SAME', scope='conv4')
  concat2 = tf.concat([conv4, conv3], 2, 'concat2')
  return concat2.op


def expected_regularization():
  """Build the expected regularization vectors."""
  # Grouping: Regularization grouping is the max of the constituents.
  reg1 = [max(a, b) for a, b in zip(REG_STUB['conv2'], REG_STUB['conv1'])]
  reg2 = [max(a, b) for a, b in zip(REG_STUB['conv3'], REG_STUB['conv4'])]
  return {
      'conv1': reg1,
      'conv2': reg1,
      'conv3': reg2,
      'conv4': reg2,
      'concat1': reg1,
      'concat2': reg2
  }


def expected_alive():
  """Build the expected alive vectors."""
  alive1 = [a or b for a, b in zip(ALIVE_STUB['conv2'], ALIVE_STUB['conv1'])]
  alive2 = [a or b for a, b in zip(ALIVE_STUB['conv3'], ALIVE_STUB['conv4'])]
  return {
      'conv1': alive1,
      'conv2': alive1,
      'conv3': alive2,
      'conv4': alive2,
      'concat1': alive1,
      'concat2': alive2
  }
