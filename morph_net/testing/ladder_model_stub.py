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
r"""A model stub with residual connections and spatial upscaling.

Model:

              -> ----------
             /            |
              -> conv2--(add)---                   -------------
            /                   |                  |           |
image--conv1                     --(add)-> upscale -> conv4--(add) --> output
            \                   |
              -> conv3--(add)---
              \           |
              -> ----------
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
    'conv2': [False, True, True],
    'conv3': [False, True, True],
    'conv4': [False, True, True],
}


REG_STUB = {
    'conv1': [0.1, 0.3, 0.6],
    'conv2': [0.1, 0.3, 0.6],
    'conv3': [0.1, 0.3, 0.6],
    'conv4': [0.1, 0.3, 0.6],
}


MOCK_REG_DICT = {
    'Conv2D':
        op_regularizer_stub.OpRegularizerStubFactory(ALIVE_STUB, REG_STUB)
        .create_conv2d_regularizer
}


def image_stub():
  return op_regularizer_stub.image_stub()


def build_model(image=None):
  """Builds the network described at the top.

  Args:
    image: A 4D tensor to be used as image. If None, image_stub will be used.

  Returns:
    The output op of the network.
  """
  if image is None:
    image = image_stub()

  conv1 = layers.conv2d(image, 3, [3, 3], padding='SAME', scope='conv1')
  conv2 = layers.conv2d(conv1, 3, [3, 3], padding='SAME', scope='conv2')
  conv3 = layers.conv2d(conv1, 3, [3, 3], padding='SAME', scope='conv3')

  res1 = tf.add(conv1, conv2, name='add_1')
  res2 = tf.add(conv1, conv3, name='add_2')
  merged_towers = tf.add(res1, res2, name='add_3')

  upscale = tf.nn.conv2d_transpose(
      merged_towers,
      tf.zeros([2, 2, merged_towers.shape[-1], merged_towers.shape[-1]]),
      tf.shape(merged_towers) * tf.constant([1, 2, 2, 1]),
      [1, 2, 2, 1], 'SAME', name='conv2d_transpose1')
  conv4 = layers.conv2d(
      upscale, 3, [3, 3], padding='SAME', scope='conv4')

  res_top = tf.add(upscale, conv4, name='add_4')

  return res_top.op


def expected_regularization():
  """Build the expected regularization vectors."""
  return {
      'conv1': REG_STUB['conv1'],
      'conv2': REG_STUB['conv2'],
      'conv3': REG_STUB['conv3'],
      'add1': REG_STUB['conv1'],
      'add2': REG_STUB['conv1'],
      'add3': REG_STUB['conv2'],
      'conv2d_transpose1': REG_STUB['conv2'],
      'conv4': REG_STUB['conv4'],
      'add4': REG_STUB['conv4'],
  }


def expected_alive():
  """Build the expected alive vectors."""
  return {
      'conv1': ALIVE_STUB['conv1'],
      'conv2': ALIVE_STUB['conv2'],
      'conv3': ALIVE_STUB['conv3'],
      'add1': ALIVE_STUB['conv1'],
      'add2': ALIVE_STUB['conv1'],
      'add3': ALIVE_STUB['conv2'],
      'conv2d_transpose1': ALIVE_STUB['conv2'],
      'conv4': ALIVE_STUB['conv4'],
      'add4': ALIVE_STUB['conv4'],
  }
