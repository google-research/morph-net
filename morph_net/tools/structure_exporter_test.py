"""Tests for structure_exporter."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import json

from absl.testing import parameterized
from morph_net.network_regularizers import flop_regularizer
from morph_net.tools import structure_exporter as se
import numpy as np

import tensorflow as tf

layers = tf.contrib.layers

LAYERS = ['conv1', 'conv2', 'conv3', 'conv4', 'conv5']

ALIVE = {
    'X/conv1/Conv2D': [False, True, True, True, True, False, True],
    'X/conv2/Conv2D': [True, False, True, False, False],
    'X/conv3/Conv2D': [False, False, True, True],
    'X/concat/Conv2D': [
        False, True, True, False, True, False, True, True, False, True, False,
        False
    ],
    'X/conv4/Conv2D': [
        False, True, True, True, True, False, True, False, False, True, False,
        False
    ],
    'X/conv5/Conv2D': [False, True, False]
}

# Map (remove_prefix) -> (expected_counts)
EXPECTED_COUNTS = {
    False: {
        'X/conv1/Conv2D': 5,
        'X/conv2/Conv2D': 2,
        'X/conv3/Conv2D': 2,
        'X/conv4/Conv2D': 7,
        'X/conv5/Conv2D': 1,
    },
    True: {
        'conv1/Conv2D': 5,
        'conv2/Conv2D': 2,
        'conv3/Conv2D': 2,
        'conv4/Conv2D': 7,
        'conv5/Conv2D': 1,
    },
}


def _build_model():
  image = tf.constant(0.0, shape=[1, 17, 19, 3])
  conv1 = layers.conv2d(image, 7, [7, 5], padding='SAME', scope='X/conv1')
  conv2 = layers.conv2d(image, 5, [1, 1], padding='SAME', scope='X/conv2')
  concat = tf.concat([conv1, conv2], 3)
  conv3 = layers.conv2d(concat, 4, [1, 1], padding='SAME', scope='X/conv3')
  conv4 = layers.conv2d(conv3, 12, [3, 3], padding='SAME', scope='X/conv4')
  conv5 = layers.conv2d(
      concat + conv4, 3, [3, 3], stride=2, padding='SAME', scope='X/conv5')
  return conv5.op


class MockFile(object):

  def __init__(self):
    self.s = ''

  def write(self, s):
    self.s += s

  def read(self):
    return self.s


class TestStructureExporter(parameterized.TestCase, tf.test.TestCase):

  def setUp(self):
    super(TestStructureExporter, self).setUp()
    tf.set_random_seed(12)
    np.random.seed(665544)

  def _batch_norm_scope(self):
    params = {
        'trainable': True,
        'normalizer_fn': layers.batch_norm,
        'normalizer_params': {
            'scale': True
        }
    }

    with tf.contrib.framework.arg_scope([layers.conv2d], **params) as sc:
      return sc

  @parameterized.named_parameters(
      ('Batch', True,),
      ('NoBatch', False),
      ('Batch_removeprefix_export', True, True, True),
      ('NoBatch_removeprefix_export', False, True, True),
  )
  def testStructureExporter(self,
                            use_batch_norm,
                            remove_common_prefix=False,
                            export=False):
    """Tests the export of alive counts.

    Args:
      use_batch_norm: A Boolean. Inidcates if batch norm should be used.
      remove_common_prefix: A Boolean, passed to StructureExporter ctor.
      export: A Boolean. Indicates if the result should be exported to test dir.
    """
    sc = self._batch_norm_scope() if use_batch_norm else []
    with tf.contrib.framework.arg_scope(sc):
      with tf.variable_scope(tf.get_variable_scope()):
        final_op = _build_model()
    variables = {v.name: v for v in tf.trainable_variables()}
    update_vars = []
    if use_batch_norm:
      network_regularizer = flop_regularizer.GammaFlopsRegularizer(
          [final_op], gamma_threshold=1e-6)
      for layer in LAYERS:
        force_alive = ALIVE['X/{}/Conv2D'.format(layer)]
        gamma = variables['X/{}/BatchNorm/gamma:0'.format(layer)]
        update_vars.append(gamma.assign(force_alive * gamma))
    else:
      network_regularizer = flop_regularizer.GroupLassoFlopsRegularizer(
          [final_op], threshold=1e-6)
      print(variables)
      for layer in LAYERS:
        force_alive = ALIVE['X/{}/Conv2D'.format(layer)]
        weights = variables['X/{}/weights:0'.format(layer)]
        update_vars.append(weights.assign(force_alive * weights))
    structure_exporter = se.StructureExporter(
        network_regularizer.op_regularizer_manager, remove_common_prefix)
    with self.cached_session() as sess:
      tf.global_variables_initializer().run()
      sess.run(update_vars)
      structure_exporter.populate_tensor_values(
          sess.run(structure_exporter.tensors))
    expected = EXPECTED_COUNTS[remove_common_prefix]
    self.assertEqual(
        expected,
        structure_exporter.get_alive_counts())
    if export:
      f = MockFile()
      structure_exporter.save_alive_counts(f)
      self.assertEqual(expected, json.loads(f.read()))

  @parameterized.parameters(
      ([], []),
      (['', ''], ['', '']),
      (['a', 'a'], ['a', 'a']),  # No / present
      (['/', '/'], ['', '']),
      (['/x', '/x'], ['x', 'x']),
      (['a/x', 'a/x'], ['x', 'x']),
      (['abc/', 'abc/', 'abc/'], ['', '', '']),
      (['abc/x', 'abc/x', 'abd/x'], ['abc/x', 'abc/x', 'abd/x']),
      (['abc/x', 'abc/y', 'abc/z'], ['x', 'y', 'z']),
      (['abc/x/', 'abc/y', 'abc/z/'], ['x/', 'y', 'z/']),
      (['abc/x/', 'abc/y/', 'abc/z/'], ['x/', 'y/', 'z/']),
  )
  def test_find_common_prefix_size(self, iterable, expected_result):
    rename_op = se.get_remove_common_prefix_op(iterable)
    self.assertEqual(expected_result, list(map(rename_op, iterable)))


if __name__ == '__main__':
  tf.test.main()
