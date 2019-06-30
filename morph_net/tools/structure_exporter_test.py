"""Tests for structure_exporter."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import json
import os

from absl import flags
from absl.testing import parameterized

from morph_net.framework import generic_regularizers
from morph_net.framework import op_regularizer_manager as orm
from morph_net.tools import structure_exporter as se
import tensorflow as tf


FLAGS = flags.FLAGS
layers = tf.contrib.layers


def _alive_from_file(filename):
  with tf.gfile.Open(os.path.join(FLAGS.test_tmpdir, filename)) as f:
    return json.loads(f.read())


class FakeOpReg(generic_regularizers.OpRegularizer):

  def __init__(self, alive):
    self.alive = alive

  @property
  def alive_vector(self):
    return self.alive

  @property
  def regularization_vector(self):
    assert False
    return 0


class FakeORM(orm.OpRegularizerManager):

  def __init__(self):
    image = tf.constant(0.0, shape=[1, 17, 19, 3])
    layers.conv2d(image, 3, 2, scope='X/c1')
    layers.conv2d(image, 4, 3, scope='X/c2')
    layers.conv2d_transpose(image, 5, 1, scope='X/c3')
    self.regularizer = {
        'X/c1/Conv2D': FakeOpReg([True, False, True]),
        'X/c2/Conv2D': FakeOpReg([True, False, True, False]),
        'X/c3/conv2d_transpose': FakeOpReg([True, True, False, True, False])
    }
    self.ops = [
        tf.get_default_graph().get_operation_by_name(op_name)
        for op_name in self.regularizer
    ]

  def ops(self):
    return self.ops

  def get_regularizer(self, op):
    return self.regularizer[op.name]


class TestStructureExporter(parameterized.TestCase, tf.test.TestCase):

  def setUp(self):
    super(TestStructureExporter, self).setUp()
    self.exporter = se.StructureExporter(
        op_regularizer_manager=FakeORM(), remove_common_prefix=False)
    self.tensor_value_1 = {
        'X/c1/Conv2D': [True] * 3,
        'X/c2/Conv2D': [False] * 4,
        'X/c3/conv2d_transpose': [True] * 5
    }
    self.expected_alive_1 = {
        'X/c1/Conv2D': 3,
        'X/c2/Conv2D': 0,
        'X/c3/conv2d_transpose': 5
    }

    self.tensor_value_2 = {
        'X/c1/Conv2D': [True, False, False],
        'X/c2/Conv2D': [True] * 4,
        'X/c3/conv2d_transpose': [False, False, False, False, True]
    }
    self.expected_alive_2 = {
        'X/c1/Conv2D': 1,
        'X/c2/Conv2D': 4,
        'X/c3/conv2d_transpose': 1
    }

  def test_tensors(self):
    expected = {
        'X/c1/Conv2D': [1, 0, 1],
        'X/c2/Conv2D': [1, 0, 1, 0],
        'X/c3/conv2d_transpose': [1, 1, 0, 1, 0]
    }
    self.assertAllEqual(sorted(self.exporter.tensors), sorted(expected))
    for name in self.exporter.tensors:
      self.assertAllEqual(self.exporter.tensors[name], expected[name])

  def test_populate_tensor_values(self):
    self.exporter.populate_tensor_values(self.tensor_value_1)
    self.assertAllEqual(self.exporter.get_alive_counts(), self.expected_alive_1)
    self.exporter.populate_tensor_values(self.tensor_value_2)
    self.assertAllEqual(self.exporter.get_alive_counts(), self.expected_alive_2)

  def test_compute_alive_count(self):
    self.assertAllEqual(
        se._compute_alive_counts({'a': [True, False, False]}), {'a': 1})
    self.assertAllEqual(
        se._compute_alive_counts({'b': [False, False]}), {'b': 0})
    self.assertAllEqual(
        se._compute_alive_counts(self.tensor_value_1), self.expected_alive_1)
    self.assertAllEqual(
        se._compute_alive_counts(self.tensor_value_2), self.expected_alive_2)

  def test_save_alive_counts(self):
    filename = 'alive007'
    self.exporter.populate_tensor_values(self.tensor_value_1)
    with tf.gfile.Open(os.path.join(FLAGS.test_tmpdir, filename), 'w') as f:
      self.exporter.save_alive_counts(f)
    self.assertAllEqual(_alive_from_file(filename), self.expected_alive_1)

  def test_create_file_and_save_alive_counts(self):
    base_dir = os.path.join(FLAGS.test_tmpdir, 'ee')

    self.exporter.populate_tensor_values(self.tensor_value_1)
    self.exporter.create_file_and_save_alive_counts(base_dir, 19)
    self.assertAllEqual(
        _alive_from_file('ee/learned_structure/alive_19'),
        self.expected_alive_1)
    self.assertAllEqual(
        _alive_from_file('ee/learned_structure/alive'), self.expected_alive_1)

    self.exporter.populate_tensor_values(self.tensor_value_2)
    self.exporter.create_file_and_save_alive_counts(base_dir, 1009)
    self.assertAllEqual(
        _alive_from_file('ee/learned_structure/alive_1009'),
        self.expected_alive_2)
    self.assertAllEqual(
        _alive_from_file('ee/learned_structure/alive_19'),
        self.expected_alive_1)
    self.assertAllEqual(
        _alive_from_file('ee/learned_structure/alive'), self.expected_alive_2)

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
    rename_op = se.get_remove_common_prefix_fn(iterable)
    self.assertEqual(expected_result, list(map(rename_op, iterable)))


class TestStructureExporterRemovePrefix(tf.test.TestCase):

  def test_removes_prefix(self):
    exporter = se.StructureExporter(
        op_regularizer_manager=FakeORM(), remove_common_prefix=True)
    expected = ['c1/Conv2D', 'c2/Conv2D', 'c3/conv2d_transpose']
    self.assertAllEqual(sorted(exporter.tensors), sorted(expected))

if __name__ == '__main__':
  tf.test.main()
