"""Tests for structure_exporter."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import json
import os

from absl import flags
from absl.testing import parameterized

from morph_net.framework import batch_norm_source_op_handler
from morph_net.framework import concat_op_handler
from morph_net.framework import generic_regularizers
from morph_net.framework import grouping_op_handler
from morph_net.framework import op_regularizer_manager as orm
from morph_net.framework import output_non_passthrough_op_handler
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


arg_scope = tf.contrib.framework.arg_scope
conv2d_transpose = tf.contrib.layers.conv2d_transpose
conv2d = tf.contrib.layers.conv2d
FLAGS = flags.FLAGS


def assign_to_gamma(scope, value):
  name_to_var = {v.op.name: v for v in tf.global_variables()}
  gamma = name_to_var[scope + '/BatchNorm/gamma']
  gamma.assign(value).eval()


def jsons_exist_in_tempdir():
  for f in tf.gfile.ListDirectory(FLAGS.test_tmpdir):
    if f.startswith('alive') or f.startswith('reg'):
      return True
  return False


class StructureExporterOpTest(tf.test.TestCase):

  def empty_test_dir(self):
    for f in tf.gfile.ListDirectory(FLAGS.test_tmpdir):
      if f.startswith('alive') or f.startswith('reg'):
        print('found f', f)
        tf.gfile.Remove(os.path.join(FLAGS.test_tmpdir, f))

  def setUp(self):
    super(StructureExporterOpTest, self).setUp()
    self.empty_test_dir()
    params = {
        'trainable': True,
        'normalizer_fn': tf.contrib.layers.batch_norm,
        'normalizer_params': {
            'scale': True
        },
        'padding': 'SAME'
    }

    image = tf.zeros([3, 10, 10, 3])
    with arg_scope([conv2d, conv2d_transpose], **params):
      conv1 = conv2d(image, 5, 3, scope='conv1')
      conv2 = conv2d(image, 5, 3, scope='conv2')
      add = conv1 + conv2
      conv3 = conv2d(add, 4, 1, scope='conv3')
      convt = conv2d_transpose(conv3, 3, 2, scope='convt')
    # Create OpHandler dict for test.
    op_handler_dict = collections.defaultdict(
        grouping_op_handler.GroupingOpHandler)
    op_handler_dict.update({
        'FusedBatchNorm':
            batch_norm_source_op_handler.BatchNormSourceOpHandler(0.1),
        'Conv2D':
            output_non_passthrough_op_handler.OutputNonPassthroughOpHandler(),
        'Conv2DBackpropInput':
            output_non_passthrough_op_handler.OutputNonPassthroughOpHandler(),
        'ConcatV2':
            concat_op_handler.ConcatOpHandler(),
    })

    # Create OpRegularizerManager and NetworkRegularizer for test.
    opreg_manager = orm.OpRegularizerManager(
        [convt.op], op_handler_dict)
    self.exporter = se.StructureExporterOp(
        directory=FLAGS.test_tmpdir,
        save=True,
        opreg_manager=opreg_manager)

  def test_simple_export(self):
    export_op = self.exporter.export()
    with self.cached_session():
      tf.global_variables_initializer().run()
      assign_to_gamma('conv1', [0, 1.5, 1, 0, 1])
      assign_to_gamma('conv2', [1, 1, .1, 0, 1])
      assign_to_gamma('conv3', [0, .8, 1, .25])
      assign_to_gamma('convt', [3, .3, .03])
      export_op.run()
    regularizers = self._read_file(reg=True)
    grouped_conv1_conv2_reg = [1, 1.5, 1, 0, 1]
    self.assertAllClose(grouped_conv1_conv2_reg, regularizers['conv1/Conv2D'])
    self.assertAllClose(grouped_conv1_conv2_reg, regularizers['conv2/Conv2D'])
    self.assertAllClose([0, .8, 1, .25], regularizers['conv3/Conv2D'])
    self.assertAllClose([3, .3, .03], regularizers['convt/conv2d_transpose'])

    alive = self._read_file(reg=False)
    self.assertAllEqual(4, alive['conv1/Conv2D'])
    self.assertAllEqual(4, alive['conv2/Conv2D'])
    self.assertAllEqual(3, alive['conv3/Conv2D'])
    self.assertAllEqual(2, alive['convt/conv2d_transpose'])

  def test_export_every_n(self):
    export_op = self.exporter.export_state_every_n(
        4, se.ExportInfo.alive)
    with self.cached_session():
      tf.initialize_all_variables().run()
      # Initially no jsons.
      self.assertFalse(jsons_exist_in_tempdir())
      export_op.run()
      # 0th iteration: jsons are saved, verified and deleted.
      self.assertTrue(jsons_exist_in_tempdir())
      self.empty_test_dir()
      for _ in range(3):
        # Itertion 1, 2, 3: jsons are not saved.
        export_op.run()
        self.assertFalse(jsons_exist_in_tempdir())
      # 4th iteration: saved again.
      export_op.run()
      self.assertTrue(jsons_exist_in_tempdir())
      self.empty_test_dir()
      # 5th: not saved.
      export_op.run()
      self.assertFalse(jsons_exist_in_tempdir())

  def _read_file(self, reg):
    filename = se._REG_FILENAME if reg else se._ALIVE_FILENAME
    with tf.gfile.Open(os.path.join(FLAGS.test_tmpdir, filename)) as f:
      data = f.read()
      return json.loads(data)

if __name__ == '__main__':
  tf.test.main()
