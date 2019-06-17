"""Tests for op_regularizer_manager."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
from absl.testing import parameterized
from morph_net.framework import batch_norm_source_op_handler
from morph_net.framework import concat_op_handler
from morph_net.framework import conv2d_source_op_handler
from morph_net.framework import depthwise_convolution_op_handler
from morph_net.framework import generic_regularizers
from morph_net.framework import grouping_op_handler
from morph_net.framework import op_regularizer_manager as orm
from morph_net.framework import output_non_passthrough_op_handler
from morph_net.testing import add_concat_model_stub
from morph_net.testing import grouping_concat_model_stub
import numpy as np
import tensorflow as tf

arg_scope = tf.contrib.framework.arg_scope
layers = tf.contrib.layers

DEBUG_PRINTS = False


def _get_op(name):
  return tf.get_default_graph().get_operation_by_name(name)


class OpRegularizerManagerTest(parameterized.TestCase, tf.test.TestCase):

  def setUp(self):
    super(OpRegularizerManagerTest, self).setUp()
    tf.set_random_seed(12)
    np.random.seed(665544)
    IndexOpRegularizer.reset_index()

    # Create default OpHandler dict for testing.
    self._default_op_handler_dict = collections.defaultdict(
        grouping_op_handler.GroupingOpHandler)
    self._default_op_handler_dict.update({
        'FusedBatchNormV3':
            IndexBatchNormSourceOpHandler(),
        'Conv2D':
            output_non_passthrough_op_handler.OutputNonPassthroughOpHandler(),
        'ConcatV2':
            concat_op_handler.ConcatOpHandler(),
        'DepthwiseConv2dNative':
            depthwise_convolution_op_handler.DepthwiseConvolutionOpHandler(),
    })

  def _batch_norm_scope(self):
    params = {
        'trainable': True,
        'normalizer_fn': layers.batch_norm,
        'normalizer_params': {
            'scale': True
        }
    }

    with arg_scope([layers.conv2d], **params) as sc:
      return sc

  @parameterized.named_parameters(('Batch_no_par1', True, False, 'conv1'),
                                  ('Batch_par1', True, True, 'conv1'),
                                  ('NoBatch_no_par1', False, False, 'conv1'),
                                  ('NoBatch_par2', False, True, 'conv2'),
                                  ('Batch_no_par2', True, False, 'conv2'),
                                  ('Batch_par2', True, True, 'conv2'),
                                  ('Batch_par3', True, True, 'conv3'),
                                  ('NoBatch_par3', False, True, 'conv3'),
                                  ('NoBatch_no_par3', False, False, 'conv3'))
  def testSimpleOpGetRegularizer(self, use_batch_norm, use_partitioner, scope):
    # Tests the alive pattern of the conv and relu ops.
    # use_batch_norm: A Boolean. Indicates if batch norm should be used.
    # use_partitioner: A Boolean. Indicates if a fixed_size_partitioner should
    #   be used.
    # scope: A String with the scope to test.
    sc = self._batch_norm_scope() if use_batch_norm else []
    partitioner = tf.fixed_size_partitioner(2) if use_partitioner else None
    model_stub = add_concat_model_stub
    with arg_scope(sc):
      with tf.variable_scope(tf.get_variable_scope(), partitioner=partitioner):
        final_op = add_concat_model_stub.build_model()

    # Instantiate OpRegularizerManager.
    op_handler_dict = self._default_op_handler_dict
    op_handler_dict['FusedBatchNormV3'] = StubBatchNormSourceOpHandler(
        model_stub)
    if not use_batch_norm:
      op_handler_dict['Conv2D'] = StubConv2DSourceOpHandler(model_stub)
    op_reg_manager = orm.OpRegularizerManager([final_op], op_handler_dict)

    expected_alive = model_stub.expected_alive()
    conv_reg = op_reg_manager.get_regularizer(_get_op(scope + '/Conv2D'))
    self.assertAllEqual(expected_alive[scope], conv_reg.alive_vector)

    relu_reg = op_reg_manager.get_regularizer(_get_op(scope + '/Relu'))
    self.assertAllEqual(expected_alive[scope], relu_reg.alive_vector)

  @parameterized.named_parameters(('Batch_no_par', True, False),
                                  ('Batch_par', True, True),
                                  ('NoBatch_no_par', False, False),
                                  ('NoBatch_par', False, True))
  def testConcatOpGetRegularizer(self, use_batch_norm, use_partitioner):
    sc = self._batch_norm_scope() if use_batch_norm else []
    partitioner = tf.fixed_size_partitioner(2) if use_partitioner else None
    model_stub = add_concat_model_stub
    with arg_scope(sc):
      with tf.variable_scope(tf.get_variable_scope(), partitioner=partitioner):
        final_op = add_concat_model_stub.build_model()

    # Instantiate OpRegularizerManager.
    op_handler_dict = self._default_op_handler_dict
    op_handler_dict['FusedBatchNormV3'] = StubBatchNormSourceOpHandler(
        model_stub)
    if not use_batch_norm:
      op_handler_dict['Conv2D'] = StubConv2DSourceOpHandler(model_stub)
    op_reg_manager = orm.OpRegularizerManager([final_op], op_handler_dict)

    expected_alive = model_stub.expected_alive()
    expected = np.logical_or(expected_alive['conv4'], expected_alive['concat'])
    conv_reg = op_reg_manager.get_regularizer(_get_op('conv4/Conv2D'))
    self.assertAllEqual(expected, conv_reg.alive_vector)

    relu_reg = op_reg_manager.get_regularizer(_get_op('conv4/Relu'))
    self.assertAllEqual(expected, relu_reg.alive_vector)

  @parameterized.named_parameters(
      ('_conv1', 'conv1/Conv2D', 'conv1'),
      ('_conv2', 'conv2/Conv2D', 'conv2'),
      ('_conv3', 'conv3/Conv2D', 'conv3'),
      ('_conv4', 'conv4/Conv2D', 'conv4'),
  )
  def testGroupConcatOpGetRegularizerValues(self, op_name, short_name):
    model_stub = grouping_concat_model_stub
    with arg_scope(self._batch_norm_scope()):
      with tf.variable_scope(tf.get_variable_scope()):
        final_op = model_stub.build_model()

    # Instantiate OpRegularizerManager.
    op_handler_dict = self._default_op_handler_dict
    op_handler_dict['FusedBatchNormV3'] = StubBatchNormSourceOpHandler(
        model_stub)

    op_reg_manager = orm.OpRegularizerManager([final_op], op_handler_dict)

    expected_alive = model_stub.expected_alive()
    expected_reg = model_stub.expected_regularization()

    reg = op_reg_manager.get_regularizer(_get_op(op_name))
    self.assertAllEqual(expected_alive[short_name], reg.alive_vector)
    self.assertAllClose(expected_reg[short_name], reg.regularization_vector)

  def testGroupConcatOpGetRegularizerObjects(self):
    model_stub = grouping_concat_model_stub
    with arg_scope(self._batch_norm_scope()):
      with tf.variable_scope(tf.get_variable_scope()):
        final_op = model_stub.build_model()

    # Instantiate OpRegularizerManager.
    op_handler_dict = self._default_op_handler_dict
    op_handler_dict['FusedBatchNormV3'] = StubBatchNormSourceOpHandler(
        model_stub)

    op_reg_manager = orm.OpRegularizerManager([final_op], op_handler_dict)
    self.assertEqual(
        op_reg_manager.get_regularizer(_get_op('conv1/Conv2D')),
        op_reg_manager.get_regularizer(_get_op('conv2/Conv2D')))
    self.assertEqual(
        op_reg_manager.get_regularizer(_get_op('conv3/Conv2D')),
        op_reg_manager.get_regularizer(_get_op('conv4/Conv2D')))

  @parameterized.named_parameters(('Concat_5', True, 5),
                                  ('Concat_7', True, 7),
                                  ('Add_6', False, 6))
  def testGetRegularizerForConcatWithNone(self, test_concat, depth):
    image = tf.constant(0.0, shape=[1, 17, 19, 3])
    conv2 = layers.conv2d(image, 5, [1, 1], padding='SAME', scope='conv2')
    other_input = tf.add(
        tf.identity(tf.constant(3.0, shape=[1, 17, 19, depth])), 3.0)
    # other_input has None as regularizer.
    concat = tf.concat([other_input, conv2], 3)
    output = tf.add(concat, concat, name='output_out')
    op = concat.op if test_concat else output.op

    # Instantiate OpRegularizerManager.
    op_handler_dict = self._default_op_handler_dict
    op_handler_dict['Conv2D'] = StubConv2DSourceOpHandler(add_concat_model_stub)
    op_reg_manager = orm.OpRegularizerManager([output.op], op_handler_dict)

    expected_alive = add_concat_model_stub.expected_alive()
    alive = op_reg_manager.get_regularizer(op).alive_vector
    self.assertAllEqual([True] * depth, alive[:depth])
    self.assertAllEqual(expected_alive['conv2'], alive[depth:])

  @parameterized.named_parameters(('add', tf.add),
                                  ('div', tf.divide),
                                  ('mul', tf.multiply),
                                  ('max', tf.maximum),
                                  ('min', tf.minimum),
                                  ('l2', tf.squared_difference))
  def testGroupingOps(self, tested_op):
    th = 0.5
    image = tf.constant(0.5, shape=[1, 17, 19, 3])

    conv1 = layers.conv2d(image, 5, [1, 1], padding='SAME', scope='conv1')
    conv2 = layers.conv2d(image, 5, [1, 1], padding='SAME', scope='conv2')
    res = tested_op(conv1, conv2)

    # Instantiate OpRegularizerManager.
    op_handler_dict = self._default_op_handler_dict
    op_handler_dict['Conv2D'] = RandomConv2DSourceOpHandler(th)
    op_reg_manager = orm.OpRegularizerManager([res.op], op_handler_dict)

    alive = op_reg_manager.get_regularizer(res.op).alive_vector
    conv1_reg = op_reg_manager.get_regularizer(conv1.op).regularization_vector
    conv2_reg = op_reg_manager.get_regularizer(conv2.op).regularization_vector
    with self.session():
      self.assertAllEqual(alive, np.logical_or(conv1_reg.eval() > th,
                                               conv2_reg.eval() > th))

  def testCascadedGrouping(self):
    inputs = tf.zeros([6, 8, 8, 10], name='prev')
    with arg_scope(
        [layers.conv2d, layers.max_pool2d],
        kernel_size=1,
        stride=1,
        padding='SAME'):
      net = layers.conv2d(inputs, 17, scope='conv/input')

      first = layers.conv2d(net, num_outputs=17, scope='conv/first')
      add_0 = tf.add(first, net, 'Add/first')  # So conv/first must be 17.
      second = layers.conv2d(add_0, num_outputs=17, scope='conv/second')
      out = tf.add(net, second, 'Add/second')  # So conv/second must be 17.

    # Instantiate OpRegularizerManager.
    op_handler_dict = self._default_op_handler_dict
    op_handler_dict['Conv2D'] = IndexConv2DSourceOpHandler()
    op_reg_manager = orm.OpRegularizerManager([out.op], op_handler_dict)

    grouped_names = [
        [op_slice.op.name for op_slice in group.op_slices]
        for group in op_reg_manager._op_group_dict.values()]
    expected = set([
        'conv/second/Conv2D', 'Add/second', 'conv/first/Conv2D',
        'conv/input/Conv2D', 'Add/first'
    ])
    groups = []
    for group in grouped_names:
      filtered = []
      for op_name in group:
        if '/Conv2D' in op_name or 'Add/' in op_name:
          filtered.append(op_name)
      if filtered:
        groups.append(set(filtered))
        if DEBUG_PRINTS:
          print('Group Found = ', filtered)
    self.assertIn(expected, groups)

  def testBroadcast(self):
    with arg_scope(self._batch_norm_scope()):
      inputs = tf.zeros([2, 4, 4, 3])
      c1 = layers.conv2d(inputs, num_outputs=1, kernel_size=3, scope='conv1')
      c2 = layers.conv2d(inputs, num_outputs=10, kernel_size=3, scope='conv2')
      tmp = c1 + c2
      final_op = layers.conv2d(
          tmp, num_outputs=13, kernel_size=3, scope='conv3')

    manager = orm.OpRegularizerManager(
        [final_op.op], self._default_op_handler_dict)

    c1_reg = manager.get_regularizer(_get_op('conv1/Conv2D'))
    c2_reg = manager.get_regularizer(_get_op('conv2/Conv2D'))
    add_reg = manager.get_regularizer(_get_op('add'))
    c3_reg = manager.get_regularizer(_get_op('conv3/Conv2D'))

    expected_c1_reg_size = 1
    self.assertEqual(expected_c1_reg_size, c1_reg.regularization_vector.shape)
    self.assertEqual(10, c2_reg.regularization_vector.shape)
    self.assertEqual(10, add_reg.regularization_vector.shape)
    self.assertEqual(13, c3_reg.regularization_vector.shape)

    c1_slice = manager.get_op_slices(c1.op)[0]
    c1_group = manager.get_op_group(c1_slice)
    c2_slice = manager.get_op_slices(c2.op)[0]
    c2_group = manager.get_op_group(c2_slice)
    add_slice = manager.get_op_slices(tmp.op)[0]
    add_group = manager.get_op_group(add_slice)
    c3_slice = manager.get_op_slices(final_op.op)[0]
    c3_group = manager.get_op_group(c3_slice)

    # Verify all OpSlice grouped with c1 have size 1.
    for op_slice in c1_group.op_slices:
      self.assertEqual(1, op_slice.slice.size)

    # Verify all OpSlice grouped with c2 have size 10.
    for op_slice in c2_group.op_slices:
      self.assertEqual(10, op_slice.slice.size)

    # Verify c1 is not grouped with c2, add, or c3.
    self.assertNotEqual(c1_group, c2_group)
    self.assertNotEqual(c1_group, add_group)
    self.assertNotEqual(c1_group, c3_group)

    # Verify c2 and add are grouped to each other, but not c3.
    self.assertEqual(c2_group, add_group)
    self.assertNotEqual(c2_group, c3_group)

  def testReuse(self):
    inputs = tf.zeros([2, 4, 4, 3])
    num_outputs = 3
    kernel_size = 1
    with arg_scope([layers.conv2d], normalizer_fn=tf.contrib.layers.batch_norm):
      with tf.variable_scope('parallel', reuse=tf.AUTO_REUSE):
        mul0 = layers.conv2d(inputs, num_outputs, kernel_size, scope='conv1')
        mul1 = layers.conv2d(inputs, num_outputs, kernel_size,
                             activation_fn=tf.nn.sigmoid, scope='conv2')
        prev1 = np.prod([mul0, mul1])
      with tf.variable_scope('parallel', reuse=tf.AUTO_REUSE):
        mul0_1 = layers.conv2d(prev1, num_outputs, kernel_size, scope='conv1')
        mul1_1 = layers.conv2d(prev1, num_outputs, kernel_size,
                               activation_fn=tf.nn.sigmoid, scope='conv2')
      prev2 = np.prod([mul0_1, mul1_1])
      prev3 = prev2 + 0.0
      # This hack produces the desired grouping due to variable reuse.
      # prev3 = prev2 + 0.0 * (mul0 + mul1 + mul0_1 + mul1_1)

    manager = orm.OpRegularizerManager(
        [prev3.op], self._default_op_handler_dict)

    mul0_reg = manager.get_regularizer(_get_op('parallel/conv1/Conv2D'))
    mul1_reg = manager.get_regularizer(_get_op('parallel/conv2/Conv2D'))
    mul0_1_reg = manager.get_regularizer(_get_op('parallel_1/conv1/Conv2D'))
    mul1_1_reg = manager.get_regularizer(_get_op('parallel_1/conv2/Conv2D'))

    # Check that regularizers were grouped properly.
    self.assertEqual(mul0_reg, mul1_reg)
    self.assertEqual(mul0_1_reg, mul1_1_reg)
    # These regularizers should be grouped due to reused variables.
    # self.assertEqual(mul0_reg, mul0_1_reg)
    # self.assertEqual(mul1_reg, mul1_1_reg)

  def testGather(self):
    gather_index = [5, 6, 7, 8, 9, 0, 1, 2, 3, 4]
    with tf.contrib.framework.arg_scope(self._batch_norm_scope()):
      inputs = tf.zeros([2, 4, 4, 3])
      c1 = layers.conv2d(inputs, num_outputs=10, kernel_size=3, scope='conv1')
      gather = tf.gather(c1, gather_index, axis=3)

    manager = orm.OpRegularizerManager(
        [gather.op], self._default_op_handler_dict)

    c1_reg = manager.get_regularizer(_get_op('conv1/Conv2D'))
    gather_reg = manager.get_regularizer(_get_op('GatherV2'))

    # Check regularizer indices.
    self.assertAllEqual(list(range(10)), c1_reg.regularization_vector)
    # This fails due to gather not being supported.  Once gather is supported,
    # this test can be enabled to verify that the regularization vector is
    # gathered in the same ordering as the tensor.
    # self.assertAllEqual(
    #     gather_index, gather_reg.regularization_vector)

    # This test shows that gather is not supported.  The regularization vector
    # has the same initial ordering after the gather op scrambled the
    # channels.  Remove this once gather is supported.
    self.assertAllEqual(list(range(10)), gather_reg.regularization_vector)

  def testConcat(self):
    with tf.contrib.framework.arg_scope(self._batch_norm_scope()):
      inputs = tf.zeros([2, 4, 4, 3])
      c1 = layers.conv2d(inputs, num_outputs=10, kernel_size=3, scope='conv1')
      c2 = layers.conv2d(inputs, num_outputs=10, kernel_size=3, scope='conv2')
      concat = tf.concat([c1, c2], axis=3)
      tmp = c1 + c2

    manager = orm.OpRegularizerManager(
        [concat.op, tmp.op], self._default_op_handler_dict)

    # Fetch OpSlice to verify grouping.
    inputs_op_slice = manager.get_op_slices(inputs.op)[0]
    c1_op_slice = manager.get_op_slices(c1.op)[0]
    c2_op_slice = manager.get_op_slices(c2.op)[0]
    tmp_op_slice = manager.get_op_slices(tmp.op)[0]
    concat_op_slice0 = manager.get_op_slices(concat.op)[0]
    concat_op_slice1 = manager.get_op_slices(concat.op)[1]

    # Verify inputs and c1 have different group.
    self.assertNotEqual(manager.get_op_group(inputs_op_slice),
                        manager.get_op_group(c1_op_slice))

    # Verify inputs and c2 have different group.
    self.assertNotEqual(manager.get_op_group(inputs_op_slice),
                        manager.get_op_group(c2_op_slice))

    # Verify c1, c2, and add have the same group.
    self.assertEqual(manager.get_op_group(c1_op_slice),
                     manager.get_op_group(c2_op_slice))
    self.assertEqual(manager.get_op_group(c1_op_slice),
                     manager.get_op_group(tmp_op_slice))

    # Verify concat slices are grouped with c1, c2, and add.
    self.assertEqual(manager.get_op_group(c1_op_slice),
                     manager.get_op_group(concat_op_slice0))
    self.assertEqual(manager.get_op_group(c1_op_slice),
                     manager.get_op_group(concat_op_slice1))

  def testGroupingConcat(self):
    with tf.contrib.framework.arg_scope(self._batch_norm_scope()):
      inputs = tf.zeros([2, 4, 4, 3])
      c1 = layers.conv2d(inputs, num_outputs=5, kernel_size=3, scope='conv1')
      c2 = layers.conv2d(inputs, num_outputs=5, kernel_size=3, scope='conv2')
      concat = tf.concat([c1, c2], axis=2)

    manager = orm.OpRegularizerManager([concat.op],
                                       self._default_op_handler_dict)

    # Fetch OpSlice to verify grouping.
    inputs_op_slice = manager.get_op_slices(inputs.op)[0]
    c1_op_slice = manager.get_op_slices(c1.op)[0]
    c2_op_slice = manager.get_op_slices(c2.op)[0]
    concat_op_slice = manager.get_op_slices(concat.op)[0]

    # Verify inputs and c1 have different group.
    self.assertNotEqual(
        manager.get_op_group(inputs_op_slice),
        manager.get_op_group(c1_op_slice))

    # Verify inputs and c2 have different group.
    self.assertNotEqual(
        manager.get_op_group(inputs_op_slice),
        manager.get_op_group(c2_op_slice))

    # Verify c1, c2, and concat have the same group.
    self.assertEqual(
        manager.get_op_group(c1_op_slice), manager.get_op_group(c2_op_slice))
    self.assertEqual(
        manager.get_op_group(c1_op_slice),
        manager.get_op_group(concat_op_slice))

  def testBatchNormAfterConcat(self):
    inputs = tf.zeros([2, 4, 4, 3])
    # BN before concat - one per conv.
    with arg_scope(
        [layers.conv2d],
        normalizer_fn=layers.batch_norm,
        normalizer_params={'fused': True, 'scale': True}):
      left = layers.conv2d(inputs, 2, kernel_size=3, scope='left')
      right = layers.conv2d(inputs, 3, kernel_size=3, scope='right')
      concat = tf.concat([left, right], -1)

    manager = orm.OpRegularizerManager(
        [concat.op], self._default_op_handler_dict)

    # Fetch OpSlice to verify grouping.
    left_op_slice = manager.get_op_slices(left.op)[0]
    right_op_slice = manager.get_op_slices(right.op)[0]
    concat_op_slice0 = manager.get_op_slices(concat.op)[0]
    concat_op_slice1 = manager.get_op_slices(concat.op)[1]

    # Verify that left op is grouped with left part of concat.
    self.assertEqual(manager.get_op_group(left_op_slice),
                     manager.get_op_group(concat_op_slice0))

    # Verify that right op is grouped with right part of concat.
    self.assertEqual(manager.get_op_group(right_op_slice),
                     manager.get_op_group(concat_op_slice1))

    # BN after concat
    tf.reset_default_graph()
    inputs = tf.zeros([2, 4, 4, 3])
    left = layers.conv2d(inputs, 3, kernel_size=3, scope='left_after')
    right = layers.conv2d(inputs, 4, kernel_size=3, scope='right_after')
    concat = tf.concat([left, right], -1)
    batch_norm = layers.batch_norm(concat, fused=True, scale=True)

    manager = orm.OpRegularizerManager(
        [batch_norm.op], self._default_op_handler_dict)

    # Fetch OpSlice to verify grouping.
    left_op_slice = manager.get_op_slices(left.op)[0]
    right_op_slice = manager.get_op_slices(right.op)[0]
    concat_op_slice0 = manager.get_op_slices(concat.op)[0]
    concat_op_slice1 = manager.get_op_slices(concat.op)[1]
    batch_norm_op_slice0 = manager.get_op_slices(batch_norm.op)[0]
    batch_norm_op_slice1 = manager.get_op_slices(batch_norm.op)[1]

    # Verify that left op is grouped with left part of concat and batch norm.
    self.assertEqual(manager.get_op_group(left_op_slice),
                     manager.get_op_group(concat_op_slice0))
    self.assertEqual(manager.get_op_group(left_op_slice),
                     manager.get_op_group(batch_norm_op_slice0))

    # Verify that right op is grouped with right part of concat and batch norm.
    self.assertEqual(manager.get_op_group(right_op_slice),
                     manager.get_op_group(concat_op_slice1))
    self.assertEqual(manager.get_op_group(right_op_slice),
                     manager.get_op_group(batch_norm_op_slice1))

    # Verify that original concat OpSlice is removed.
    old_concat_op_slice = orm.OpSlice(concat.op, orm.Slice(0, 7))
    self.assertIsNone(manager.get_op_group(old_concat_op_slice))

  def testNestedConcat(self):
    inputs = tf.zeros([2, 4, 4, 3])
    conv1 = layers.conv2d(inputs, num_outputs=1, kernel_size=3, scope='conv1')
    conv2 = layers.conv2d(inputs, num_outputs=1, kernel_size=3, scope='conv2')
    conv3 = layers.conv2d(inputs, num_outputs=1, kernel_size=3, scope='conv3')
    conv4 = layers.conv2d(inputs, num_outputs=1, kernel_size=3, scope='conv4')
    conv5 = layers.conv2d(inputs, num_outputs=1, kernel_size=3, scope='conv5')
    conv6 = layers.conv2d(inputs, num_outputs=1, kernel_size=3, scope='conv6')
    conv7 = layers.conv2d(inputs, num_outputs=1, kernel_size=3, scope='conv7')
    conv8 = layers.conv2d(inputs, num_outputs=1, kernel_size=3, scope='conv8')
    conv9 = layers.conv2d(inputs, num_outputs=1, kernel_size=3, scope='conv9')
    conv10 = layers.conv2d(inputs, num_outputs=1, kernel_size=3, scope='conv10')

    concat1 = tf.concat([conv1, conv2, conv3], axis=3)
    concat2 = tf.concat([conv4, conv5, conv6, conv7], axis=3)
    concat3 = tf.concat([conv8, conv9], axis=3)
    concat4 = tf.concat([conv10], axis=3)

    concat5 = tf.concat([concat1, concat2], axis=3)
    concat6 = tf.concat([concat3, concat4], axis=3)

    # This looks like [[[1, 2, 3], [4, 5, 6, 7]], [[8, 9], [10]]].
    concat7 = tf.concat([concat5, concat6], axis=3)
    batch_norm = layers.batch_norm(concat7)

    manager = orm.OpRegularizerManager(
        [batch_norm.op], self._default_op_handler_dict)

    # Verify that batch norm gets sliced into individual channels due to
    # concatenation of all the convolutions.
    batch_norm_op_slices = manager.get_op_slices(batch_norm.op)
    self.assertLen(batch_norm_op_slices, 10)
    for i in range(10):
      op_slice = batch_norm_op_slices[i]
      self.assertEqual(i, op_slice.slice.start_index)
      self.assertEqual(1, op_slice.slice.size)

      # Verify other OpSlice are not grouped with this one.
      group_op_slices = manager.get_op_group(op_slice).op_slices
      for j in range(10):
        if i != j:
          self.assertNotIn(batch_norm_op_slices[j], group_op_slices)

  def testSplit(self):
    with tf.contrib.framework.arg_scope(self._batch_norm_scope()):
      inputs = tf.zeros([2, 4, 4, 3])
      c1 = layers.conv2d(inputs, num_outputs=10, kernel_size=3, scope='conv1')
      split = tf.split(c1, [5, 5], axis=3)
      c2 = layers.conv2d(inputs, num_outputs=5, kernel_size=3, scope='conv2')
      c3 = layers.conv2d(inputs, num_outputs=5, kernel_size=3, scope='conv3')
      out1 = split[0] + c2
      out2 = split[1] + c3

    with self.assertRaises(RuntimeError):
      # Regularizer assignment fails because c2/c3 have size 5 while split has
      # size 10, so regularizer grouping fails.
      orm.OpRegularizerManager(
          [out1.op, out2.op], self._default_op_handler_dict,
          iteration_limit=100)

  @parameterized.named_parameters(('DepthMultiplier_1', 8, 1),
                                  ('DepthMultiplier_2', 8, 2),
                                  ('DepthMultiplier_7', 8, 7),
                                  ('DepthMultiplier_1_no_pointwise', None, 1),
                                  ('DepthMultiplier_2_no_pointwise', None, 2),
                                  ('DepthMultiplier_7_no_pointwise', None, 7))
  def testSeparableConv2D_DepthMultiplier(
      self, pointwise_outputs, depth_multiplier):
    with tf.contrib.framework.arg_scope(self._batch_norm_scope()):
      inputs = tf.zeros([2, 4, 4, 3])
      num_outputs = 5
      c1 = layers.conv2d(
          inputs, num_outputs=num_outputs, kernel_size=3, scope='conv1')
      c2 = layers.separable_conv2d(
          c1, num_outputs=pointwise_outputs, kernel_size=3,
          depth_multiplier=depth_multiplier, scope='conv2')
      identity = tf.identity(c2)

    manager = orm.OpRegularizerManager(
        [identity.op], self._default_op_handler_dict)

    # If separable_conv2d is passed num_outputs=None, the name of the depthwise
    # convolution changes.
    depthwise_conv_name = 'conv2/separable_conv2d/depthwise'
    if pointwise_outputs is None:
      depthwise_conv_name = 'conv2/depthwise'

    dwise_op = _get_op(depthwise_conv_name)
    dwise_reg = manager.get_regularizer(dwise_op)

    # Verify that depthwise convolution has output tensor and regularization
    # vector of size 5 * depth_multiplier channels where 5 is the number of
    # input channels from c1.
    self.assertEqual(num_outputs * depth_multiplier,
                     dwise_op.outputs[0].shape[-1])
    self.assertEqual(num_outputs * depth_multiplier,
                     dwise_reg.regularization_vector.shape[-1])

    # Verify OpSlice in the depthwise convolution has the correct grouping.
    relu1_op_slices = manager.get_op_slices(c1.op)
    dwise_op_slices = manager.get_op_slices(dwise_op)
    relu2_op_slices = manager.get_op_slices(c2.op)

    # Expected input grouping has a pattern like [0, 0, 1, 1, 2, 2, ...].
    expected_input_grouping = [j
                               for j in range(num_outputs)
                               for i in range(depth_multiplier)]
    # Expected output grouping is just linear, but with
    # num_outputs * depth_multiplier channels (e.g. [0, 1, 2, 3, ...]).
    expected_output_grouping = range(num_outputs * depth_multiplier)

    for i, op_slice in enumerate(dwise_op_slices):
      group = manager.get_op_group(op_slice)
      group_op_slices = group.op_slices
      self.assertIn(relu1_op_slices[expected_input_grouping[i]],
                    group_op_slices)
      self.assertIn(dwise_op_slices[expected_output_grouping[i]],
                    group_op_slices)
      if pointwise_outputs is None:
        # When pointwise_outputs is None, the pointwise convolution is omitted
        # and the depthwise convolution is immediately followed by
        # BiasAdd -> Relu ops.  In that case, verify that input channels of the
        # depthwise convolution are correctly grouped with the output (relu2)
        # channels.  Otherwise, the depthwise convolution is immediately
        # followed by a pointwise convolution which is non-passthrough, so there
        # is no output grouping to verify.
        self.assertIn(relu2_op_slices[expected_output_grouping[i]],
                      group_op_slices)

  def testAddN(self):
    inputs = tf.zeros([2, 4, 4, 3])
    identity1 = tf.identity(inputs)
    identity2 = tf.identity(inputs)
    identity3 = tf.identity(inputs)
    identity4 = tf.identity(inputs)
    add_n = tf.add_n([identity1, identity2, identity3, identity4])
    batch_norm = layers.batch_norm(add_n)

    manager = orm.OpRegularizerManager(
        [batch_norm.op], op_handler_dict=self._default_op_handler_dict)

    op_slices = manager.get_op_slices(identity1.op)
    self.assertLen(op_slices, 1)
    op_group = manager.get_op_group(op_slices[0]).op_slices

    # Verify all ops are in the same group.
    for test_op in (identity1.op, identity2.op, identity3.op, identity4.op,
                    add_n.op, batch_norm.op):
      test_op_slices = manager.get_op_slices(test_op)
      self.assertLen(test_op_slices, 1)
      self.assertIn(test_op_slices[0], op_group)

  def testAddN_Duplicates(self):
    inputs = tf.zeros([2, 4, 4, 3])
    identity = tf.identity(inputs)
    add_n = tf.add_n([identity, identity, identity, identity])
    batch_norm = layers.batch_norm(add_n)

    manager = orm.OpRegularizerManager(
        [batch_norm.op], op_handler_dict=self._default_op_handler_dict)

    op_slices = manager.get_op_slices(identity.op)
    self.assertLen(op_slices, 1)
    op_group = manager.get_op_group(op_slices[0]).op_slices

    # Verify all ops are in the same group.
    for test_op in (identity.op, add_n.op, batch_norm.op):
      test_op_slices = manager.get_op_slices(test_op)
      self.assertLen(test_op_slices, 1)
      self.assertIn(test_op_slices[0], op_group)

  def testInit_Add(self):
    with tf.contrib.framework.arg_scope(self._batch_norm_scope()):
      inputs = tf.zeros([2, 4, 4, 3])
      c1 = layers.conv2d(inputs, num_outputs=10, kernel_size=3, scope='conv1')
      c2 = layers.conv2d(inputs, num_outputs=10, kernel_size=3, scope='conv2')
      add = c1 + c2
      c3 = layers.conv2d(add, num_outputs=10, kernel_size=3, scope='conv3')
      out = tf.identity(c3)

    manager = orm.OpRegularizerManager(
        [out.op], self._default_op_handler_dict, SumGroupingRegularizer)

    # Fetch OpSlice to verify grouping.
    inputs_op_slice = manager.get_op_slices(inputs.op)[0]
    c1_op_slice = manager.get_op_slices(c1.op)[0]
    c2_op_slice = manager.get_op_slices(c2.op)[0]
    add_op_slice = manager.get_op_slices(add.op)[0]
    c3_op_slice = manager.get_op_slices(c3.op)[0]
    out_op_slice = manager.get_op_slices(out.op)[0]

    # Verify inputs and c1 have different group.
    self.assertNotEqual(manager.get_op_group(inputs_op_slice),
                        manager.get_op_group(c1_op_slice))
    self.assertNotEqual(manager.get_regularizer(inputs.op),
                        manager.get_regularizer(c1.op))

    # Verify inputs and c2 have different group.
    self.assertNotEqual(manager.get_op_group(inputs_op_slice),
                        manager.get_op_group(c2_op_slice))
    self.assertNotEqual(manager.get_regularizer(inputs.op),
                        manager.get_regularizer(c2.op))

    # Verify c1, c2, and add have the same group.
    self.assertEqual(manager.get_op_group(c1_op_slice),
                     manager.get_op_group(c2_op_slice))
    self.assertEqual(manager.get_op_group(c1_op_slice),
                     manager.get_op_group(add_op_slice))
    self.assertEqual(manager.get_regularizer(c1.op),
                     manager.get_regularizer(c2.op))
    self.assertEqual(manager.get_regularizer(c1.op),
                     manager.get_regularizer(add.op))

    # Verify c3 and out have the same group, which differs from c1 and c2.
    self.assertEqual(manager.get_op_group(c3_op_slice),
                     manager.get_op_group(out_op_slice))
    self.assertNotEqual(manager.get_op_group(c3_op_slice),
                        manager.get_op_group(c1_op_slice))
    self.assertEqual(manager.get_regularizer(c3.op),
                     manager.get_regularizer(out.op))
    self.assertNotEqual(manager.get_regularizer(c3.op),
                        manager.get_regularizer(c1.op))

  def testInit_Concat(self):
    with tf.contrib.framework.arg_scope(self._batch_norm_scope()):
      inputs = tf.zeros([2, 4, 4, 3])
      c1 = layers.conv2d(inputs, num_outputs=10, kernel_size=3, scope='conv1')
      c2 = layers.conv2d(inputs, num_outputs=10, kernel_size=3, scope='conv2')
      concat = tf.concat([c1, c2], axis=3)
      out = tf.identity(concat)

    manager = orm.OpRegularizerManager(
        [out.op], self._default_op_handler_dict, SumGroupingRegularizer)

    # Fetch OpSlice to verify grouping.
    inputs_op_slice = manager.get_op_slices(inputs.op)[0]
    c1_op_slice = manager.get_op_slices(c1.op)[0]
    c2_op_slice = manager.get_op_slices(c2.op)[0]
    out_op_slice0 = manager.get_op_slices(out.op)[0]
    out_op_slice1 = manager.get_op_slices(out.op)[1]

    # Verify inputs and c1 have different group and OpRegularizer.
    self.assertNotEqual(manager.get_op_group(inputs_op_slice),
                        manager.get_op_group(c1_op_slice))
    self.assertNotEqual(manager.get_regularizer(inputs.op),
                        manager.get_regularizer(c1.op))

    # Verify inputs and c2 have different group and OpRegularizer.
    self.assertNotEqual(manager.get_op_group(inputs_op_slice),
                        manager.get_op_group(c2_op_slice))
    self.assertNotEqual(manager.get_regularizer(inputs.op),
                        manager.get_regularizer(c2.op))

    # Verify c1 and c2 have different group and OpRegularizer.
    self.assertNotEqual(manager.get_op_group(c1_op_slice),
                        manager.get_op_group(c2_op_slice))
    self.assertNotEqual(manager.get_regularizer(c1.op),
                        manager.get_regularizer(c2.op))

    # Verify c1 is grouped with first slice of out.
    self.assertEqual(manager.get_op_group(c1_op_slice),
                     manager.get_op_group(out_op_slice0))

    # Verify c2 is grouped with second slice of out.
    self.assertEqual(manager.get_op_group(c2_op_slice),
                     manager.get_op_group(out_op_slice1))

    # Verify out regularization_vector is the concat of c1 and c2
    # regularizertion_vector.
    self.assertAllEqual(
        manager.get_regularizer(c1.op).regularization_vector,
        manager.get_regularizer(out.op).regularization_vector[0:10])
    self.assertAllEqual(
        manager.get_regularizer(c2.op).regularization_vector,
        manager.get_regularizer(out.op).regularization_vector[10:20])

  def testInit_AddConcat(self):
    with tf.contrib.framework.arg_scope(self._batch_norm_scope()):
      inputs = tf.zeros([2, 4, 4, 3])
      c1 = layers.conv2d(inputs, num_outputs=10, kernel_size=3, scope='conv1')
      c2 = layers.conv2d(inputs, num_outputs=10, kernel_size=3, scope='conv2')
      add = c1 + c2
      c3 = layers.conv2d(add, num_outputs=10, kernel_size=3, scope='conv3')
      out = tf.identity(c3)
      concat = tf.concat([c1, c2], axis=3)
      c4 = layers.conv2d(concat, num_outputs=10, kernel_size=3, scope='conv4')

    manager = orm.OpRegularizerManager(
        [out.op, c4.op], self._default_op_handler_dict, SumGroupingRegularizer)

    # Fetch OpSlice to verify grouping.
    inputs_op_slice = manager.get_op_slices(inputs.op)[0]
    c1_op_slice = manager.get_op_slices(c1.op)[0]
    c2_op_slice = manager.get_op_slices(c2.op)[0]
    add_op_slice = manager.get_op_slices(add.op)[0]
    c3_op_slice = manager.get_op_slices(c3.op)[0]
    out_op_slice = manager.get_op_slices(out.op)[0]
    concat_op_slice0 = manager.get_op_slices(concat.op)[0]
    concat_op_slice1 = manager.get_op_slices(concat.op)[1]
    c4_op_slice = manager.get_op_slices(c4.op)[0]

    # Verify inputs and c1 have different group.
    self.assertNotEqual(manager.get_op_group(inputs_op_slice),
                        manager.get_op_group(c1_op_slice))
    self.assertNotEqual(manager.get_regularizer(inputs.op),
                        manager.get_regularizer(c1.op))

    # Verify inputs and c2 have different group.
    self.assertNotEqual(manager.get_op_group(inputs_op_slice),
                        manager.get_op_group(c2_op_slice))
    self.assertNotEqual(manager.get_regularizer(inputs.op),
                        manager.get_regularizer(c2.op))

    # Verify c1, c2, and add have the same group.
    self.assertEqual(manager.get_op_group(c1_op_slice),
                     manager.get_op_group(c2_op_slice))
    self.assertEqual(manager.get_op_group(c1_op_slice),
                     manager.get_op_group(add_op_slice))
    self.assertEqual(manager.get_regularizer(c1.op),
                     manager.get_regularizer(c2.op))
    self.assertEqual(manager.get_regularizer(c1.op),
                     manager.get_regularizer(add.op))

    # Verify c3 and out have the same group, which differs from c1 and c2.
    self.assertEqual(manager.get_op_group(c3_op_slice),
                     manager.get_op_group(out_op_slice))
    self.assertNotEqual(manager.get_op_group(c3_op_slice),
                        manager.get_op_group(c1_op_slice))
    self.assertEqual(manager.get_regularizer(c3.op),
                     manager.get_regularizer(out.op))
    self.assertNotEqual(manager.get_regularizer(c3.op),
                        manager.get_regularizer(c1.op))

    # Verify concat slices are grouped with c1, c2, and add.
    self.assertEqual(manager.get_op_group(c1_op_slice),
                     manager.get_op_group(concat_op_slice0))
    self.assertEqual(manager.get_op_group(c1_op_slice),
                     manager.get_op_group(concat_op_slice1))

    # Verify concat regularization_vector is the concat of c1 and c2
    # regularizertion_vector.
    self.assertAllEqual(
        manager.get_regularizer(c1.op).regularization_vector,
        manager.get_regularizer(concat.op).regularization_vector[0:10])
    self.assertAllEqual(
        manager.get_regularizer(c2.op).regularization_vector,
        manager.get_regularizer(concat.op).regularization_vector[10:20])

    # Verify c4 has a different group than c1, c2, and add.
    self.assertNotEqual(manager.get_op_group(c1_op_slice),
                        manager.get_op_group(c4_op_slice))
    self.assertNotEqual(manager.get_regularizer(c1.op),
                        manager.get_regularizer(c4.op))

  def testInit_AddConcat_AllOps(self):
    with tf.contrib.framework.arg_scope(self._batch_norm_scope()):
      inputs = tf.zeros([2, 4, 4, 3])
      c1 = layers.conv2d(inputs, num_outputs=10, kernel_size=3, scope='conv1')
      c2 = layers.conv2d(inputs, num_outputs=10, kernel_size=3, scope='conv2')
      add = c1 + c2
      c3 = layers.conv2d(add, num_outputs=10, kernel_size=3, scope='conv3')
      out = tf.identity(c3)
      concat = tf.concat([c1, c2], axis=3)
      c4 = layers.conv2d(concat, num_outputs=10, kernel_size=3, scope='conv4')

    manager = orm.OpRegularizerManager(
        [out.op], self._default_op_handler_dict, SumGroupingRegularizer)

    # Op c4 is not in the DFS path of out.  Verify that OpRegularizerManager
    # does not process c4.
    self.assertNotIn(c4.op, manager.ops)
    self.assertNotIn(concat.op, manager.ops)

  def testInit_ForceGroup(self):
    with tf.contrib.framework.arg_scope(self._batch_norm_scope()):
      inputs = tf.zeros([2, 4, 4, 3])
      c1 = layers.conv2d(inputs, num_outputs=10, kernel_size=3, scope='conv1')
      c2 = layers.conv2d(c1, num_outputs=10, kernel_size=3, scope='conv2')
      c3 = layers.conv2d(c2, num_outputs=10, kernel_size=3, scope='conv3')

    # Initialize OpRegularizerManager with no force-grouping.
    manager = orm.OpRegularizerManager(
        [c3.op], self._default_op_handler_dict, SumGroupingRegularizer)

    # Verify that c2 is not grouped with c3.
    c2_op_slices = manager.get_op_slices(c2.op)
    self.assertLen(c2_op_slices, 1)
    c2_op_slice = c2_op_slices[0]
    c2_group = manager.get_op_group(c2_op_slice)
    c3_op_slices = manager.get_op_slices(c3.op)
    self.assertLen(c3_op_slices, 1)
    c3_op_slice = c3_op_slices[0]
    self.assertNotIn(c3_op_slice, c2_group.op_slices)

    # Force-group c2 and c3.
    manager = orm.OpRegularizerManager(
        [c3.op], self._default_op_handler_dict, SumGroupingRegularizer,
        force_group=['conv2|conv3'])

    # Verify that c2 is grouped with c3.
    c2_op_slices = manager.get_op_slices(c2.op)
    self.assertLen(c2_op_slices, 1)
    c2_op_slice = c2_op_slices[0]
    c2_group = manager.get_op_group(c2_op_slice)
    c3_op_slices = manager.get_op_slices(c3.op)
    self.assertLen(c3_op_slices, 1)
    c3_op_slice = c3_op_slices[0]
    self.assertIn(c3_op_slice, c2_group.op_slices)

  def testInit_ForceGroup_MultipleOpSlice(self):
    with tf.contrib.framework.arg_scope(self._batch_norm_scope()):
      inputs = tf.zeros([2, 4, 4, 3])
      c1 = layers.conv2d(inputs, num_outputs=5, kernel_size=3, scope='conv1')
      c2 = layers.conv2d(inputs, num_outputs=5, kernel_size=3, scope='conv2')
      concat = tf.concat([c1, c2], axis=3)
      c3 = layers.conv2d(concat, num_outputs=10, kernel_size=3, scope='conv3')

    # Verify force-group with multiple OpSlice raises ValueError.
    self.assertRaisesRegexp(
        ValueError,
        r'Cannot force-group ops with more than 1 OpSlice: \[u?\'concat\'\]',
        orm.OpRegularizerManager, [c3.op], self._default_op_handler_dict,
        SumGroupingRegularizer, force_group=['conv3|concat'])

  def testInit_ForceGroup_SizeMismatch(self):
    with tf.contrib.framework.arg_scope(self._batch_norm_scope()):
      inputs = tf.zeros([2, 4, 4, 3])
      c1 = layers.conv2d(inputs, num_outputs=10, kernel_size=3, scope='conv1')
      c2 = layers.conv2d(c1, num_outputs=10, kernel_size=3, scope='conv2')
      # c3 has size 9 instead of 10.
      c3 = layers.conv2d(c2, num_outputs=9, kernel_size=3, scope='conv3')

    # Verify size mismatch raises ValueError.
    self.assertRaisesRegexp(
        ValueError,
        r'Cannot force-group ops with different sizes: \[.*\]',
        orm.OpRegularizerManager, [c3.op], self._default_op_handler_dict,
        SumGroupingRegularizer, force_group=['conv2|conv3'])

  def testInit_ForceGroup_NotList(self):
    inputs = tf.zeros([2, 4, 4, 3])

    # Verify that force_group string instead of a list raises exception.
    self.assertRaisesRegexp(
        TypeError,
        r'force_group must be a list of regex.',
        orm.OpRegularizerManager, [inputs.op], self._default_op_handler_dict,
        SumGroupingRegularizer, force_group='conv')

  def testInit_Blacklist(self):
    with tf.contrib.framework.arg_scope(self._batch_norm_scope()):
      inputs = tf.zeros([2, 4, 4, 3])
      c1 = layers.conv2d(inputs, num_outputs=3, kernel_size=3, scope='conv1')
      c2 = layers.conv2d(c1, num_outputs=4, kernel_size=3, scope='conv2')
      c3 = layers.conv2d(c2, num_outputs=5, kernel_size=3, scope='conv3')

    # Verify c2 has a regularizer.
    manager = orm.OpRegularizerManager(
        [c3.op], self._default_op_handler_dict, SumGroupingRegularizer)
    self.assertIsNotNone(manager.get_regularizer(c2.op))

    # Verify c2 has None regularizer after blacklisting.
    manager = orm.OpRegularizerManager(
        [c3.op], self._default_op_handler_dict, SumGroupingRegularizer,
        regularizer_blacklist=['conv2'])
    self.assertIsNone(manager.get_regularizer(c2.op))

  def testInit_BlacklistGroup(self):
    with tf.contrib.framework.arg_scope(self._batch_norm_scope()):
      inputs = tf.zeros([2, 4, 4, 3])
      c1 = layers.conv2d(inputs, num_outputs=10, kernel_size=3, scope='conv1')
      c2 = layers.conv2d(inputs, num_outputs=10, kernel_size=3, scope='conv2')
      add = c1 + c2
      c3 = layers.conv2d(add, num_outputs=10, kernel_size=3, scope='conv3')

    # Verify c2 has a regularizer.
    manager = orm.OpRegularizerManager(
        [c3.op], self._default_op_handler_dict, SumGroupingRegularizer)
    self.assertIsNotNone(manager.get_regularizer(c2.op))

    # Verify c2 has None regularizer after blacklisting c1 which is grouped.
    manager = orm.OpRegularizerManager(
        [c3.op], self._default_op_handler_dict, SumGroupingRegularizer,
        regularizer_blacklist=['conv1'])
    self.assertIsNone(manager.get_regularizer(c2.op))

  def testInit_BlacklistGroup_NoMatch(self):
    with tf.contrib.framework.arg_scope(self._batch_norm_scope()):
      inputs = tf.zeros([2, 4, 4, 3])
      c1 = layers.conv2d(inputs, num_outputs=10, kernel_size=3, scope='conv1')
      c2 = layers.conv2d(inputs, num_outputs=10, kernel_size=3, scope='conv2')
      add = c1 + c2
      c3 = layers.conv2d(add, num_outputs=10, kernel_size=3, scope='conv3')

    # Verify blacklist regex without match raises ValueError
    self.assertRaisesWithLiteralMatch(
        ValueError,
        'Blacklist regex never used: \'oops\'.',
        orm.OpRegularizerManager, [c3.op], self._default_op_handler_dict,
        SumGroupingRegularizer, regularizer_blacklist=['oops'])

  def testInit_BlacklistGroup_NotList(self):
    inputs = tf.zeros([2, 4, 4, 3])

    # Verify that regularizer_blacklist string instead of a list raises
    # exception.
    self.assertRaisesRegexp(
        TypeError,
        r'regularizer_blacklist must be a list of regex.',
        orm.OpRegularizerManager, [inputs.op], self._default_op_handler_dict,
        SumGroupingRegularizer, regularizer_blacklist='conv')

  def testInit_IterationLimit(self):
    inputs = tf.zeros([2, 4, 4, 3])

    # Verify that reaching iteration limit raises exception.
    self.assertRaisesRegexp(
        RuntimeError,
        r'OpRegularizerManager could not handle ops:',
        orm.OpRegularizerManager, [inputs.op], self._default_op_handler_dict,
        SumGroupingRegularizer, iteration_limit=0)

  def testGetRegularizer(self):
    op1 = tf.zeros([2, 4, 4, 3])
    op2 = tf.zeros([2, 4, 4, 3])
    op3 = tf.zeros([2, 4, 4, 10])

    manager = orm.OpRegularizerManager([], self._default_op_handler_dict)
    manager.slice_op(op3.op, [1, 2, 3, 4])

    # op2 has 1 OpSlice and op3 has 4 OpSlice of size [1, 2, 3, 4].
    op2_slices = manager.get_op_slices(op2.op)
    op3_slices = manager.get_op_slices(op3.op)

    op2_reg = IndexOpRegularizer(op2_slices[0], manager)
    op3_reg0 = IndexOpRegularizer(op3_slices[0], manager)
    op3_reg1 = IndexOpRegularizer(op3_slices[1], manager)
    op3_reg2 = IndexOpRegularizer(op3_slices[2], manager)
    op3_reg3 = IndexOpRegularizer(op3_slices[3], manager)

    # Map OpSlice to OpRegularizer.
    manager._op_regularizer_dict = {
        op2_slices[0]: op2_reg,
        op3_slices[0]: op3_reg0,
        op3_slices[1]: op3_reg1,
        op3_slices[2]: op3_reg2,
        op3_slices[3]: op3_reg3,
    }

    # Verify None is returned if OpSlice does not have OpRegularizer.
    self.assertIsNone(manager.get_regularizer(op1.op))

    # Verify OpRegularizer for op with single OpSlice.
    self.assertAllEqual([0, 1, 2],
                        manager.get_regularizer(op2.op).regularization_vector)

    # Verify OpRegularizer for op with multiple OpSlice.
    self.assertAllEqual(list(range(3, 13)),
                        manager.get_regularizer(op3.op).regularization_vector)

    # Verify OpRegularzier for op with multiple OpSlice but not all slices have
    # a regularizer.
    del manager._op_regularizer_dict[op3_slices[2]]
    expected_regularization_vector = [3, 4, 5, 0, 0, 0, 9, 10, 11, 12]
    self.assertAllEqual(expected_regularization_vector,
                        manager.get_regularizer(op3.op).regularization_vector)

  def testCreateOpGroupForOpSlice_Source(self):
    inputs = tf.zeros([2, 4, 4, 3])
    identity = tf.identity(inputs)

    manager = orm.OpRegularizerManager([])

    # Create OpSlice for each identity op.
    op_slice = manager.get_op_slices(identity.op)[0]

    # Create OpGroup for each OpSlice.
    op_group = manager.create_op_group_for_op_slice(op_slice)

    self.assertListEqual([op_slice], op_group.op_slices)
    self.assertListEqual([op_slice], op_group.source_op_slices)
    self.assertEqual(op_group, manager.get_op_group(op_slice))

  def testCreateOpGroupForOpSlice_NotSource(self):
    inputs = tf.zeros([2, 4, 4, 3])
    identity = tf.identity(inputs)

    manager = orm.OpRegularizerManager([])

    # Create OpSlice for each identity op.
    op_slice = manager.get_op_slices(identity.op)[0]

    # Create OpGroup for each OpSlice.
    op_group = manager.create_op_group_for_op_slice(op_slice, is_source=False)

    self.assertListEqual([op_slice], op_group.op_slices)
    self.assertListEqual([], op_group.source_op_slices)
    self.assertEqual(op_group, manager.get_op_group(op_slice))

  def testGroupOpSlices(self):
    inputs = tf.zeros([2, 4, 4, 3])
    identity1 = tf.identity(inputs)
    identity2 = tf.identity(inputs)
    identity3 = tf.identity(inputs)
    identity4 = tf.identity(inputs)
    identity5 = tf.identity(inputs)
    identity6 = tf.identity(inputs)
    identity7 = tf.identity(inputs)
    identity8 = tf.identity(inputs)

    manager = orm.OpRegularizerManager([])

    # Create OpSlice for each identity op.
    op_slice1 = manager.get_op_slices(identity1.op)[0]
    op_slice2 = manager.get_op_slices(identity2.op)[0]
    op_slice3 = manager.get_op_slices(identity3.op)[0]
    op_slice4 = manager.get_op_slices(identity4.op)[0]
    op_slice5 = manager.get_op_slices(identity5.op)[0]
    op_slice6 = manager.get_op_slices(identity6.op)[0]
    op_slice7 = manager.get_op_slices(identity7.op)[0]
    op_slice8 = manager.get_op_slices(identity8.op)[0]

    # Create OpGroup for each OpSlice.
    op_group1 = manager.create_op_group_for_op_slice(op_slice1)
    op_group2 = manager.create_op_group_for_op_slice(op_slice2)
    op_group3 = manager.create_op_group_for_op_slice(op_slice3)
    op_group4 = manager.create_op_group_for_op_slice(op_slice4)
    op_group5 = manager.create_op_group_for_op_slice(op_slice5)
    op_group6 = manager.create_op_group_for_op_slice(op_slice6)
    op_group7 = manager.create_op_group_for_op_slice(op_slice7)
    op_group8 = manager.create_op_group_for_op_slice(op_slice8)

    # Group all OpGroup together by grouping their OpSlice.
    manager.group_op_slices([op_slice1, op_slice2, op_slice3, op_slice4,
                             op_slice5, op_slice6, op_slice7, op_slice8])

    expected_group = orm.OpGroup(
        op_groups=[op_group1, op_group2, op_group3, op_group4, op_group5,
                   op_group6, op_group7, op_group8])

    # Check all OpSlice are in one big group.
    self.assertListEqual(
        expected_group.op_slices,
        manager.get_op_group(op_slice1).op_slices)
    self.assertListEqual(
        expected_group.op_slices,
        manager.get_op_group(op_slice2).op_slices)
    self.assertListEqual(
        expected_group.op_slices,
        manager.get_op_group(op_slice3).op_slices)
    self.assertListEqual(
        expected_group.op_slices,
        manager.get_op_group(op_slice4).op_slices)
    self.assertListEqual(
        expected_group.op_slices,
        manager.get_op_group(op_slice5).op_slices)
    self.assertListEqual(
        expected_group.op_slices,
        manager.get_op_group(op_slice6).op_slices)
    self.assertListEqual(
        expected_group.op_slices,
        manager.get_op_group(op_slice7).op_slices)
    self.assertListEqual(
        expected_group.op_slices,
        manager.get_op_group(op_slice8).op_slices)

  def testGroupOpSlices_TransitiveGrouping(self):
    inputs = tf.zeros([2, 4, 4, 3])
    identity1 = tf.identity(inputs)
    identity2 = tf.identity(inputs)
    identity3 = tf.identity(inputs)
    identity4 = tf.identity(inputs)
    identity5 = tf.identity(inputs)
    identity6 = tf.identity(inputs)
    identity7 = tf.identity(inputs)
    identity8 = tf.identity(inputs)

    manager = orm.OpRegularizerManager([])

    # Create OpSlice for each identity op.
    op_slice1 = manager.get_op_slices(identity1.op)[0]
    op_slice2 = manager.get_op_slices(identity2.op)[0]
    op_slice3 = manager.get_op_slices(identity3.op)[0]
    op_slice4 = manager.get_op_slices(identity4.op)[0]
    op_slice5 = manager.get_op_slices(identity5.op)[0]
    op_slice6 = manager.get_op_slices(identity6.op)[0]
    op_slice7 = manager.get_op_slices(identity7.op)[0]
    op_slice8 = manager.get_op_slices(identity8.op)[0]

    # Create OpGroup for each OpSlice.
    op_group1 = manager.create_op_group_for_op_slice(op_slice1)
    op_group2 = manager.create_op_group_for_op_slice(op_slice2)
    op_group3 = manager.create_op_group_for_op_slice(op_slice3)
    op_group4 = manager.create_op_group_for_op_slice(op_slice4)
    op_group5 = manager.create_op_group_for_op_slice(op_slice5)
    op_group6 = manager.create_op_group_for_op_slice(op_slice6)
    op_group7 = manager.create_op_group_for_op_slice(op_slice7)
    op_group8 = manager.create_op_group_for_op_slice(op_slice8)

    # Group all OpGroup together by grouping their OpSlice.
    manager.group_op_slices([op_slice1, op_slice2, op_slice3, op_slice4])
    manager.group_op_slices([op_slice5, op_slice6, op_slice7, op_slice8])
    # Transitively create one large group by grouping one OpSlice from each
    # group.
    manager.group_op_slices([op_slice3, op_slice6])

    expected_group = orm.OpGroup(
        op_groups=[op_group1, op_group2, op_group3, op_group4, op_group5,
                   op_group6, op_group7, op_group8])

    # Check all OpSlice are in one big group.
    self.assertListEqual(
        expected_group.op_slices,
        manager.get_op_group(op_slice1).op_slices)
    self.assertListEqual(
        expected_group.op_slices,
        manager.get_op_group(op_slice2).op_slices)
    self.assertListEqual(
        expected_group.op_slices,
        manager.get_op_group(op_slice3).op_slices)
    self.assertListEqual(
        expected_group.op_slices,
        manager.get_op_group(op_slice4).op_slices)
    self.assertListEqual(
        expected_group.op_slices,
        manager.get_op_group(op_slice5).op_slices)
    self.assertListEqual(
        expected_group.op_slices,
        manager.get_op_group(op_slice6).op_slices)
    self.assertListEqual(
        expected_group.op_slices,
        manager.get_op_group(op_slice7).op_slices)
    self.assertListEqual(
        expected_group.op_slices,
        manager.get_op_group(op_slice8).op_slices)

  def testSliceOp_SingleSlice(self):
    inputs = tf.zeros([2, 4, 4, 3])
    identity1 = tf.identity(inputs)
    identity2 = tf.identity(inputs)
    identity3 = tf.identity(inputs)
    identity4 = tf.identity(inputs)
    identity5 = tf.identity(inputs)
    identity6 = tf.identity(inputs)
    identity7 = tf.identity(inputs)
    identity8 = tf.identity(inputs)

    manager = orm.OpRegularizerManager([], self._default_op_handler_dict)

    # Create OpSlice for each identity op.
    op_slice1 = manager.get_op_slices(identity1.op)[0]
    op_slice2 = manager.get_op_slices(identity2.op)[0]
    op_slice3 = manager.get_op_slices(identity3.op)[0]
    op_slice4 = manager.get_op_slices(identity4.op)[0]
    op_slice5 = manager.get_op_slices(identity5.op)[0]
    op_slice6 = manager.get_op_slices(identity6.op)[0]
    op_slice7 = manager.get_op_slices(identity7.op)[0]
    op_slice8 = manager.get_op_slices(identity8.op)[0]

    # Create OpGroup for each OpSlice.
    manager.create_op_group_for_op_slice(op_slice1)
    manager.create_op_group_for_op_slice(op_slice2)
    manager.create_op_group_for_op_slice(op_slice3)
    manager.create_op_group_for_op_slice(op_slice4)
    manager.create_op_group_for_op_slice(op_slice5)
    manager.create_op_group_for_op_slice(op_slice6)
    manager.create_op_group_for_op_slice(op_slice7)
    manager.create_op_group_for_op_slice(op_slice8)

    # Group all OpGroup together by grouping their OpSlice.
    manager.group_op_slices([op_slice1, op_slice2, op_slice3, op_slice4])
    manager.group_op_slices([op_slice5, op_slice6, op_slice7, op_slice8])

    # Only slice identity1 op.  This will also slice identity2, identity3, and
    # identity4 because the slices are grouped.  The ops identity5, identity6,
    # identity7, and identity8 are unaffected.
    manager.slice_op(identity1.op, [1, 2])

    # Verify ops grouped with identity1 are sliced, while other ops are not.
    self.assertLen(manager.get_op_slices(identity1.op), 2)
    self.assertLen(manager.get_op_slices(identity2.op), 2)
    self.assertLen(manager.get_op_slices(identity3.op), 2)
    self.assertLen(manager.get_op_slices(identity4.op), 2)
    self.assertLen(manager.get_op_slices(identity5.op), 1)
    self.assertLen(manager.get_op_slices(identity6.op), 1)
    self.assertLen(manager.get_op_slices(identity7.op), 1)
    self.assertLen(manager.get_op_slices(identity8.op), 1)

    # Verify sliced ops have sizes [1, 2].
    for op in (identity1.op, identity2.op, identity3.op, identity4.op):
      op_slices = manager.get_op_slices(op)
      self.assertEqual(0, op_slices[0].slice.start_index)
      self.assertEqual(1, op_slices[0].slice.size)
      self.assertEqual(1, op_slices[1].slice.start_index)
      self.assertEqual(2, op_slices[1].slice.size)

  def testSliceOp_SingleSlice_Ungrouped(self):
    inputs = tf.zeros([2, 4, 4, 3])
    identity1 = tf.identity(inputs)

    manager = orm.OpRegularizerManager([], self._default_op_handler_dict)

    # Only slice identity1 op which is ungrouped.
    manager.slice_op(identity1.op, [1, 2])

    # Verify identity1 op is sliced.
    self.assertLen(manager.get_op_slices(identity1.op), 2)

    # Verify sliced op has size [1, 2].
    op_slices = manager.get_op_slices(identity1.op)
    self.assertEqual(0, op_slices[0].slice.start_index)
    self.assertEqual(1, op_slices[0].slice.size)
    self.assertEqual(1, op_slices[1].slice.start_index)
    self.assertEqual(2, op_slices[1].slice.size)

  def testSliceOp_MultipleSlices(self):
    inputs = tf.zeros([2, 4, 4, 20])
    identity1 = tf.identity(inputs)
    identity2 = tf.identity(inputs)
    identity3 = tf.identity(inputs)

    manager = orm.OpRegularizerManager([], self._default_op_handler_dict)

    # First op has sizes [4, 3, 7, 6].
    op_slice1_0_4 = orm.OpSlice(identity1.op, orm.Slice(0, 4))
    op_slice1_4_7 = orm.OpSlice(identity1.op, orm.Slice(4, 3))
    op_slice1_7_14 = orm.OpSlice(identity1.op, orm.Slice(7, 7))
    op_slice1_14_20 = orm.OpSlice(identity1.op, orm.Slice(14, 6))

    # Second op has sizes [3, 7, 10].
    op_slice2_0_3 = orm.OpSlice(identity2.op, orm.Slice(0, 3))
    op_slice2_3_10 = orm.OpSlice(identity2.op, orm.Slice(3, 7))
    op_slice2_10_20 = orm.OpSlice(identity2.op, orm.Slice(10, 10))

    # Third op has sizes [2, 2, 2, 2, 3, 7, 2].
    op_slice3_0_2 = orm.OpSlice(identity3.op, orm.Slice(0, 2))
    op_slice3_2_4 = orm.OpSlice(identity3.op, orm.Slice(2, 2))
    op_slice3_4_6 = orm.OpSlice(identity3.op, orm.Slice(4, 2))
    op_slice3_6_8 = orm.OpSlice(identity3.op, orm.Slice(6, 2))
    op_slice3_8_11 = orm.OpSlice(identity3.op, orm.Slice(8, 3))
    op_slice3_11_18 = orm.OpSlice(identity3.op, orm.Slice(11, 7))
    op_slice3_18_20 = orm.OpSlice(identity3.op, orm.Slice(18, 2))

    manager._op_slice_dict = {
        identity1.op: [op_slice1_0_4, op_slice1_4_7, op_slice1_7_14,
                       op_slice1_14_20],
        identity2.op: [op_slice2_0_3, op_slice2_3_10, op_slice2_10_20],
        identity3.op: [op_slice3_0_2, op_slice3_2_4, op_slice3_4_6,
                       op_slice3_6_8, op_slice3_8_11, op_slice3_11_18,
                       op_slice3_18_20],
    }

    # Only the [3, 7] slices of the ops are grouped.  Only the first op is a
    # source.
    manager.group_op_slices(
        [op_slice1_4_7, op_slice2_0_3, op_slice3_8_11],
        omit_source_op_slices=[op_slice2_0_3, op_slice3_8_11])
    manager.group_op_slices(
        [op_slice1_7_14, op_slice2_3_10, op_slice3_11_18],
        omit_source_op_slices=[op_slice2_3_10, op_slice3_11_18])

    # Slice the [3, 7] grouped slices into [1, 2, 3, 4].
    manager.slice_op(identity1.op, [4, 1, 2, 3, 4, 6])

    # Verify grouped ops are sliced into the correct sizes.
    op_slices1 = manager.get_op_slices(identity1.op)
    op_slices2 = manager.get_op_slices(identity2.op)
    op_slices3 = manager.get_op_slices(identity3.op)

    expected_sizes1 = [4, 1, 2, 3, 4, 6]
    expected_sizes2 = [1, 2, 3, 4, 10]
    expected_sizes3 = [2, 2, 2, 2, 1, 2, 3, 4, 2]

    self.assertListEqual(
        expected_sizes1, [s.slice.size for s in op_slices1])
    self.assertListEqual(
        expected_sizes2, [s.slice.size for s in op_slices2])
    self.assertListEqual(
        expected_sizes3, [s.slice.size for s in op_slices3])

    # Verify new slices are grouped.
    op_slice1_4_5 = orm.OpSlice(identity1.op, orm.Slice(4, 1))
    op_slice1_5_7 = orm.OpSlice(identity1.op, orm.Slice(5, 2))
    op_slice1_7_10 = orm.OpSlice(identity1.op, orm.Slice(7, 3))
    op_slice1_10_14 = orm.OpSlice(identity1.op, orm.Slice(10, 4))

    op_slice2_0_1 = orm.OpSlice(identity2.op, orm.Slice(0, 1))
    op_slice2_1_3 = orm.OpSlice(identity2.op, orm.Slice(1, 2))
    op_slice2_3_6 = orm.OpSlice(identity2.op, orm.Slice(3, 3))
    op_slice2_6_10 = orm.OpSlice(identity2.op, orm.Slice(6, 4))

    op_slice3_8_9 = orm.OpSlice(identity3.op, orm.Slice(8, 1))
    op_slice3_9_11 = orm.OpSlice(identity3.op, orm.Slice(9, 2))
    op_slice3_11_14 = orm.OpSlice(identity3.op, orm.Slice(11, 3))
    op_slice3_14_18 = orm.OpSlice(identity3.op, orm.Slice(14, 4))

    expected_group1 = [op_slice1_4_5, op_slice2_0_1, op_slice3_8_9]
    expected_group2 = [op_slice1_5_7, op_slice2_1_3, op_slice3_9_11]
    expected_group3 = [op_slice1_7_10, op_slice2_3_6, op_slice3_11_14]
    expected_group4 = [op_slice1_10_14, op_slice2_6_10, op_slice3_14_18]

    self.assertListEqual(
        expected_group1, manager.get_op_group(op_slice1_4_5).op_slices)
    self.assertListEqual(
        expected_group1, manager.get_op_group(op_slice2_0_1).op_slices)
    self.assertListEqual(
        expected_group1, manager.get_op_group(op_slice3_8_9).op_slices)

    self.assertListEqual(
        expected_group2, manager.get_op_group(op_slice1_5_7).op_slices)
    self.assertListEqual(
        expected_group2, manager.get_op_group(op_slice2_1_3).op_slices)
    self.assertListEqual(
        expected_group2, manager.get_op_group(op_slice3_9_11).op_slices)

    self.assertListEqual(
        expected_group3, manager.get_op_group(op_slice1_7_10).op_slices)
    self.assertListEqual(
        expected_group3, manager.get_op_group(op_slice2_3_6).op_slices)
    self.assertListEqual(
        expected_group3, manager.get_op_group(op_slice3_11_14).op_slices)

    self.assertListEqual(
        expected_group4, manager.get_op_group(op_slice1_10_14).op_slices)
    self.assertListEqual(
        expected_group4, manager.get_op_group(op_slice2_6_10).op_slices)
    self.assertListEqual(
        expected_group4, manager.get_op_group(op_slice3_14_18).op_slices)

  def testProcessOps(self):
    inputs = tf.zeros([2, 4, 4, 3])
    batch_norm = layers.batch_norm(inputs)
    identity1 = tf.identity(batch_norm)
    identity2 = tf.identity(batch_norm)

    manager = orm.OpRegularizerManager(
        [identity1.op, identity2.op],
        op_handler_dict=self._default_op_handler_dict)
    manager.process_ops([identity1.op, identity2.op, batch_norm.op])

    self.assertLen(manager._op_deque, 3)
    self.assertEqual(batch_norm.op, manager._op_deque.pop())
    self.assertEqual(identity2.op, manager._op_deque.pop())
    self.assertEqual(identity1.op, manager._op_deque.pop())

  def testProcessOps_DuplicatesRemoved(self):
    inputs = tf.zeros([2, 4, 4, 3])
    batch_norm = layers.batch_norm(inputs)
    identity1 = tf.identity(batch_norm)
    identity2 = tf.identity(batch_norm)

    manager = orm.OpRegularizerManager(
        [identity1.op, identity2.op],
        op_handler_dict=self._default_op_handler_dict)
    manager.process_ops([identity1.op, identity2.op, batch_norm.op])
    # Try to process the same ops again.
    manager.process_ops([identity1.op, identity2.op, batch_norm.op])

    self.assertLen(manager._op_deque, 3)
    self.assertEqual(batch_norm.op, manager._op_deque.pop())
    self.assertEqual(identity2.op, manager._op_deque.pop())
    self.assertEqual(identity1.op, manager._op_deque.pop())

  def testProcessOpsLast(self):
    inputs = tf.zeros([2, 4, 4, 3])
    batch_norm = layers.batch_norm(inputs)
    identity1 = tf.identity(batch_norm)
    identity2 = tf.identity(batch_norm)

    manager = orm.OpRegularizerManager(
        [identity1.op, identity2.op],
        op_handler_dict=self._default_op_handler_dict)
    manager.process_ops([identity1.op])
    manager.process_ops_last([identity2.op, batch_norm.op])

    self.assertLen(manager._op_deque, 3)
    self.assertEqual(identity1.op, manager._op_deque.pop())
    self.assertEqual(identity2.op, manager._op_deque.pop())
    self.assertEqual(batch_norm.op, manager._op_deque.pop())

  def testProcessOpsLast_DuplicatesRemoved(self):
    inputs = tf.zeros([2, 4, 4, 3])
    batch_norm = layers.batch_norm(inputs)
    identity1 = tf.identity(batch_norm)
    identity2 = tf.identity(batch_norm)

    manager = orm.OpRegularizerManager(
        [identity1.op, identity2.op],
        op_handler_dict=self._default_op_handler_dict)
    manager.process_ops([identity1.op])
    manager.process_ops_last([identity2.op, batch_norm.op])
    # Try to process the same ops again.
    manager.process_ops_last([identity2.op, batch_norm.op])

    self.assertLen(manager._op_deque, 3)
    self.assertEqual(identity1.op, manager._op_deque.pop())
    self.assertEqual(identity2.op, manager._op_deque.pop())
    self.assertEqual(batch_norm.op, manager._op_deque.pop())

  def testIsSourceOp(self):
    inputs = tf.zeros([2, 4, 4, 3])
    identity = tf.identity(inputs)
    batch_norm = layers.batch_norm(identity)

    manager = orm.OpRegularizerManager([], self._default_op_handler_dict)

    self.assertFalse(manager.is_source_op(identity.op))
    self.assertTrue(manager.is_source_op(batch_norm.op))

  def testIsPassthrough(self):
    inputs = tf.zeros([2, 4, 4, 3])
    identity = tf.identity(inputs)
    layers.conv2d(identity, 5, 3, scope='conv1')

    manager = orm.OpRegularizerManager([], self._default_op_handler_dict)

    self.assertTrue(manager.is_passthrough(identity.op))
    # TODO(a1): Verify OutputNonPassthrough OpHandler returns False.

  def testGetOpSlices(self):
    inputs = tf.zeros([2, 4, 4, 3])
    identity = tf.identity(inputs)

    # Create OpRegularizerManager with OpSlice mapping.
    manager = orm.OpRegularizerManager([])
    op_slice = orm.OpSlice(identity.op, orm.Slice(0, 3))
    manager._op_slice_dict[identity.op] = [op_slice]

    op_slices = manager.get_op_slices(identity.op)

    self.assertLen(op_slices, 1)
    self.assertEqual(op_slice, op_slices[0])

  def testGetOpSlices_CreateNew(self):
    inputs = tf.zeros([2, 4, 4, 3])
    identity = tf.identity(inputs)

    # Create OpRegularizerManager with empty OpSlice dictionary.
    manager = orm.OpRegularizerManager([])
    manager._op_slice_dict = {}

    op_slices = manager.get_op_slices(identity.op)

    # Verify OpSlice is created correctly.
    self.assertLen(op_slices, 1)
    op_slice = op_slices[0]
    self.assertEqual(identity.op, op_slice.op)
    self.assertEqual(0, op_slice.slice.start_index)
    self.assertEqual(3, op_slice.slice.size)

  def testGetOpSlices_CreateNew_MultipleOutputs(self):
    inputs = tf.zeros([2, 4, 4, 10])
    split = tf.split(inputs, [3, 7], axis=3)
    split_op = split[0].op

    # Create OpRegularizerManager with empty OpSlice dictionary.
    manager = orm.OpRegularizerManager([])
    manager._op_slice_dict = {}

    op_slices = manager.get_op_slices(split_op)

    # Verify OpSlice is created correctly.
    self.assertLen(op_slices, 1)
    op_slice = op_slices[0]
    self.assertEqual(split_op, op_slice.op)
    self.assertEqual(0, op_slice.slice.start_index)
    self.assertEqual(10, op_slice.slice.size)

  def testGetOpSlices_ZeroSize(self):
    constant = tf.constant(123)

    # Create OpRegularizerManager with empty OpSlice dictionary.
    manager = orm.OpRegularizerManager([])
    manager._op_slice_dict = {}

    op_slices = manager.get_op_slices(constant.op)

    # Verify zero-size op has no slices.
    self.assertListEqual([], op_slices)

  def testSliceOpSlice(self):
    inputs = tf.zeros([2, 4, 4, 10])
    identity = tf.identity(inputs)

    op_slice1 = orm.OpSlice(identity.op, orm.Slice(0, 2))
    op_slice2 = orm.OpSlice(identity.op, orm.Slice(2, 6))
    op_slice3 = orm.OpSlice(identity.op, orm.Slice(8, 2))

    manager = orm.OpRegularizerManager([])
    manager._op_slice_dict[identity.op] = [op_slice1, op_slice2, op_slice3]

    # Original op has slice sizes [2, 6, 2].  The middle op is being sliced into
    # [1, 3, 2], so the new slice sizes are [2, 1, 3, 2, 2].
    sizes = [2, 1, 3, 2, 2]
    size_index = 1
    size_count = 3
    new_op_slice_group = [list() for _ in range(size_count)]
    manager._slice_op_slice(op_slice2, sizes, size_index, size_count,
                            new_op_slice_group)

    # Verify new slices are created.
    self.assertLen(new_op_slice_group, size_count)
    for i in range(size_count):
      self.assertLen(new_op_slice_group[i], 1)

    # Verify new slices are correct.
    new_slice1 = new_op_slice_group[0][0]
    self.assertEqual(2, new_slice1.slice.start_index)
    self.assertEqual(1, new_slice1.slice.size)

    new_slice2 = new_op_slice_group[1][0]
    self.assertEqual(3, new_slice2.slice.start_index)
    self.assertEqual(3, new_slice2.slice.size)

    new_slice3 = new_op_slice_group[2][0]
    self.assertEqual(6, new_slice3.slice.start_index)
    self.assertEqual(2, new_slice3.slice.size)

  def testSliceOpWithSizes(self):
    inputs = tf.zeros([2, 4, 4, 10])
    identity = tf.identity(inputs)

    manager = orm.OpRegularizerManager([])

    sizes = [1, 2, 3, 4]
    is_source = [True, False, True, False]
    is_resliced = [True, True, True, True]
    op_slices = manager._slice_op_with_sizes(identity.op, sizes, is_source,
                                             is_resliced)

    # Verify OpSlice count and whether they are sources.
    self.assertLen(op_slices, 4)

    slice1 = op_slices[0]
    op_group1 = manager.get_op_group(slice1)
    self.assertIn(slice1, op_group1.source_op_slices)

    slice2 = op_slices[1]
    op_group2 = manager.get_op_group(slice2)
    self.assertIsNone(op_group2)

    slice3 = op_slices[2]
    op_group3 = manager.get_op_group(slice3)
    self.assertIn(slice3, op_group3.source_op_slices)

    slice4 = op_slices[3]
    op_group4 = manager.get_op_group(slice4)
    self.assertIsNone(op_group4)

  def testGetSourceSlices(self):
    inputs = tf.zeros([2, 4, 4, 10])
    identity = tf.identity(inputs)

    manager = orm.OpRegularizerManager([])

    # Create OpSlices with size [3, 7].
    identity_slice1 = orm.OpSlice(identity.op, orm.Slice(0, 3))
    identity_slice2 = orm.OpSlice(identity.op, orm.Slice(3, 7))

    # Create OpGroup where only first group has source OpSlice.
    manager.create_op_group_for_op_slice(identity_slice1)
    manager.create_op_group_for_op_slice(identity_slice2,
                                         is_source=False)

    # First slice of size 3 is sliced into [1, 2], so these are sources.  Second
    # slice of size 7 is sliced into [3, 4], which are not sources.
    sizes = [1, 2, 3, 4]
    expected_sources = [True, True, False, False]
    self.assertListEqual(
        expected_sources,
        manager._get_source_slices(sizes, [identity_slice1, identity_slice2]))

  def testDfsForSourceOps(self):
    with tf.contrib.framework.arg_scope(self._batch_norm_scope()):
      inputs = tf.zeros([2, 4, 4, 3])
      c1 = layers.conv2d(inputs, num_outputs=10, kernel_size=3, scope='conv1')
      c2 = layers.conv2d(inputs, num_outputs=10, kernel_size=3, scope='conv2')
      tmp = c1 + c2
      c3 = layers.conv2d(tmp, num_outputs=10, kernel_size=3, scope='conv3')
      out = tf.identity(c3)
      # Extra branch that is not a dependency of out.
      concat = tf.concat([c1, c2], axis=3)
      layers.conv2d(concat, num_outputs=10, kernel_size=3, scope='conv4')

    manager = orm.OpRegularizerManager([], self._default_op_handler_dict)
    manager._dfs_for_source_ops([out.op])

    # Verify source ops were found.
    expected_queue = collections.deque([
        _get_op('conv3/BatchNorm/FusedBatchNormV3'),
        _get_op('conv2/BatchNorm/FusedBatchNormV3'),
        _get_op('conv1/BatchNorm/FusedBatchNormV3')
    ])
    self.assertEqual(expected_queue, manager._op_deque)

    # Verify extra branch was not included.
    self.assertNotIn(
        _get_op('conv4/BatchNorm/FusedBatchNormV3'), manager._op_deque)

  def testOpGroup_NewSourceGroup(self):
    inputs = tf.zeros([2, 4, 4, 3])
    identity = tf.identity(inputs)
    op_slice = orm.OpSlice(identity.op, None)
    op_group = orm.OpGroup(op_slice)

    self.assertListEqual([op_slice], op_group.op_slices)
    self.assertListEqual([op_slice], op_group.source_op_slices)

  def testOpGroup_NewGroupNoSource(self):
    inputs = tf.zeros([2, 4, 4, 3])
    identity = tf.identity(inputs)
    op_slice = orm.OpSlice(identity.op, None)
    op_group = orm.OpGroup(op_slice, omit_source_op_slices=[op_slice])

    self.assertListEqual([op_slice], op_group.op_slices)
    self.assertListEqual([], op_group.source_op_slices)

  def testOpGroup_NewSourceGroup_DuplicateOpSlice(self):
    inputs = tf.zeros([2, 4, 4, 3])
    identity1 = tf.identity(inputs)
    identity2 = tf.identity(inputs)
    op_slice1 = orm.OpSlice(identity1.op, None)
    op_slice2 = orm.OpSlice(identity2.op, None)
    op_group1 = orm.OpGroup(op_slice1)
    op_group2 = orm.OpGroup(
        op_slice2, [op_group1], omit_source_op_slices=[op_slice2])
    op_group3 = orm.OpGroup(op_groups=[op_group1, op_group2])

    self.assertListEqual([op_slice1, op_slice2], op_group3.op_slices)
    self.assertListEqual([op_slice1], op_group3.source_op_slices)

  def testOpGroup_MergeGroups(self):
    inputs = tf.zeros([2, 4, 4, 3])
    identity1 = tf.identity(inputs)
    identity2 = tf.identity(inputs)
    identity3 = tf.identity(inputs)
    identity4 = tf.identity(inputs)
    identity5 = tf.identity(inputs)
    identity6 = tf.identity(inputs)
    identity7 = tf.identity(inputs)
    identity8 = tf.identity(inputs)

    # Reset OpGroup counter.
    orm.OpGroup._static_index = 0

    # Create OpGroup where only identity3, identity6, and identity7 are sources.
    op_slice1 = orm.OpSlice(identity1.op, None)
    op_group1 = orm.OpGroup(op_slice1, omit_source_op_slices=[op_slice1])
    op_slice2 = orm.OpSlice(identity2.op, None)
    op_group2 = orm.OpGroup(op_slice2, omit_source_op_slices=[op_slice2])
    op_slice3 = orm.OpSlice(identity3.op, None)
    op_group3 = orm.OpGroup(op_slice3)
    op_slice4 = orm.OpSlice(identity4.op, None)
    op_group4 = orm.OpGroup(op_slice4, omit_source_op_slices=[op_slice4])
    op_slice5 = orm.OpSlice(identity5.op, None)
    op_group5 = orm.OpGroup(op_slice5, omit_source_op_slices=[op_slice5])
    op_slice6 = orm.OpSlice(identity6.op, None)
    op_group6 = orm.OpGroup(op_slice6)
    op_slice7 = orm.OpSlice(identity7.op, None)
    op_group7 = orm.OpGroup(op_slice7)
    op_slice8 = orm.OpSlice(identity8.op, None)
    op_group8 = orm.OpGroup(op_slice8, omit_source_op_slices=[op_slice8])

    # Merge group1 and group2 into group9.
    op_group9 = orm.OpGroup(op_groups=[op_group1, op_group2])
    self.assertListEqual([op_slice1, op_slice2], op_group9.op_slices)
    self.assertListEqual([], op_group9.source_op_slices)
    self.assertEqual(8, op_group9._index)  # OpGroup is zero-indexed.

    # Merge group3 and group4 into group10.
    op_group10 = orm.OpGroup(op_groups=[op_group3, op_group4])
    self.assertListEqual([op_slice3, op_slice4], op_group10.op_slices)
    self.assertListEqual([op_slice3], op_group10.source_op_slices)
    self.assertEqual(9, op_group10._index)  # OpGroup is zero-indexed.

    # Merge group5, group6, group7, and group8 into group 11.
    op_group11 = orm.OpGroup(
        op_groups=[op_group5, op_group6, op_group7, op_group8])
    self.assertListEqual(
        [op_slice5, op_slice6, op_slice7, op_slice8], op_group11.op_slices)
    self.assertListEqual([op_slice6, op_slice7], op_group11.source_op_slices)
    self.assertEqual(10, op_group11._index)  # OpGroup is zero-indexed.

    # Merge group9 and group10 into group12.
    op_group12 = orm.OpGroup(op_groups=[op_group9, op_group10])
    self.assertListEqual(
        [op_slice1, op_slice2, op_slice3, op_slice4], op_group12.op_slices)
    self.assertListEqual([op_slice3], op_group12.source_op_slices)
    self.assertEqual(11, op_group12._index)  # OpGroup is zero-indexed.

    # Merge group11 and group12 into group13.
    op_group13 = orm.OpGroup(op_groups=[op_group11, op_group12])
    self.assertListEqual(
        [op_slice5, op_slice6, op_slice7, op_slice8, op_slice1, op_slice2,
         op_slice3, op_slice4],
        op_group13.op_slices)
    self.assertListEqual([op_slice6, op_slice7, op_slice3],
                         op_group13.source_op_slices)
    self.assertEqual(12, op_group13._index)  # OpGroup is zero-indexed.

  def testPrintOpSlices(self):
    inputs = tf.zeros([2, 4, 4, 3])
    identity1 = tf.identity(inputs)
    identity2 = tf.identity(inputs)

    manager = orm.OpRegularizerManager(
        [identity1.op, identity2.op],
        op_handler_dict=self._default_op_handler_dict)
    op_slices1 = manager.get_op_slices(identity1.op)
    op_slices2 = manager.get_op_slices(identity2.op)
    all_slices = op_slices1 + op_slices2

    self.assertEqual('[Identity (0, 3), Identity_1 (0, 3)]',
                     str(all_slices))


class IndexOpRegularizer(generic_regularizers.OpRegularizer):
  """A test OpRegularizer with a self-incrementing index.

  This class creates a regularizer where the regularization vector contains
  self-incrementing values (e.g. [0, 1, 2, ...]).  The index continues to
  increment as regularizers are created.  This is convenient for testing in
  order to track individual elements of the regularization vector (e.g. gather).

  For example, creating 2 regularizers of size 3 results in
  r1 = [0, 1, 2] and r2 = [3, 4, 5].
  """

  index = 0

  def __init__(self, op_slice, op_reg_manager):
    size = op_slice.slice.size
    self._alive_vector = tf.cast(tf.ones(size), tf.bool)
    self._regularization_vector = tf.constant(
        list(range(IndexOpRegularizer.index, IndexOpRegularizer.index + size)),
        tf.float32)
    IndexOpRegularizer.index += size

  @classmethod
  def reset_index(cls):
    IndexOpRegularizer.index = 0

  @property
  def regularization_vector(self):
    return self._regularization_vector

  @property
  def alive_vector(self):
    return self._alive_vector


class SumGroupingRegularizer(generic_regularizers.OpRegularizer):
  """A regularizer that groups others by summing their regularization values."""

  def __init__(self, regularizers_to_group):
    """Creates an instance.

    Args:
      regularizers_to_group: A list of generic_regularizers.OpRegularizer
        objects.Their regularization_vector (alive_vector) are expected to be of
        the same length.

    Raises:
      ValueError: regularizers_to_group is not of length at least 2.
    """
    if len(regularizers_to_group) < 2:
      raise ValueError('Groups must be of at least size 2.')
    self._regularization_vector = tf.add_n(
        [r.regularization_vector for r in regularizers_to_group])
    self._alive_vector = tf.cast(
        tf.ones(self._regularization_vector.get_shape()[-1]), tf.bool)

  @property
  def regularization_vector(self):
    return self._regularization_vector

  @property
  def alive_vector(self):
    return self._alive_vector


class IndexBatchNormSourceOpHandler(
    batch_norm_source_op_handler.BatchNormSourceOpHandler):
  """An OpHandler that creates OpRegularizer using IndexOpRegularizer.

  A wrapper around BatchNormSourceOpHandler that overrides the
  create_regularizer method to use IndexOpRegularizer for testing.
  """

  def __init__(self):
    super(IndexBatchNormSourceOpHandler, self).__init__(0.0)

  def create_regularizer(self, op_slice):
    return IndexOpRegularizer(op_slice, None)


class StubBatchNormSourceOpHandler(
    batch_norm_source_op_handler.BatchNormSourceOpHandler):
  """An OpHandler that creates OpRegularizer using stub values.

  A wrapper around BatchNormSourceOpHandler that overrides the
  create_regularizer method to use stub values for testing.
  """

  def __init__(self, model_stub):
    super(StubBatchNormSourceOpHandler, self).__init__(0.0)
    self._model_stub = model_stub

  def create_regularizer(self, op_slice):
    return _stub_create_regularizer(op_slice, self._model_stub)


class IndexConv2DSourceOpHandler(
    conv2d_source_op_handler.Conv2DSourceOpHandler):
  """An OpHandler that creates OpRegularizer using IndexOpRegularizer.

  A wrapper around Conv2DSourceOpHandler that overrides the create_regularizer
  method to use IndexOpRegularizer for testing.
  """

  def __init__(self):
    pass

  def create_regularizer(self, op_slice):
    return IndexOpRegularizer(op_slice, None)


class StubConv2DSourceOpHandler(conv2d_source_op_handler.Conv2DSourceOpHandler):
  """An OpHandler that creates OpRegularizer using stub values.

  A wrapper around Conv2DSourceOpHandler that overrides the create_regularizer
  method to use stub values for testing.
  """

  def __init__(self, model_stub):
    super(StubConv2DSourceOpHandler, self).__init__(0.1)
    self._model_stub = model_stub

  def create_regularizer(self, op_slice):
    return _stub_create_regularizer(op_slice, self._model_stub)


class RandomConv2DSourceOpHandler(
    conv2d_source_op_handler.Conv2DSourceOpHandler):
  """An OpHandler that creates OpRegularizer using random values.

  A wrapper around Conv2DSourceOpHandler that overrides the create_regularizer
  method to use random values for testing.
  """

  def create_regularizer(self, op_slice):
    regularization_vector = np.random.random(op_slice.slice.size)
    return StubOpRegularizer(regularization_vector,
                             regularization_vector > self._threshold)


def _stub_create_regularizer(op_slice, model_stub):
  """Create a StubOpRegularizer for a given OpSlice.

  Args:
    op_slice: A op_regularizer_manager.OpSlice.
    model_stub: Module name where REG_STUB and ALIVE_STUB will be found.

  Returns:
    StubOpRegularizer with stubbed regularization and alive vectors.
  """
  op = op_slice.op
  start_index = op_slice.slice.start_index
  size = op_slice.slice.size
  for key in model_stub.REG_STUB:
    if op.name.startswith(key):
      return StubOpRegularizer(
          model_stub.REG_STUB[key][start_index:start_index + size],
          model_stub.ALIVE_STUB[key][start_index:start_index + size])
  raise ValueError('No regularizer for %s' % op.name)


class StubOpRegularizer(generic_regularizers.OpRegularizer):
  """A test OpRegularizer with configured regularization vectors.

  Regularization values are stored in a dict and keyed on op name prefix.
  """

  def __init__(self, regularization_vector, alive_vector):
    self._regularization_vector = tf.constant(regularization_vector)
    self._alive_vector = tf.constant(alive_vector, dtype=tf.bool)

  @property
  def regularization_vector(self):
    return self._regularization_vector

  @property
  def alive_vector(self):
    return self._alive_vector


if __name__ == '__main__':
  tf.test.main()
