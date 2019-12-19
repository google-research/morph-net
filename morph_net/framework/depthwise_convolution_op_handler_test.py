"""Tests for depthwise_convolution_op_handler."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import mock
from morph_net.framework import depthwise_convolution_op_handler
from morph_net.framework import op_regularizer_manager as orm
import tensorflow.compat.v1 as tf
from tensorflow.contrib import framework as framework
from tensorflow.contrib import layers

arg_scope = framework.arg_scope


class DepthwiseConvolutionOpHandlerTest(tf.test.TestCase):

  def _batch_norm_scope(self):
    params = {
        'trainable': True,
        'normalizer_fn': layers.batch_norm,
        'normalizer_params': {
            'scale': True,
        },
    }

    with arg_scope([layers.conv2d], **params) as sc:
      return sc

  def setUp(self):
    super(DepthwiseConvolutionOpHandlerTest, self).setUp()
    tf.reset_default_graph()

    # This tests a Conv2D -> SeparableConv2D -> Conv2D chain of ops.
    with framework.arg_scope(self._batch_norm_scope()):
      inputs = tf.zeros([2, 4, 4, 3])
      c1 = layers.conv2d(inputs, num_outputs=5, kernel_size=3, scope='conv1')
      c2 = layers.separable_conv2d(c1, num_outputs=8, kernel_size=3,
                                   depth_multiplier=2, scope='conv2')
      layers.conv2d(c2, num_outputs=6, kernel_size=3, scope='conv3')

    g = tf.get_default_graph()

    # Declare OpSlice and OpGroup for ops of interest.
    self.dwise_conv2_op = g.get_operation_by_name(
        'conv2/separable_conv2d/depthwise')
    self.dwise_conv2_op_slice = orm.OpSlice(
        self.dwise_conv2_op, orm.Slice(0, 10))
    self.dwise_conv2_op_slice_0_1 = orm.OpSlice(
        self.dwise_conv2_op, orm.Slice(0, 1))
    self.dwise_conv2_op_slice_1_2 = orm.OpSlice(
        self.dwise_conv2_op, orm.Slice(1, 1))
    self.dwise_conv2_op_slice_2_3 = orm.OpSlice(
        self.dwise_conv2_op, orm.Slice(2, 1))
    self.dwise_conv2_op_slice_3_4 = orm.OpSlice(
        self.dwise_conv2_op, orm.Slice(3, 1))
    self.dwise_conv2_op_slice_4_5 = orm.OpSlice(
        self.dwise_conv2_op, orm.Slice(4, 1))
    self.dwise_conv2_op_slice_5_6 = orm.OpSlice(
        self.dwise_conv2_op, orm.Slice(5, 1))
    self.dwise_conv2_op_slice_6_7 = orm.OpSlice(
        self.dwise_conv2_op, orm.Slice(6, 1))
    self.dwise_conv2_op_slice_7_8 = orm.OpSlice(
        self.dwise_conv2_op, orm.Slice(7, 1))
    self.dwise_conv2_op_slice_8_9 = orm.OpSlice(
        self.dwise_conv2_op, orm.Slice(8, 1))
    self.dwise_conv2_op_slice_9_10 = orm.OpSlice(
        self.dwise_conv2_op, orm.Slice(9, 1))

    self.conv2_op = g.get_operation_by_name('conv2/separable_conv2d')
    self.conv2_op_slice = orm.OpSlice(self.conv2_op, orm.Slice(0, 8))

    self.relu1_op = g.get_operation_by_name('conv1/Relu')
    self.relu1_op_slice = orm.OpSlice(self.relu1_op, orm.Slice(0, 5))
    self.relu1_op_slice_0_1 = orm.OpSlice(self.relu1_op, orm.Slice(0, 1))
    self.relu1_op_slice_1_2 = orm.OpSlice(self.relu1_op, orm.Slice(1, 1))
    self.relu1_op_slice_2_3 = orm.OpSlice(self.relu1_op, orm.Slice(2, 1))
    self.relu1_op_slice_3_4 = orm.OpSlice(self.relu1_op, orm.Slice(3, 1))
    self.relu1_op_slice_4_5 = orm.OpSlice(self.relu1_op, orm.Slice(4, 1))
    self.relu1_op_group = orm.OpGroup(self.relu1_op_slice)

    self.conv3_op = g.get_operation_by_name('conv3/Conv2D')
    self.conv3_op_slice = orm.OpSlice(self.conv3_op, orm.Slice(0, 6))

    # Create mock OpRegularizerManager with custom mapping of OpSlice and
    # OpGroup.
    self.mock_op_reg_manager = mock.create_autospec(orm.OpRegularizerManager)

    self.op_slice_dict = {
        self.dwise_conv2_op: [self.dwise_conv2_op_slice],
        self.conv2_op: [self.conv2_op_slice],
        self.relu1_op: [self.relu1_op_slice],
        self.conv3_op: [self.conv3_op_slice],
    }
    def get_op_slices(op):
      return self.op_slice_dict.get(op)

    def get_op_group(op_slice):
      return self.op_group_dict.get(op_slice)

    # Update op_slice_dict when an op is sliced.
    def slice_op(op, _):
      if op == self.dwise_conv2_op:
        self.op_slice_dict[self.dwise_conv2_op] = [
            self.dwise_conv2_op_slice_0_1,
            self.dwise_conv2_op_slice_1_2,
            self.dwise_conv2_op_slice_2_3,
            self.dwise_conv2_op_slice_3_4,
            self.dwise_conv2_op_slice_4_5,
            self.dwise_conv2_op_slice_5_6,
            self.dwise_conv2_op_slice_6_7,
            self.dwise_conv2_op_slice_7_8,
            self.dwise_conv2_op_slice_8_9,
            self.dwise_conv2_op_slice_9_10]
      if op == self.relu1_op:
        self.op_slice_dict[self.relu1_op] = [
            self.relu1_op_slice_0_1,
            self.relu1_op_slice_1_2,
            self.relu1_op_slice_2_3,
            self.relu1_op_slice_3_4,
            self.relu1_op_slice_4_5]

    self.mock_op_reg_manager.get_op_slices.side_effect = get_op_slices
    self.mock_op_reg_manager.get_op_group.side_effect = get_op_group
    self.mock_op_reg_manager.is_source_op.return_value = False
    self.mock_op_reg_manager.slice_op.side_effect = slice_op
    self.mock_op_reg_manager.ops = [
        self.relu1_op, self.dwise_conv2_op, self.conv2_op, self.conv3_op]

  def testAssignGrouping_DepthMultiplier(self):
    # All neighbor ops have groups.
    self.op_group_dict = {
        self.relu1_op_slice: self.relu1_op_group,
    }

    # Call handler to assign grouping.
    handler = depthwise_convolution_op_handler.DepthwiseConvolutionOpHandler()
    handler.assign_grouping(self.dwise_conv2_op, self.mock_op_reg_manager)

    # Verify manager looks up OpSlice for ops of interest.
    self.mock_op_reg_manager.get_op_slices.assert_has_calls(
        # Checking for ops to process.
        [mock.call(self.relu1_op),
         mock.call(self.conv2_op),
         # Reslicing.
         mock.call(self.relu1_op),
         mock.call(self.dwise_conv2_op),
         # Refreshing slice data.
         mock.call(self.relu1_op),
         # Group depthwise convolution.
         mock.call(self.dwise_conv2_op)])

    # Verify manager groups batch norm with inputs and outputs.
    self.mock_op_reg_manager.group_op_slices.assert_has_calls(
        [mock.call([self.dwise_conv2_op_slice_0_1, self.relu1_op_slice_0_1]),
         mock.call([self.dwise_conv2_op_slice_1_2, self.relu1_op_slice_0_1]),
         mock.call([self.dwise_conv2_op_slice_2_3, self.relu1_op_slice_1_2]),
         mock.call([self.dwise_conv2_op_slice_3_4, self.relu1_op_slice_1_2]),
         mock.call([self.dwise_conv2_op_slice_4_5, self.relu1_op_slice_2_3]),
         mock.call([self.dwise_conv2_op_slice_5_6, self.relu1_op_slice_2_3]),
         mock.call([self.dwise_conv2_op_slice_6_7, self.relu1_op_slice_3_4]),
         mock.call([self.dwise_conv2_op_slice_7_8, self.relu1_op_slice_3_4]),
         mock.call([self.dwise_conv2_op_slice_8_9, self.relu1_op_slice_4_5]),
         mock.call([self.dwise_conv2_op_slice_9_10, self.relu1_op_slice_4_5])])

    # Verify manager does not process any additional ops.
    self.mock_op_reg_manager.process_ops.assert_called_once_with(
        [self.conv2_op])
    self.mock_op_reg_manager.process_ops_last.assert_not_called()

  def testAssignGrouping_NoDepthMultiplier(self):
    # Repeat setUp, but with depth_multiplier=1.  Unfortunately, this involves
    # rebuilding the graph from scratch.
    tf.reset_default_graph()

    # This tests a Conv2D -> SeparableConv2D -> Conv2D chain of ops.
    with framework.arg_scope(self._batch_norm_scope()):
      inputs = tf.zeros([2, 4, 4, 3])
      c1 = layers.conv2d(inputs, num_outputs=5, kernel_size=3, scope='conv1')
      c2 = layers.separable_conv2d(c1, num_outputs=8, kernel_size=3,
                                   depth_multiplier=1, scope='conv2')
      layers.conv2d(c2, num_outputs=6, kernel_size=3, scope='conv3')

    g = tf.get_default_graph()

    # Declare OpSlice and OpGroup for ops of interest.
    self.dwise_conv2_op = g.get_operation_by_name(
        'conv2/separable_conv2d/depthwise')
    self.dwise_conv2_op_slice = orm.OpSlice(
        self.dwise_conv2_op, orm.Slice(0, 5))

    self.conv2_op = g.get_operation_by_name('conv2/separable_conv2d')
    self.conv2_op_slice = orm.OpSlice(self.conv2_op, orm.Slice(0, 8))

    self.relu1_op = g.get_operation_by_name('conv1/Relu')
    self.relu1_op_slice = orm.OpSlice(self.relu1_op, orm.Slice(0, 5))
    self.relu1_op_group = orm.OpGroup(self.relu1_op_slice)

    self.conv3_op = g.get_operation_by_name('conv3/Conv2D')
    self.conv3_op_slice = orm.OpSlice(self.conv3_op, orm.Slice(0, 6))

    # Create mock OpRegularizerManager with custom mapping of OpSlice and
    # OpGroup.
    self.mock_op_reg_manager = mock.create_autospec(orm.OpRegularizerManager)

    self.op_slice_dict = {
        self.dwise_conv2_op: [self.dwise_conv2_op_slice],
        self.conv2_op: [self.conv2_op_slice],
        self.relu1_op: [self.relu1_op_slice],
        self.conv3_op: [self.conv3_op_slice],
    }
    def get_op_slices(op):
      return self.op_slice_dict.get(op)

    def get_op_group(op_slice):
      return self.op_group_dict.get(op_slice)

    self.mock_op_reg_manager.get_op_slices.side_effect = get_op_slices
    self.mock_op_reg_manager.get_op_group.side_effect = get_op_group
    self.mock_op_reg_manager.is_source_op.return_value = False
    self.mock_op_reg_manager.ops = [
        self.relu1_op, self.dwise_conv2_op, self.conv2_op, self.conv3_op]

    # All neighbor ops have groups.
    self.op_group_dict = {
        self.relu1_op_slice: self.relu1_op_group,
    }

    # Call handler to assign grouping.
    handler = depthwise_convolution_op_handler.DepthwiseConvolutionOpHandler()
    handler.assign_grouping(self.dwise_conv2_op, self.mock_op_reg_manager)

    # Verify manager looks up OpSlice for ops of interest.
    self.mock_op_reg_manager.get_op_slices.assert_has_calls(
        # Checking for ops to process.
        [mock.call(self.relu1_op),
         mock.call(self.conv2_op),
         # Initial slice data.
         mock.call(self.dwise_conv2_op),
         mock.call(self.relu1_op),
         # Reslicing.
         mock.call(self.relu1_op),
         mock.call(self.dwise_conv2_op),
         # Refreshing slice data.
         mock.call(self.relu1_op),
         # Group depthwise convolution.
         mock.call(self.dwise_conv2_op)])

    # Verify manager groups batch norm with inputs and outputs.
    self.mock_op_reg_manager.group_op_slices.assert_called_once_with(
        [self.dwise_conv2_op_slice, self.relu1_op_slice])

    # Verify manager does not process any additional ops.
    self.mock_op_reg_manager.process_ops.assert_called_once_with(
        [self.conv2_op])
    self.mock_op_reg_manager.process_ops_last.assert_not_called()

  def testDepthwiseChannelMapping(self):
    """Verify depth multiplier maps input to output as expected."""
    tf.reset_default_graph()

    # Construct input tensor with shape [1, 4, 4, 5].  There are 5 channels
    # where each channel has values corresponding to the channel index.
    channel0 = tf.ones([1, 4, 4, 1]) * 0
    channel1 = tf.ones([1, 4, 4, 1]) * 1
    channel2 = tf.ones([1, 4, 4, 1]) * 2
    channel3 = tf.ones([1, 4, 4, 1]) * 3
    channel4 = tf.ones([1, 4, 4, 1]) * 4
    inputs = tf.concat(
        [channel0, channel1, channel2, channel3, channel4], axis=3)
    # Sanity check that input tensor is the right shape.
    self.assertAllEqual([1, 4, 4, 5], inputs.shape.as_list())

    conv = layers.separable_conv2d(
        inputs, num_outputs=None, kernel_size=3, depth_multiplier=2,
        weights_initializer=identity_initializer, scope='depthwise_conv')

    with self.cached_session():
      with tf.variable_scope('', reuse=tf.AUTO_REUSE):
        weights = tf.get_variable('depthwise_conv/depthwise_weights')
        biases = tf.get_variable('depthwise_conv/biases', [10],
                                 initializer=tf.zeros_initializer)
      init = tf.variables_initializer([weights, biases])
      init.run()

      # The depth_multiplier replicates channels with [a, a, b, b, c, c, ...]
      # pattern.  Expected output has shape [1, 4, 4, 10].
      expected_output = tf.concat(
          [channel0, channel0,
           channel1, channel1,
           channel2, channel2,
           channel3, channel3,
           channel4, channel4],
          axis=3)
      # Sanity check that output tensor is the right shape.
      self.assertAllEqual([1, 4, 4, 10], expected_output.shape.as_list())

      self.assertAllEqual(expected_output.eval(), conv.eval())


def identity_initializer(shape, dtype=None, partition_info=None):
  """Fake weight initializer to initialize a 3x3 identity kernel."""
  del shape  # Unused.
  del dtype  # Unused.
  del partition_info  # Unused.

  # Start with a 3x3 kernel identity kernel.
  kernel = [[0, 0, 0],
            [0, 1, 0],
            [0, 0, 0]]

  # Expand and tile kernel to get a tensor with shape [3, 3, 5, 2].
  kernel = tf.expand_dims(kernel, axis=-1)
  kernel = tf.expand_dims(kernel, axis=-1)
  tensor = tf.tile(kernel, [1, 1, 5, 2])
  return tf.cast(tensor, dtype=tf.float32)


if __name__ == '__main__':
  tf.test.main()
