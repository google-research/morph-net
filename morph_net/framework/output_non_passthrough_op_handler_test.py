"""Tests for output_non_passthrough_op_handler."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import mock
from morph_net.framework import op_regularizer_manager as orm
from morph_net.framework import output_non_passthrough_op_handler
import tensorflow as tf

layers = tf.contrib.layers


class OutputNonPassthroughOpHandlerTest(tf.test.TestCase):

  def _batch_norm_scope(self):
    params = {
        'trainable': True,
        'normalizer_fn': layers.batch_norm,
        'normalizer_params': {
            'scale': True,
        },
    }

    with tf.contrib.framework.arg_scope([layers.conv2d], **params) as sc:
      return sc

  def setUp(self):
    tf.reset_default_graph()

    # This tests 2 Conv2D ops with batch norm at the top.
    with tf.contrib.framework.arg_scope(self._batch_norm_scope()):
      inputs = tf.zeros([2, 4, 4, 3])
      c1 = layers.conv2d(inputs, num_outputs=5,
                         kernel_size=3, scope='conv1', normalizer_fn=None)
      layers.conv2d(c1, num_outputs=6, kernel_size=3, scope='conv2')

    g = tf.get_default_graph()

    # Declare OpSlice and OpGroup for ops of interest.
    self.conv1_op = g.get_operation_by_name('conv1/Conv2D')
    self.conv1_op_slice = orm.OpSlice(self.conv1_op, orm.Slice(0, 5))
    self.conv1_op_group = orm.OpGroup(
        self.conv1_op_slice, omit_source_op_slices=[self.conv1_op_slice])

    self.relu1_op = g.get_operation_by_name('conv1/Relu')
    self.relu1_op_slice = orm.OpSlice(self.relu1_op, orm.Slice(0, 5))
    self.relu1_op_group = orm.OpGroup(
        self.relu1_op_slice, omit_source_op_slices=[self.relu1_op_slice])

    self.conv2_op = g.get_operation_by_name('conv2/Conv2D')
    self.conv2_op_slice = orm.OpSlice(self.conv2_op, orm.Slice(0, 6))
    self.conv2_op_group = orm.OpGroup(
        self.conv2_op_slice, omit_source_op_slices=[self.conv2_op_slice])

    self.relu2_op = g.get_operation_by_name('conv2/Relu')
    self.relu2_op_slice = orm.OpSlice(self.relu2_op, orm.Slice(0, 6))
    self.relu2_op_group = orm.OpGroup(
        self.relu2_op_slice, omit_source_op_slices=[self.relu2_op_slice])

    self.batch_norm_op = g.get_operation_by_name(
        'conv2/BatchNorm/FusedBatchNormV3')
    self.batch_norm_op_slice = orm.OpSlice(self.batch_norm_op, orm.Slice(0, 6))
    self.batch_norm_op_group = orm.OpGroup(self.batch_norm_op_slice)

    # Create mock OpRegularizerManager with custom mapping of OpSlice and
    # OpGroup.
    self.mock_op_reg_manager = mock.create_autospec(orm.OpRegularizerManager)

    def get_op_slices(op):
      return self.op_slice_dict.get(op, [])

    def get_op_group(op_slice):
      return self.op_group_dict.get(op_slice)

    def is_passthrough(op):
      if op in [self.conv1_op, self.conv2_op]:
        h = output_non_passthrough_op_handler.OutputNonPassthroughOpHandler()
        return h.is_passthrough
      if op == self.batch_norm_op:
        return True
      else:
        return False

    self.mock_op_reg_manager.get_op_slices.side_effect = get_op_slices
    self.mock_op_reg_manager.get_op_group.side_effect = get_op_group
    self.mock_op_reg_manager.is_source_op.return_value = False
    self.mock_op_reg_manager.is_passthrough.side_effect = is_passthrough
    self.mock_op_reg_manager.ops = [
        self.conv1_op, self.relu1_op, self.conv2_op, self.relu2_op,
        self.batch_norm_op]

  def testAssignGrouping_GroupWithOutputOnly(self):
    # Map ops to slices.
    self.op_slice_dict = {
        self.conv1_op: [self.conv1_op_slice],
        self.relu1_op: [self.relu1_op_slice],
        self.conv2_op: [self.conv2_op_slice],
        self.relu2_op: [self.relu2_op_slice],
        self.batch_norm_op: [self.batch_norm_op_slice],
    }

    # Map each slice to a group. Corresponding op slices have the same group.
    self.op_group_dict = {
        self.batch_norm_op_slice: self.batch_norm_op_group,
    }

    # Call handler to assign grouping.
    handler = output_non_passthrough_op_handler.OutputNonPassthroughOpHandler()
    handler.assign_grouping(self.conv2_op, self.mock_op_reg_manager)

    # Verify manager looks up OpSlice for ops of interest.
    self.mock_op_reg_manager.get_op_slices.assert_has_calls(
        # Checking for ops to process.
        [mock.call(self.relu1_op),
         mock.call(self.batch_norm_op),
         # Align.
         mock.call(self.conv2_op),
         mock.call(self.batch_norm_op),
         # Slice.
         mock.call(self.conv2_op),
         mock.call(self.batch_norm_op),
         # Update after align.
         mock.call(self.batch_norm_op),
         # Grouping.
         mock.call(self.conv2_op)])

    # Verify manager does not slice any ops.
    self.mock_op_reg_manager.slice_op.assert_not_called()

    # Verify manager adds inputs to process queue.
    self.mock_op_reg_manager.process_ops.assert_called_once_with(
        [self.relu1_op])

    # Verify manager groups c2 with bn.
    self.mock_op_reg_manager.group_op_slices.assert_called_once_with(
        [self.conv2_op_slice, self.batch_norm_op_slice])


if __name__ == '__main__':
  tf.test.main()
