"""Tests for leaf_op_handler."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import mock
from morph_net.framework import leaf_op_handler
from morph_net.framework import op_regularizer_manager as orm
import tensorflow as tf

arg_scope = tf.contrib.framework.arg_scope
layers = tf.contrib.layers


class LeafOpHandlerTest(tf.test.TestCase):

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
    tf.reset_default_graph()

    # This tests a Conv2D -> BatchNorm -> ReLU chain of ops.
    with tf.contrib.framework.arg_scope(self._batch_norm_scope()):
      inputs = tf.zeros([2, 4, 4, 3])
      layers.conv2d(inputs, num_outputs=5, kernel_size=3, scope='conv1')

    g = tf.get_default_graph()

    # Declare OpSlice and OpGroup for ops of interest.
    self.batch_norm_op = g.get_operation_by_name(
        'conv1/BatchNorm/FusedBatchNormV3')
    self.batch_norm_op_slice = orm.OpSlice(self.batch_norm_op, orm.Slice(0, 5))
    self.batch_norm_op_group = orm.OpGroup(self.batch_norm_op_slice)

    self.conv_op = g.get_operation_by_name('conv1/Conv2D')
    self.conv_op_slice = orm.OpSlice(self.conv_op, orm.Slice(0, 5))
    self.conv_op_group = orm.OpGroup(
        self.conv_op_slice, omit_source_op_slices=[self.conv_op_slice])

    self.relu_op = g.get_operation_by_name('conv1/Relu')
    self.relu_op_slice = orm.OpSlice(self.relu_op, orm.Slice(0, 5))
    self.relu_op_group = orm.OpGroup(self.relu_op_slice)

    self.gamma_op = g.get_operation_by_name('conv1/BatchNorm/gamma/read')
    self.gamma_op_slice = orm.OpSlice(self.gamma_op, orm.Slice(0, 5))

    self.beta_op = g.get_operation_by_name('conv1/BatchNorm/beta/read')
    self.beta_op_slice = orm.OpSlice(self.beta_op, orm.Slice(0, 5))

    self.mean_op = g.get_operation_by_name(
        'conv1/BatchNorm/AssignMovingAvg/sub_1')
    self.mean_op_slice = orm.OpSlice(self.mean_op, orm.Slice(0, 5))

    self.std_op = g.get_operation_by_name(
        'conv1/BatchNorm/AssignMovingAvg_1/sub_1')
    self.std_op_slice = orm.OpSlice(self.std_op, orm.Slice(0, 5))

    # Create mock OpRegularizerManager with custom mapping of OpSlice and
    # OpGroup.
    self.mock_op_reg_manager = mock.create_autospec(orm.OpRegularizerManager)

    self.op_slice_dict = {
        self.batch_norm_op: [self.batch_norm_op_slice],
        self.conv_op: [self.conv_op_slice],
        self.relu_op: [self.relu_op_slice],
        self.gamma_op: [self.gamma_op_slice],
        self.beta_op: [self.beta_op_slice],
        self.mean_op: [self.mean_op_slice],
        self.std_op: [self.std_op_slice],
    }
    def get_op_slices(op):
      return self.op_slice_dict.get(op)

    def get_op_group(op_slice):
      return self.op_group_dict.get(op_slice)

    self.mock_op_reg_manager.get_op_slices.side_effect = get_op_slices
    self.mock_op_reg_manager.get_op_group.side_effect = get_op_group
    self.mock_op_reg_manager.is_source_op.return_value = False
    self.mock_op_reg_manager.ops = [
        self.batch_norm_op, self.conv_op, self.relu_op, self.gamma_op,
        self.beta_op, self.mean_op, self.std_op]

  def testAssignGrouping_NoNeighborGroups(self):
    # No ops have groups.
    self.op_group_dict = {}

    # Call handler to assign grouping.
    handler = leaf_op_handler.LeafOpHandler()
    handler.assign_grouping(self.batch_norm_op, self.mock_op_reg_manager)

    # Verify manager looks up OpSlice for ops of interest.
    self.mock_op_reg_manager.get_op_slices.assert_has_calls(
        # Checking for ops to process.
        [mock.call(self.relu_op),
         mock.call(self.mean_op),
         mock.call(self.std_op),
         # Initial slice data.
         mock.call(self.batch_norm_op),
         mock.call(self.relu_op),
         mock.call(self.mean_op),
         mock.call(self.std_op),
         # Reslicing.
         mock.call(self.batch_norm_op),
         mock.call(self.relu_op),
         mock.call(self.mean_op),
         mock.call(self.std_op),
         # Refreshing slice data.
         mock.call(self.relu_op),
         mock.call(self.mean_op),
         mock.call(self.std_op)])

    # Verify manager groups leaf op.
    self.mock_op_reg_manager.group_op_slices.assert_called_once_with(
        [self.batch_norm_op_slice])

    # Verify manager processes grouping for batch norm and output ops.
    self.mock_op_reg_manager.process_ops.assert_called_once_with(
        [self.relu_op, self.mean_op, self.std_op])
    self.mock_op_reg_manager.process_ops_last.assert_not_called()

  def testAssignGrouping_AllOutputsGrouped(self):
    # All outputs have groups.
    self.op_group_dict = {
        self.batch_norm_op_slice: self.batch_norm_op_group,
        self.relu_op_slice: self.relu_op_group,
        self.mean_op_slice: self.relu_op_group,
        self.std_op_slice: self.relu_op_group,
    }

    # Call handler to assign grouping.
    handler = leaf_op_handler.LeafOpHandler()
    handler.assign_grouping(self.batch_norm_op, self.mock_op_reg_manager)

    # Verify manager looks up OpSlice for ops of interest.
    self.mock_op_reg_manager.get_op_slices.assert_has_calls(
        # Checking for ops to process.
        [mock.call(self.relu_op),
         mock.call(self.mean_op),
         mock.call(self.std_op),
         # Initial slice data.
         mock.call(self.batch_norm_op),
         mock.call(self.relu_op),
         mock.call(self.mean_op),
         mock.call(self.std_op),
         # Reslicing.
         mock.call(self.batch_norm_op),
         mock.call(self.relu_op),
         mock.call(self.mean_op),
         mock.call(self.std_op),
         # Refreshing slice data.
         mock.call(self.relu_op),
         mock.call(self.mean_op),
         mock.call(self.std_op)])

    # Verify manager groups batch norm and outputs.
    self.mock_op_reg_manager.group_op_slices.assert_called_once_with(
        [self.batch_norm_op_slice, self.relu_op_slice, self.mean_op_slice,
         self.std_op_slice])

    # Verify manager processes all neighbors.
    self.mock_op_reg_manager.process_ops.assert_not_called()
    self.mock_op_reg_manager.process_ops_last.assert_not_called()

  def testAssignGrouping_AllNeighborsGrouped(self):
    # All neighbor ops have groups.
    self.op_group_dict = {
        self.batch_norm_op_slice: self.batch_norm_op_group,
        self.conv_op_slice: self.conv_op_group,
        self.relu_op_slice: self.relu_op_group,
        self.gamma_op_slice: self.conv_op_group,
        self.beta_op_slice: self.conv_op_group,
        self.mean_op_slice: self.relu_op_group,
        self.std_op_slice: self.relu_op_group,
    }

    # Call handler to assign grouping.
    handler = leaf_op_handler.LeafOpHandler()
    handler.assign_grouping(self.batch_norm_op, self.mock_op_reg_manager)

    # Verify manager looks up OpSlice for ops of interest.
    self.mock_op_reg_manager.get_op_slices.assert_has_calls(
        # Checking for ops to process.
        [mock.call(self.relu_op),
         mock.call(self.mean_op),
         mock.call(self.std_op),
         # Initial slice data.
         mock.call(self.batch_norm_op),
         mock.call(self.relu_op),
         mock.call(self.mean_op),
         mock.call(self.std_op),
         # Reslicing.
         mock.call(self.batch_norm_op),
         mock.call(self.relu_op),
         mock.call(self.mean_op),
         mock.call(self.std_op),
         # Refreshing slice data.
         mock.call(self.relu_op),
         mock.call(self.mean_op),
         mock.call(self.std_op),
         # Group batch norm op.
         mock.call(self.batch_norm_op)])

    # Verify manager groups batch norm and outputs.
    self.mock_op_reg_manager.group_op_slices.assert_called_once_with(
        [self.batch_norm_op_slice, self.relu_op_slice, self.mean_op_slice,
         self.std_op_slice])

    # Verify manager does not process any additional ops.
    self.mock_op_reg_manager.process_ops.assert_not_called()
    self.mock_op_reg_manager.process_ops_last.assert_not_called()

  def testAssignGrouping_AllNeighborsGroupedSameGroup(self):
    # All neighbor ops have same group as batch norm.
    self.op_group_dict = {
        self.batch_norm_op_slice: self.batch_norm_op_group,
        self.conv_op_slice: self.batch_norm_op_group,
        self.relu_op_slice: self.batch_norm_op_group,
        self.gamma_op_slice: self.batch_norm_op_group,
        self.beta_op_slice: self.batch_norm_op_group,
        self.mean_op_slice: self.batch_norm_op_group,
        self.std_op_slice: self.batch_norm_op_group,
    }

    # Call handler to assign grouping.
    handler = leaf_op_handler.LeafOpHandler()
    handler.assign_grouping(self.batch_norm_op, self.mock_op_reg_manager)

    # Verify manager looks up OpSlice for ops of interest.
    self.mock_op_reg_manager.get_op_slices.assert_has_calls(
        # Checking for ops to process.
        [mock.call(self.relu_op),
         mock.call(self.mean_op),
         mock.call(self.std_op),
         # Initial slice data.
         mock.call(self.batch_norm_op),
         mock.call(self.relu_op),
         mock.call(self.mean_op),
         mock.call(self.std_op),
         # Reslicing.
         mock.call(self.batch_norm_op),
         mock.call(self.relu_op),
         mock.call(self.mean_op),
         mock.call(self.std_op),
         # Refreshing slice data.
         mock.call(self.relu_op),
         mock.call(self.mean_op),
         mock.call(self.std_op),
         # Group batch norm op.
         mock.call(self.batch_norm_op)])

    # Verify manager doesn't perform any additional grouping.
    self.mock_op_reg_manager.group_op_slices.assert_not_called()

    # Verify manager does not process any additional ops.
    self.mock_op_reg_manager.process_ops.assert_not_called()
    self.mock_op_reg_manager.process_ops_last.assert_not_called()

  def testAssignGrouping_NonPassthroughOutputsSkipped(self):
    # Designate ReLU as non-passthrough for this test to demonstrate that batch
    # norm op does not group with ReLU.
    def is_passthrough(op):
      if op == self.relu_op:
        return False
      return True

    self.mock_op_reg_manager.is_passthrough.side_effect = is_passthrough

    # All neighbor ops have groups.
    self.op_group_dict = {
        self.batch_norm_op_slice: self.batch_norm_op_group,
        self.conv_op_slice: self.conv_op_group,
        self.relu_op_slice: self.relu_op_group,
        self.gamma_op_slice: self.conv_op_group,
        self.beta_op_slice: self.conv_op_group,
        self.mean_op_slice: self.relu_op_group,
        self.std_op_slice: self.relu_op_group,
    }

    # Call handler to assign grouping.
    handler = leaf_op_handler.LeafOpHandler()
    handler.assign_grouping(self.batch_norm_op, self.mock_op_reg_manager)

    # Verify manager looks up OpSlice for ops of interest.
    self.mock_op_reg_manager.get_op_slices.assert_has_calls(
        # Checking for ops to process.
        [mock.call(self.relu_op),
         mock.call(self.mean_op),
         mock.call(self.std_op),
         # Initial slice data.
         mock.call(self.batch_norm_op),
         mock.call(self.mean_op),
         mock.call(self.std_op),
         # Reslicing.
         mock.call(self.batch_norm_op),
         mock.call(self.mean_op),
         mock.call(self.std_op),
         # Refreshing slice data.
         mock.call(self.mean_op),
         mock.call(self.std_op),
         # Group batch norm op.
         mock.call(self.batch_norm_op)])

    # Verify manager groups batch norm and outputs.  ReLU is not part of the
    # grouping.
    self.mock_op_reg_manager.group_op_slices.assert_called_once_with(
        [self.batch_norm_op_slice, self.mean_op_slice, self.std_op_slice])

    # Verify manager does not process any additional ops.
    self.mock_op_reg_manager.process_ops.assert_not_called()
    self.mock_op_reg_manager.process_ops_last.assert_not_called()


if __name__ == '__main__':
  tf.test.main()
