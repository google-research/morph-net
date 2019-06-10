"""Tests for BatchNormSourceOpHandler."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import mock
from morph_net.framework import batch_norm_source_op_handler
from morph_net.framework import op_regularizer_manager as orm
import tensorflow as tf

arg_scope = tf.contrib.framework.arg_scope
layers = tf.contrib.layers

_GAMMA_THRESHOLD = 0.001


class BatchNormSourceOpHandlerTest(tf.test.TestCase):

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

    # Declare OpSlice and OpGroup for ops that are created in the test network.
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
    self.gamma_op_group = orm.OpGroup(
        self.gamma_op_slice, omit_source_op_slices=[self.gamma_op_slice])

    self.beta_op = g.get_operation_by_name('conv1/BatchNorm/beta/read')
    self.beta_op_slice = orm.OpSlice(self.beta_op, orm.Slice(0, 5))
    self.beta_op_group = orm.OpGroup(
        self.beta_op_slice, omit_source_op_slices=[self.beta_op_slice])

    self.mean_op = g.get_operation_by_name(
        'conv1/BatchNorm/AssignMovingAvg/sub_1')
    self.mean_op_slice = orm.OpSlice(self.mean_op, orm.Slice(0, 5))
    self.mean_op_group = orm.OpGroup(
        self.mean_op_slice, omit_source_op_slices=[self.mean_op_slice])

    self.std_op = g.get_operation_by_name(
        'conv1/BatchNorm/AssignMovingAvg_1/sub_1')
    self.std_op_slice = orm.OpSlice(self.std_op, orm.Slice(0, 5))
    self.std_op_group = orm.OpGroup(
        self.std_op_slice, omit_source_op_slices=[self.std_op_slice])

    # Create custom mapping of OpSlice and OpGroup in manager.
    self.mock_op_reg_manager = mock.create_autospec(orm.OpRegularizerManager)

    def get_op_slices(op):
      return self.op_slice_dict.get(op, [])

    def get_op_group(op_slice):
      return self.op_group_dict.get(op_slice)

    self.mock_op_reg_manager.get_op_slices.side_effect = get_op_slices
    self.mock_op_reg_manager.get_op_group.side_effect = get_op_group
    self.mock_op_reg_manager.is_source_op.return_value = False
    self.mock_op_reg_manager.ops = [
        self.batch_norm_op, self.conv_op, self.relu_op, self.gamma_op,
        self.beta_op, self.mean_op, self.std_op]

  def testAssignGrouping_NoNeighborGroups(self):
    self.op_slice_dict = {
        self.batch_norm_op: [self.batch_norm_op_slice],
        self.conv_op: [self.conv_op_slice],
        self.relu_op: [self.relu_op_slice],
        self.gamma_op: [self.gamma_op_slice],
        self.beta_op: [self.beta_op_slice],
        self.mean_op: [self.mean_op_slice],
        self.std_op: [self.std_op_slice],
    }

    # No neighbor ops have groups.
    self.op_group_dict = {
        self.batch_norm_op_slice: self.batch_norm_op_group,
    }

    # Call handler to assign grouping.
    handler = batch_norm_source_op_handler.BatchNormSourceOpHandler(
        _GAMMA_THRESHOLD)
    handler.assign_grouping(self.batch_norm_op, self.mock_op_reg_manager)

    # Verify manager looks up op slice for ops of interest.
    self.mock_op_reg_manager.get_op_slices.assert_any_call(self.batch_norm_op)
    self.mock_op_reg_manager.get_op_slices.assert_any_call(self.conv_op)
    self.mock_op_reg_manager.get_op_slices.assert_any_call(self.relu_op)

    # Verify manager creates OpGroup for batch norm op.
    self.mock_op_reg_manager.create_op_group_for_op_slice(
        self.batch_norm_op_slice)

    # Verify manager groups batch norm with outputs and inputs.
    self.mock_op_reg_manager.group_op_slices.assert_not_called()

    # Verify manager processes grouping for input and output ops.
    self.mock_op_reg_manager.process_ops.assert_called_once_with(
        [self.relu_op, self.mean_op, self.std_op, self.conv_op, self.gamma_op,
         self.beta_op])
    self.mock_op_reg_manager.process_ops_last.assert_called_once_with(
        [self.batch_norm_op])

  def testAssignGrouping_ProcessNeighborGroups(self):
    self.op_slice_dict = {
        self.batch_norm_op: [self.batch_norm_op_slice],
        self.conv_op: [self.conv_op_slice],
        self.relu_op: [self.relu_op_slice],
        self.gamma_op: [self.gamma_op_slice],
        self.beta_op: [self.beta_op_slice],
        self.mean_op: [self.mean_op_slice],
        self.std_op: [self.std_op_slice],
    }

    # All ops have groups.
    self.op_group_dict = {
        self.batch_norm_op_slice: self.batch_norm_op_group,
        self.conv_op_slice: self.conv_op_group,
        self.relu_op_slice: self.relu_op_group,
        self.gamma_op_slice: self.gamma_op_group,
        self.beta_op_slice: self.beta_op_group,
        self.mean_op_slice: self.mean_op_group,
        self.std_op_slice: self.std_op_group,
    }

    # Call handler to assign grouping.
    handler = batch_norm_source_op_handler.BatchNormSourceOpHandler(
        _GAMMA_THRESHOLD)
    handler.assign_grouping(self.batch_norm_op, self.mock_op_reg_manager)

    # Verify manager looks up op slice for ops of interest.
    self.mock_op_reg_manager.get_op_slices.assert_any_call(self.batch_norm_op)
    self.mock_op_reg_manager.get_op_slices.assert_any_call(self.conv_op)
    self.mock_op_reg_manager.get_op_slices.assert_any_call(self.relu_op)

    # Verify manager groups batch norm with outputs and inputs.
    self.mock_op_reg_manager.group_op_slices.assert_has_calls(
        [mock.call([self.batch_norm_op_slice, self.relu_op_slice,
                    self.mean_op_slice, self.std_op_slice]),
         mock.call([self.batch_norm_op_slice, self.conv_op_slice,
                    self.gamma_op_slice, self.beta_op_slice],
                   omit_source_op_slices=[])])

    # Verify manager does not reprocess any ops.
    self.mock_op_reg_manager.process_ops.assert_not_called()
    self.mock_op_reg_manager.process_ops_last.assert_not_called()

  def testAssignGrouping_ProcessNeighborGroupsWithSlices(self):
    batch_norm_op_slice_0_2 = orm.OpSlice(self.batch_norm_op, orm.Slice(0, 2))
    batch_norm_op_slice_2_3 = orm.OpSlice(self.batch_norm_op, orm.Slice(2, 1))
    batch_norm_op_group1 = orm.OpGroup(batch_norm_op_slice_0_2)
    batch_norm_op_group2 = orm.OpGroup(batch_norm_op_slice_2_3)

    conv_op_slice_0_2 = orm.OpSlice(self.conv_op, orm.Slice(0, 2))
    conv_op_slice_2_3 = orm.OpSlice(self.conv_op, orm.Slice(2, 1))
    conv_op_group1 = orm.OpGroup(
        conv_op_slice_0_2, omit_source_op_slices=[conv_op_slice_0_2])
    conv_op_group2 = orm.OpGroup(
        conv_op_slice_2_3, omit_source_op_slices=[conv_op_slice_2_3])

    relu_op_slice_0_2 = orm.OpSlice(self.relu_op, orm.Slice(0, 2))
    relu_op_slice_2_3 = orm.OpSlice(self.relu_op, orm.Slice(2, 1))
    relu_op_group1 = orm.OpGroup(relu_op_slice_0_2)
    relu_op_group2 = orm.OpGroup(relu_op_slice_2_3)

    gamma_op_slice_0_2 = orm.OpSlice(self.gamma_op, orm.Slice(0, 2))
    gamma_op_slice_2_3 = orm.OpSlice(self.gamma_op, orm.Slice(2, 1))
    gamma_op_group1 = orm.OpGroup(
        gamma_op_slice_0_2, omit_source_op_slices=[gamma_op_slice_0_2])
    gamma_op_group2 = orm.OpGroup(
        gamma_op_slice_2_3, omit_source_op_slices=[gamma_op_slice_2_3])

    beta_op_slice_0_2 = orm.OpSlice(self.beta_op, orm.Slice(0, 2))
    beta_op_slice_2_3 = orm.OpSlice(self.beta_op, orm.Slice(2, 1))
    beta_op_group1 = orm.OpGroup(
        beta_op_slice_0_2, omit_source_op_slices=[beta_op_slice_0_2])
    beta_op_group2 = orm.OpGroup(
        beta_op_slice_2_3, omit_source_op_slices=[beta_op_slice_2_3])

    mean_op_slice_0_2 = orm.OpSlice(self.mean_op, orm.Slice(0, 2))
    mean_op_slice_2_3 = orm.OpSlice(self.mean_op, orm.Slice(2, 1))
    mean_op_group1 = orm.OpGroup(
        mean_op_slice_0_2, omit_source_op_slices=[mean_op_slice_0_2])
    mean_op_group2 = orm.OpGroup(
        mean_op_slice_2_3, omit_source_op_slices=[mean_op_slice_2_3])

    std_op_slice_0_2 = orm.OpSlice(self.std_op, orm.Slice(0, 2))
    std_op_slice_2_3 = orm.OpSlice(self.std_op, orm.Slice(2, 1))
    std_op_group1 = orm.OpGroup(
        std_op_slice_0_2, omit_source_op_slices=[std_op_slice_0_2])
    std_op_group2 = orm.OpGroup(
        std_op_slice_2_3, omit_source_op_slices=[std_op_slice_2_3])

    self.op_slice_dict = {
        self.batch_norm_op: [batch_norm_op_slice_0_2, batch_norm_op_slice_2_3],
        self.conv_op: [conv_op_slice_0_2, conv_op_slice_2_3],
        self.relu_op: [relu_op_slice_0_2, relu_op_slice_2_3],
        self.gamma_op: [gamma_op_slice_0_2, gamma_op_slice_2_3],
        self.beta_op: [beta_op_slice_0_2, beta_op_slice_2_3],
        self.mean_op: [mean_op_slice_0_2, mean_op_slice_2_3],
        self.std_op: [std_op_slice_0_2, std_op_slice_2_3],
    }

    # All OpSlice have groups.
    self.op_group_dict = {
        batch_norm_op_slice_0_2: batch_norm_op_group1,
        batch_norm_op_slice_2_3: batch_norm_op_group2,
        conv_op_slice_0_2: conv_op_group1,
        conv_op_slice_2_3: conv_op_group2,
        relu_op_slice_0_2: relu_op_group1,
        relu_op_slice_2_3: relu_op_group2,
        gamma_op_slice_0_2: gamma_op_group1,
        gamma_op_slice_2_3: gamma_op_group2,
        beta_op_slice_0_2: beta_op_group1,
        beta_op_slice_2_3: beta_op_group2,
        mean_op_slice_0_2: mean_op_group1,
        mean_op_slice_2_3: mean_op_group2,
        std_op_slice_0_2: std_op_group1,
        std_op_slice_2_3: std_op_group2,
    }

    # Call handler to assign grouping.
    handler = batch_norm_source_op_handler.BatchNormSourceOpHandler(
        _GAMMA_THRESHOLD)
    handler.assign_grouping(self.batch_norm_op, self.mock_op_reg_manager)

    # Verify manager looks up op slice for ops of interest.
    self.mock_op_reg_manager.get_op_slices.assert_any_call(self.batch_norm_op)
    self.mock_op_reg_manager.get_op_slices.assert_any_call(self.conv_op)
    self.mock_op_reg_manager.get_op_slices.assert_any_call(self.relu_op)

    # Verify manager groups batch norm with outputs and inputs by slice.
    self.mock_op_reg_manager.group_op_slices.assert_has_calls(
        [mock.call([batch_norm_op_slice_0_2, relu_op_slice_0_2,
                    mean_op_slice_0_2, std_op_slice_0_2]),
         mock.call([batch_norm_op_slice_0_2, conv_op_slice_0_2,
                    gamma_op_slice_0_2, beta_op_slice_0_2],
                   omit_source_op_slices=[]),
         mock.call([batch_norm_op_slice_2_3, relu_op_slice_2_3,
                    mean_op_slice_2_3, std_op_slice_2_3]),
         mock.call([batch_norm_op_slice_2_3, conv_op_slice_2_3,
                    gamma_op_slice_2_3, beta_op_slice_2_3],
                   omit_source_op_slices=[])])

    # Verify manager does not reprocess any ops.
    self.mock_op_reg_manager.process_ops.assert_not_called()
    self.mock_op_reg_manager.process_ops_last.assert_not_called()

  def testAssignGrouping_NeighborsHaveSameGroup(self):
    self.op_slice_dict = {
        self.batch_norm_op: [self.batch_norm_op_slice],
        self.conv_op: [self.batch_norm_op_slice],
        self.relu_op: [self.batch_norm_op_slice],
        self.gamma_op: [self.batch_norm_op_slice],
        self.beta_op: [self.batch_norm_op_slice],
        self.mean_op: [self.batch_norm_op_slice],
        self.std_op: [self.batch_norm_op_slice],
    }

    # All ops have the same group.
    self.op_group_dict = {
        self.batch_norm_op_slice: self.batch_norm_op_group,
        self.conv_op_slice: self.batch_norm_op_group,
        self.relu_op_slice: self.batch_norm_op_group,
        self.gamma_op_slice: self.batch_norm_op_group,
        self.beta_op_slice: self.batch_norm_op_group,
    }

    # Call handler to assign grouping.
    handler = batch_norm_source_op_handler.BatchNormSourceOpHandler(
        _GAMMA_THRESHOLD)
    handler.assign_grouping(self.batch_norm_op, self.mock_op_reg_manager)

    # Verify manager looks up op slice for ops of interest.
    self.mock_op_reg_manager.get_op_slices.assert_any_call(self.batch_norm_op)
    self.mock_op_reg_manager.get_op_slices.assert_any_call(self.conv_op)
    self.mock_op_reg_manager.get_op_slices.assert_any_call(self.relu_op)

    # Verify manager does not group any ops.
    self.mock_op_reg_manager.group_op_slices.assert_not_called()

    # Verify manager does not process additional ops.
    self.mock_op_reg_manager.process_ops.assert_not_called()
    self.mock_op_reg_manager.process_ops_last.assert_not_called()

  def testAssignGrouping_NeighborsHaveSameGroup_ReprocessSources(self):
    source_conv_op_group = orm.OpGroup(self.conv_op_slice)

    self.op_slice_dict = {
        self.batch_norm_op: [self.batch_norm_op_slice],
        self.conv_op: [self.conv_op_slice],
        self.relu_op: [self.relu_op_slice],
        self.gamma_op: [self.gamma_op_slice],
        self.beta_op: [self.beta_op_slice],
        self.mean_op: [self.mean_op_slice],
        self.std_op: [self.std_op_slice],
    }

    self.op_group_dict = {
        self.batch_norm_op_slice: self.batch_norm_op_group,
        self.conv_op_slice: source_conv_op_group,
        self.relu_op_slice: self.batch_norm_op_group,
        self.gamma_op_slice: self.batch_norm_op_group,
        self.beta_op_slice: self.batch_norm_op_group,
    }

    source_ops = (self.conv_op,)
    def is_source_op(op):
      return op in source_ops

    self.mock_op_reg_manager.is_source_op.side_effect = is_source_op

    # Call handler to assign grouping.
    handler = batch_norm_source_op_handler.BatchNormSourceOpHandler(
        _GAMMA_THRESHOLD)
    handler.assign_grouping(self.batch_norm_op, self.mock_op_reg_manager)

    # Verify manager looks up op slice for ops of interest.
    self.mock_op_reg_manager.get_op_slices.assert_any_call(self.batch_norm_op)
    self.mock_op_reg_manager.get_op_slices.assert_any_call(self.conv_op)
    self.mock_op_reg_manager.get_op_slices.assert_any_call(self.relu_op)

    # Verify manager groups batch norm with inputs and overrides source.
    self.mock_op_reg_manager.group_op_slices.assert_called_once_with(
        [self.batch_norm_op_slice, self.conv_op_slice, self.gamma_op_slice,
         self.beta_op_slice],
        omit_source_op_slices=[self.conv_op_slice])

    # Verify manager adds ungrouped output ops to queue.
    self.mock_op_reg_manager.process_ops.assert_called_once_with(
        [self.mean_op, self.std_op])
    self.mock_op_reg_manager.process_ops_last.assert_not_called()

  def testCreateRegularizer(self):
    # Call handler to create regularizer.
    handler = batch_norm_source_op_handler.BatchNormSourceOpHandler(
        _GAMMA_THRESHOLD)
    regularizer = handler.create_regularizer(self.batch_norm_op_slice)

    # Verify regularizer is the gamma tensor.
    g = tf.get_default_graph()
    gamma_tensor = g.get_tensor_by_name('conv1/BatchNorm/gamma/read:0')
    self.assertEqual(gamma_tensor, regularizer._gamma)

  def testCreateRegularizer_Sliced(self):
    # Call handler to create regularizer.
    handler = batch_norm_source_op_handler.BatchNormSourceOpHandler(
        _GAMMA_THRESHOLD)
    batch_norm_op_slice = orm.OpSlice(self.batch_norm_op, orm.Slice(0, 3))
    regularizer = handler.create_regularizer(batch_norm_op_slice)

    # Verify regularizer is the gamma tensor.
    with self.cached_session():
      # Initialize the gamma tensor to check value equality.
      with tf.variable_scope('', reuse=tf.AUTO_REUSE):
        gamma_tensor = tf.get_variable('conv1/BatchNorm/gamma')
      init = tf.variables_initializer([gamma_tensor])
      init.run()

      # Verify regularizer is the sliced gamma tensor.
      self.assertAllEqual(gamma_tensor.eval()[0:3],
                          regularizer._gamma.eval())


if __name__ == '__main__':
  tf.test.main()
