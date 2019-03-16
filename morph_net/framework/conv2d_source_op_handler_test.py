"""Tests for regularizers.framework.conv2d_source_op_handler."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import mock
from morph_net.framework import conv2d_source_op_handler
from morph_net.framework import op_regularizer_manager as orm
import tensorflow as tf

layers = tf.contrib.layers

_DEFAULT_THRESHOLD = 0.001


class Conv2dSourceOpHandlerTest(tf.test.TestCase):

  def setUp(self):
    tf.reset_default_graph()

    # This tests 2 Conv2D ops.
    inputs = tf.zeros([2, 4, 4, 3])
    c1 = layers.conv2d(inputs, num_outputs=5,
                       kernel_size=3, scope='conv1', normalizer_fn=None)
    layers.conv2d(c1, num_outputs=6, kernel_size=3, scope='conv2',
                  normalizer_fn=None)

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

    self.conv2_weights_op = g.get_operation_by_name('conv2/weights/read')
    self.conv2_weights_op_slice = orm.OpSlice(
        self.conv2_weights_op, orm.Slice(0, 6))
    self.conv2_weights_op_group = orm.OpGroup(
        self.conv2_weights_op_slice,
        omit_source_op_slices=[self.conv2_weights_op_slice])

    self.relu2_op = g.get_operation_by_name('conv2/Relu')
    self.relu2_op_slice = orm.OpSlice(self.relu2_op, orm.Slice(0, 6))
    self.relu2_op_group = orm.OpGroup(
        self.relu2_op_slice, omit_source_op_slices=[self.relu2_op_slice])

    # Create mock OpRegularizerManager with custom mapping of OpSlice and
    # OpGroup.
    self.mock_op_reg_manager = mock.create_autospec(orm.OpRegularizerManager)

    def get_op_slices(op):
      return self.op_slice_dict.get(op, [])

    def get_op_group(op_slice):
      return self.op_group_dict.get(op_slice)

    def is_passthrough(op):
      if op in [self.conv1_op, self.conv2_op]:
        h = conv2d_source_op_handler.Conv2DSourceOpHandler(_DEFAULT_THRESHOLD)
        return h.is_passthrough
      else:
        return False

    self.mock_op_reg_manager.get_op_slices.side_effect = get_op_slices
    self.mock_op_reg_manager.get_op_group.side_effect = get_op_group
    self.mock_op_reg_manager.is_source_op.return_value = False
    self.mock_op_reg_manager.is_passthrough.side_effect = is_passthrough
    self.mock_op_reg_manager.ops = [
        self.conv1_op, self.relu1_op, self.conv2_op, self.relu2_op,
        self.conv2_weights_op]

  def testAssignGrouping_GroupWithOutputOnly(self):
    # Map ops to slices.
    self.op_slice_dict = {
        self.conv1_op: [self.conv1_op_slice],
        self.relu1_op: [self.relu1_op_slice],
        self.conv2_op: [self.conv2_op_slice],
        self.relu2_op: [self.relu2_op_slice],
    }

    # Map each slice to a group. Corresponding op slices have the same group.
    self.op_group_dict = {
        self.conv2_op_slice: self.conv2_op_group,
    }

    # Call handler to assign grouping.
    handler = conv2d_source_op_handler.Conv2DSourceOpHandler(_DEFAULT_THRESHOLD)
    handler.assign_grouping(self.conv2_op, self.mock_op_reg_manager)

    # Verify manager looks up op slice for ops of interest.
    self.mock_op_reg_manager.get_op_slices.assert_any_call(self.conv2_op)

    # Verify manager does not slice any ops.
    self.mock_op_reg_manager.slice_op.assert_not_called()

    # Verify manager adds inputs to process queue.
    self.mock_op_reg_manager.process_ops.assert_called_once_with(
        [self.relu1_op])

  def testCreateRegularizer(self):
    # Call handler to create regularizer.
    handler = conv2d_source_op_handler.Conv2DSourceOpHandler(_DEFAULT_THRESHOLD)
    regularizer = handler.create_regularizer(self.conv2_op_slice)

    # Verify regularizer produces correctly shaped tensors.
    # Most of the regularizer testing is in group_lasso_regularizer_test.py
    expected_norm_dim = self.conv2_op.inputs[1].shape.as_list()[-1]
    self.assertEqual(expected_norm_dim,
                     regularizer.regularization_vector.shape.as_list()[0])

  def testCreateRegularizer_Sliced(self):
    # Call handler to create regularizer.
    handler = conv2d_source_op_handler.Conv2DSourceOpHandler(_DEFAULT_THRESHOLD)
    conv2_op_slice = orm.OpSlice(self.conv2_op, orm.Slice(0, 3))
    regularizer = handler.create_regularizer(conv2_op_slice)

    # Verify regularizer produces correctly shaped tensors.
    # Most of the regularizer testing is in group_lasso_regularizer_test.py
    expected_norm_dim = 3
    self.assertEqual(expected_norm_dim,
                     regularizer.regularization_vector.shape.as_list()[0])


if __name__ == '__main__':
  tf.test.main()
