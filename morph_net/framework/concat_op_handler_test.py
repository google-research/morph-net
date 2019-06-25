"""Tests for concat_op_handler."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import mock
from morph_net.framework import concat_op_handler
from morph_net.framework import op_regularizer_manager as orm
import tensorflow as tf

layers = tf.contrib.layers
arg_scope = tf.contrib.framework.arg_scope


class ConcatOpHandlerTest(tf.test.TestCase):

  def setUp(self):
    tf.reset_default_graph()

    # This tests 3 Conv2D ops being concatenated.
    inputs = tf.zeros([2, 4, 4, 3])
    c1 = layers.conv2d(inputs, num_outputs=5, kernel_size=3, scope='conv1')
    c2 = layers.conv2d(inputs, num_outputs=6, kernel_size=3, scope='conv2')
    c3 = layers.conv2d(inputs, num_outputs=7, kernel_size=3, scope='conv3')
    net = tf.concat([c1, c2, c3], axis=3)
    layers.batch_norm(net)

    g = tf.get_default_graph()

    # Declare OpSlice and OpGroup for ops of interest.
    self.concat_op = g.get_operation_by_name('concat')
    self.concat_op_slice = orm.OpSlice(self.concat_op, orm.Slice(0, 18))
    self.concat_op_slice_0_5 = orm.OpSlice(self.concat_op, orm.Slice(0, 5))
    self.concat_op_slice_5_11 = orm.OpSlice(self.concat_op, orm.Slice(5, 6))
    self.concat_op_slice_11_18 = orm.OpSlice(self.concat_op, orm.Slice(11, 7))
    self.concat_op_group1 = orm.OpGroup(
        self.concat_op_slice_0_5,
        omit_source_op_slices=[self.concat_op_slice_0_5])
    self.concat_op_group2 = orm.OpGroup(
        self.concat_op_slice_5_11,
        omit_source_op_slices=[self.concat_op_slice_5_11])
    self.concat_op_group3 = orm.OpGroup(
        self.concat_op_slice_11_18,
        omit_source_op_slices=[self.concat_op_slice_11_18])

    self.relu1_op = g.get_operation_by_name('conv1/Relu')
    self.relu1_op_slice = orm.OpSlice(self.relu1_op, orm.Slice(0, 5))
    self.relu1_op_group = orm.OpGroup(
        self.relu1_op_slice, omit_source_op_slices=[self.relu1_op_slice])

    self.relu2_op = g.get_operation_by_name('conv2/Relu')
    self.relu2_op_slice = orm.OpSlice(self.relu2_op, orm.Slice(0, 6))
    self.relu2_op_group = orm.OpGroup(
        self.relu2_op_slice, omit_source_op_slices=[self.relu2_op_slice])

    self.relu3_op = g.get_operation_by_name('conv3/Relu')
    self.relu3_op_slice = orm.OpSlice(self.relu3_op, orm.Slice(0, 7))
    self.relu3_op_group = orm.OpGroup(
        self.relu3_op_slice, omit_source_op_slices=[self.relu3_op_slice])

    self.axis_op = g.get_operation_by_name('concat/axis')

    self.batch_norm_op = g.get_operation_by_name('BatchNorm/FusedBatchNormV3')
    self.batch_norm_op_slice = orm.OpSlice(self.batch_norm_op, orm.Slice(0, 18))
    self.batch_norm_op_group = orm.OpGroup(
        self.batch_norm_op_slice,
        omit_source_op_slices=[self.batch_norm_op_slice])
    self.batch_norm_op_slice_0_5 = orm.OpSlice(
        self.batch_norm_op, orm.Slice(0, 5))
    self.batch_norm_op_slice_5_11 = orm.OpSlice(
        self.batch_norm_op, orm.Slice(5, 6))
    self.batch_norm_op_slice_11_18 = orm.OpSlice(
        self.batch_norm_op, orm.Slice(11, 7))
    self.batch_norm_op_group1 = orm.OpGroup(
        self.batch_norm_op_slice_0_5,
        omit_source_op_slices=[self.batch_norm_op_slice_0_5])
    self.batch_norm_op_group2 = orm.OpGroup(
        self.batch_norm_op_slice_5_11,
        omit_source_op_slices=[self.batch_norm_op_slice_5_11])
    self.batch_norm_op_group3 = orm.OpGroup(
        self.batch_norm_op_slice_11_18,
        omit_source_op_slices=[self.batch_norm_op_slice_11_18])

    # Create mock OpRegularizerManager with custom mapping of OpSlice and
    # OpGroup.
    self.mock_op_reg_manager = mock.create_autospec(orm.OpRegularizerManager)

    def get_op_slices(op):
      return self.op_slice_dict.get(op, [])

    def get_op_group(op_slice):
      return self.op_group_dict.get(op_slice)

    # Update op_slice_dict when an op is sliced.
    def slice_op(op, _):
      if op == self.batch_norm_op:
        self.op_slice_dict[self.batch_norm_op] = [
            self.batch_norm_op_slice_0_5,
            self.batch_norm_op_slice_5_11,
            self.batch_norm_op_slice_11_18]
      if op == self.concat_op:
        self.op_slice_dict[self.concat_op] = [
            self.concat_op_slice_0_5,
            self.concat_op_slice_5_11,
            self.concat_op_slice_11_18]

    self.mock_op_reg_manager.get_op_slices.side_effect = get_op_slices
    self.mock_op_reg_manager.get_op_group.side_effect = get_op_group
    self.mock_op_reg_manager.is_source_op.return_value = False
    self.mock_op_reg_manager.slice_op.side_effect = slice_op
    self.mock_op_reg_manager.is_passthrough.return_value = True
    self.mock_op_reg_manager.ops = [
        self.concat_op, self.relu1_op, self.relu2_op, self.relu3_op,
        self.batch_norm_op]

  def testAssignGrouping_AllNeighborsGrouped_SlicesAligned(self):
    # In this test, the output op (batch norm) has size 18 and is sliced into
    # sizes [5, 6, 7] which matches the Conv2D sizes which are [5, 6, 7].

    # Map ops to slices.  Batch norm op is composed of multiple slices.
    self.op_slice_dict = {
        self.relu1_op: [self.relu1_op_slice],
        self.relu2_op: [self.relu2_op_slice],
        self.relu3_op: [self.relu3_op_slice],
        self.concat_op: [self.concat_op_slice],
        self.batch_norm_op: [self.batch_norm_op_slice_0_5,
                             self.batch_norm_op_slice_5_11,
                             self.batch_norm_op_slice_11_18],
    }

    # Map each slice to a group.
    self.op_group_dict = {
        self.relu1_op_slice: self.relu1_op_group,
        self.relu2_op_slice: self.relu2_op_group,
        self.relu3_op_slice: self.relu3_op_group,
        self.batch_norm_op_slice_0_5: self.batch_norm_op_group1,
        self.batch_norm_op_slice_5_11: self.batch_norm_op_group2,
        self.batch_norm_op_slice_11_18: self.batch_norm_op_group3,
    }

    # Call handler to assign grouping.
    handler = concat_op_handler.ConcatOpHandler()
    handler.assign_grouping(self.concat_op, self.mock_op_reg_manager)

    # Verify manager looks up OpSlice for ops of interest.
    self.mock_op_reg_manager.get_op_slices.assert_has_calls(
        # Checking for ops to process.
        [mock.call(self.relu1_op),
         mock.call(self.relu2_op),
         mock.call(self.relu3_op),
         mock.call(self.batch_norm_op),
         # Initial slice data.
         mock.call(self.concat_op),
         mock.call(self.relu1_op),
         mock.call(self.relu2_op),
         mock.call(self.relu3_op),
         mock.call(self.batch_norm_op),
         # Reslicing.
         mock.call(self.relu1_op),
         mock.call(self.relu2_op),
         mock.call(self.relu3_op),
         mock.call(self.batch_norm_op),
         mock.call(self.concat_op),
         # Refreshing slice data.
         mock.call(self.relu1_op),
         mock.call(self.relu2_op),
         mock.call(self.relu3_op),
         mock.call(self.batch_norm_op),
         # Group concat op.
         mock.call(self.concat_op)])

    # Verify manager only slices the concat op.
    self.mock_op_reg_manager.slice_op.assert_called_once_with(
        self.concat_op, [5, 6, 7])

    # Verify manager groups the new slices.
    self.mock_op_reg_manager.group_op_slices.assert_has_calls(
        [mock.call([self.concat_op_slice_0_5, self.relu1_op_slice,
                    self.batch_norm_op_slice_0_5]),
         mock.call([self.concat_op_slice_5_11, self.relu2_op_slice,
                    self.batch_norm_op_slice_5_11]),
         mock.call([self.concat_op_slice_11_18, self.relu3_op_slice,
                    self.batch_norm_op_slice_11_18])])

  def testAssignGrouping_AllNeighborsGrouped_SlicesAligned_SameGroup(self):
    # This test verifies that no slicing or grouping occurs.

    # Map ops to slices.  Batch norm op is composed of multiple slices.
    self.op_slice_dict = {
        self.relu1_op: [self.relu1_op_slice],
        self.relu2_op: [self.relu2_op_slice],
        self.relu3_op: [self.relu3_op_slice],
        self.concat_op: [self.concat_op_slice_0_5, self.concat_op_slice_5_11,
                         self.concat_op_slice_11_18],
        self.batch_norm_op: [self.batch_norm_op_slice_0_5,
                             self.batch_norm_op_slice_5_11,
                             self.batch_norm_op_slice_11_18],
    }

    # Map each slice to a group.  Corresponding op slices have the same group.
    self.op_group_dict = {
        self.relu1_op_slice: self.batch_norm_op_group1,
        self.relu2_op_slice: self.batch_norm_op_group2,
        self.relu3_op_slice: self.batch_norm_op_group3,
        self.concat_op_slice_0_5: self.batch_norm_op_group1,
        self.concat_op_slice_5_11: self.batch_norm_op_group2,
        self.concat_op_slice_11_18: self.batch_norm_op_group3,
        self.batch_norm_op_slice_0_5: self.batch_norm_op_group1,
        self.batch_norm_op_slice_5_11: self.batch_norm_op_group2,
        self.batch_norm_op_slice_11_18: self.batch_norm_op_group3,
    }

    # Call handler to assign grouping.
    handler = concat_op_handler.ConcatOpHandler()
    handler.assign_grouping(self.concat_op, self.mock_op_reg_manager)

    # Verify manager looks up OpSlice for ops of interest.
    self.mock_op_reg_manager.get_op_slices.assert_has_calls(
        # Checking for ops to process.
        [mock.call(self.relu1_op),
         mock.call(self.relu2_op),
         mock.call(self.relu3_op),
         mock.call(self.batch_norm_op),
         # Initial slice data.
         mock.call(self.concat_op),
         mock.call(self.relu1_op),
         mock.call(self.relu2_op),
         mock.call(self.relu3_op),
         mock.call(self.batch_norm_op),
         # Reslicing.
         mock.call(self.relu1_op),
         mock.call(self.relu2_op),
         mock.call(self.relu3_op),
         mock.call(self.batch_norm_op),
         mock.call(self.concat_op),
         # Refreshing slice data.
         mock.call(self.relu1_op),
         mock.call(self.relu2_op),
         mock.call(self.relu3_op),
         mock.call(self.batch_norm_op),
         # Group concat op.
         mock.call(self.concat_op)])

    # Verify manager does not slice any ops.
    self.mock_op_reg_manager.slice_op.assert_not_called()

    # Verify manager does not group any ops.
    self.mock_op_reg_manager.group_op_slices.assert_not_called()

  def testAssignGrouping_AllNeighborsGrouped_OutputSlicesNotAligned(self):
    # The output (batch norm) has sizes [9, 4, 5] which are not aligned.  This
    # test verifies that the concat, batch norm, and Conv2D ops are sliced in
    # alignment.
    concat_op_slice_0_5 = orm.OpSlice(self.concat_op, orm.Slice(0, 5))
    concat_op_slice_5_9 = orm.OpSlice(self.concat_op, orm.Slice(5, 4))
    concat_op_slice_9_11 = orm.OpSlice(self.concat_op, orm.Slice(9, 2))
    concat_op_slice_11_13 = orm.OpSlice(self.concat_op, orm.Slice(11, 2))
    concat_op_slice_13_18 = orm.OpSlice(self.concat_op, orm.Slice(13, 5))

    relu2_op_slice_0_4 = orm.OpSlice(self.relu2_op, orm.Slice(0, 4))
    relu2_op_slice_4_6 = orm.OpSlice(self.relu2_op, orm.Slice(4, 2))

    relu3_op_slice_0_2 = orm.OpSlice(self.relu3_op, orm.Slice(0, 2))
    relu3_op_slice_2_7 = orm.OpSlice(self.relu3_op, orm.Slice(2, 5))

    batch_norm_op_slice_0_9 = orm.OpSlice(self.batch_norm_op, orm.Slice(0, 9))
    batch_norm_op_group1 = orm.OpGroup(
        batch_norm_op_slice_0_9,
        omit_source_op_slices=[batch_norm_op_slice_0_9])
    batch_norm_op_slice_9_13 = orm.OpSlice(self.batch_norm_op, orm.Slice(9, 4))
    batch_norm_op_group2 = orm.OpGroup(
        batch_norm_op_slice_9_13,
        omit_source_op_slices=[batch_norm_op_slice_9_13])
    batch_norm_op_slice_13_18 = orm.OpSlice(
        self.batch_norm_op, orm.Slice(13, 5))
    batch_norm_op_group3 = orm.OpGroup(
        batch_norm_op_slice_13_18,
        omit_source_op_slices=[batch_norm_op_slice_13_18])
    batch_norm_op_slice_0_5 = orm.OpSlice(self.batch_norm_op, orm.Slice(0, 5))
    batch_norm_op_group4 = orm.OpGroup(
        batch_norm_op_slice_0_5,
        omit_source_op_slices=[batch_norm_op_slice_0_5])
    batch_norm_op_slice_5_9 = orm.OpSlice(self.batch_norm_op, orm.Slice(5, 4))
    batch_norm_op_group5 = orm.OpGroup(
        batch_norm_op_slice_5_9,
        omit_source_op_slices=[batch_norm_op_slice_5_9])
    batch_norm_op_slice_9_11 = orm.OpSlice(self.batch_norm_op, orm.Slice(9, 2))
    batch_norm_op_group6 = orm.OpGroup(
        batch_norm_op_slice_9_11,
        omit_source_op_slices=[batch_norm_op_slice_9_11])
    batch_norm_op_slice_11_13 = orm.OpSlice(
        self.batch_norm_op, orm.Slice(11, 2))
    batch_norm_op_group7 = orm.OpGroup(
        batch_norm_op_slice_11_13,
        omit_source_op_slices=[batch_norm_op_slice_11_13])
    batch_norm_op_slice_13_18 = orm.OpSlice(
        self.batch_norm_op, orm.Slice(11, 5))
    batch_norm_op_group8 = orm.OpGroup(
        batch_norm_op_slice_13_18,
        omit_source_op_slices=[batch_norm_op_slice_13_18])

    # Map ops to slices.  Batch norm op is composed of multiple slices.
    self.op_slice_dict = {
        self.relu1_op: [self.relu1_op_slice],
        self.relu2_op: [self.relu2_op_slice],
        self.relu3_op: [self.relu3_op_slice],
        self.concat_op: [self.concat_op_slice],
        self.batch_norm_op: [batch_norm_op_slice_0_9, batch_norm_op_slice_9_13,
                             batch_norm_op_slice_13_18],
    }

    # Map each slice to a group.
    self.op_group_dict = {
        self.relu1_op_slice: self.relu1_op_group,
        self.relu2_op_slice: self.relu2_op_group,
        self.relu3_op_slice: self.relu3_op_group,
        batch_norm_op_slice_0_9: batch_norm_op_group1,
        batch_norm_op_slice_9_13: batch_norm_op_group2,
        batch_norm_op_slice_13_18: batch_norm_op_group3,
        batch_norm_op_slice_0_5: batch_norm_op_group4,
        batch_norm_op_slice_5_9: batch_norm_op_group5,
        batch_norm_op_slice_9_11: batch_norm_op_group6,
        batch_norm_op_slice_11_13: batch_norm_op_group7,
        batch_norm_op_slice_13_18: batch_norm_op_group8,
    }

    # Update op_slice_dict when an op is sliced.
    def slice_op(op, _):
      if op == self.batch_norm_op:
        self.op_slice_dict[self.batch_norm_op] = [
            batch_norm_op_slice_0_5,
            batch_norm_op_slice_5_9,
            batch_norm_op_slice_9_11,
            batch_norm_op_slice_11_13,
            batch_norm_op_slice_13_18]
      if op == self.concat_op:
        self.op_slice_dict[self.concat_op] = [
            concat_op_slice_0_5,
            concat_op_slice_5_9,
            concat_op_slice_9_11,
            concat_op_slice_11_13,
            concat_op_slice_13_18]
      if op == self.relu2_op:
        self.op_slice_dict[self.relu2_op] = [
            relu2_op_slice_0_4,
            relu2_op_slice_4_6]
      if op == self.relu3_op:
        self.op_slice_dict[self.relu3_op] = [
            relu3_op_slice_0_2,
            relu3_op_slice_2_7]

    self.mock_op_reg_manager.slice_op.side_effect = slice_op

    # Call handler to assign grouping.
    handler = concat_op_handler.ConcatOpHandler()
    handler.assign_grouping(self.concat_op, self.mock_op_reg_manager)

    # Verify manager looks up OpSlice for ops of interest.
    self.mock_op_reg_manager.get_op_slices.assert_has_calls(
        # Checking for ops to process.
        [mock.call(self.relu1_op),
         mock.call(self.relu2_op),
         mock.call(self.relu3_op),
         mock.call(self.batch_norm_op),
         # Initial slice data.
         mock.call(self.concat_op),
         mock.call(self.relu1_op),
         mock.call(self.relu2_op),
         mock.call(self.relu3_op),
         mock.call(self.batch_norm_op),
         # Reslicing.
         mock.call(self.relu1_op),
         mock.call(self.relu2_op),
         mock.call(self.relu3_op),
         mock.call(self.batch_norm_op),
         mock.call(self.concat_op),
         # Refreshing slice data.
         mock.call(self.relu1_op),
         mock.call(self.relu2_op),
         mock.call(self.relu3_op),
         mock.call(self.batch_norm_op),
         # Group concat op.
         mock.call(self.concat_op)])

    # Verify manager slices ops that do not have aligned OpSlice sizes.
    self.mock_op_reg_manager.slice_op.assert_has_calls(
        [mock.call(self.relu2_op, [4, 2]),
         mock.call(self.relu3_op, [2, 5]),
         mock.call(self.batch_norm_op, [5, 4, 2, 2, 5]),
         mock.call(self.concat_op, [5, 4, 2, 2, 5])])

    # Verify manager groups the new slices.
    self.mock_op_reg_manager.group_op_slices.assert_has_calls(
        [mock.call([concat_op_slice_0_5, self.relu1_op_slice,
                    batch_norm_op_slice_0_5]),
         mock.call([concat_op_slice_5_9, relu2_op_slice_0_4,
                    batch_norm_op_slice_5_9]),
         mock.call([concat_op_slice_9_11, relu2_op_slice_4_6,
                    batch_norm_op_slice_9_11]),
         mock.call([concat_op_slice_11_13, relu3_op_slice_0_2,
                    batch_norm_op_slice_11_13]),
         mock.call([concat_op_slice_13_18, relu3_op_slice_2_7,
                    batch_norm_op_slice_13_18])])

  def testAssignGrouping_AllNeighborsGrouped_InputSlicesNotAligned(self):
    # In this test, the op c2 has size 6 but is split into 2 slices of size 3.
    # The concat op (and its output, the batch norm) both have size 18.  This
    # test verifies that the concat and batch norm are sliced according to the
    # sizes of c1, c2, and c3, and takes into account that c2 is also composed
    # of multiple slices.
    concat_op_slice_0_5 = orm.OpSlice(self.concat_op, orm.Slice(0, 5))
    concat_op_slice_5_8 = orm.OpSlice(self.concat_op, orm.Slice(5, 3))
    concat_op_slice_8_11 = orm.OpSlice(self.concat_op, orm.Slice(8, 3))
    concat_op_slice_11_18 = orm.OpSlice(self.concat_op, orm.Slice(11, 7))

    relu2_op_slice_0_3 = orm.OpSlice(self.relu2_op, orm.Slice(0, 3))
    relu2_op_slice_3_6 = orm.OpSlice(self.relu2_op, orm.Slice(3, 3))
    relu2_op_group1 = orm.OpGroup(
        relu2_op_slice_0_3, omit_source_op_slices=[relu2_op_slice_0_3])
    relu2_op_group2 = orm.OpGroup(
        relu2_op_slice_3_6, omit_source_op_slices=[relu2_op_slice_3_6])

    batch_norm_op_slice = orm.OpSlice(self.batch_norm_op, orm.Slice(0, 18))
    batch_norm_op_group = orm.OpGroup(
        batch_norm_op_slice, omit_source_op_slices=[batch_norm_op_slice])
    batch_norm_op_slice_0_5 = orm.OpSlice(self.batch_norm_op, orm.Slice(0, 5))
    batch_norm_op_group1 = orm.OpGroup(
        batch_norm_op_slice_0_5,
        omit_source_op_slices=[batch_norm_op_slice_0_5])
    batch_norm_op_slice_5_8 = orm.OpSlice(self.batch_norm_op, orm.Slice(5, 3))
    batch_norm_op_group2 = orm.OpGroup(
        batch_norm_op_slice_5_8,
        omit_source_op_slices=[batch_norm_op_slice_5_8])
    batch_norm_op_slice_8_11 = orm.OpSlice(self.batch_norm_op, orm.Slice(8, 3))
    batch_norm_op_group3 = orm.OpGroup(
        batch_norm_op_slice_8_11,
        omit_source_op_slices=[batch_norm_op_slice_8_11])
    batch_norm_op_slice_11_18 = orm.OpSlice(
        self.batch_norm_op, orm.Slice(11, 7))
    batch_norm_op_group4 = orm.OpGroup(
        batch_norm_op_slice_11_18,
        omit_source_op_slices=[batch_norm_op_slice_11_18])

    # Map ops to slices.  The op c2 is composed of multiple slices.
    self.op_slice_dict = {
        self.relu1_op: [self.relu1_op_slice],
        self.relu2_op: [relu2_op_slice_0_3, relu2_op_slice_3_6],
        self.relu3_op: [self.relu3_op_slice],
        self.concat_op: [self.concat_op_slice],
        self.batch_norm_op: [batch_norm_op_slice],
    }

    # Map each slice to a group.
    self.op_group_dict = {
        self.relu1_op_slice: self.relu1_op_group,
        relu2_op_slice_0_3: relu2_op_group1,
        relu2_op_slice_3_6: relu2_op_group2,
        self.relu3_op_slice: self.relu3_op_group,
        batch_norm_op_slice: batch_norm_op_group,
        batch_norm_op_slice_0_5: batch_norm_op_group1,
        batch_norm_op_slice_5_8: batch_norm_op_group2,
        batch_norm_op_slice_8_11: batch_norm_op_group3,
        batch_norm_op_slice_11_18: batch_norm_op_group4,
    }

    # Update op_slice_dict when an op is sliced.
    def slice_op(op, _):
      if op == self.batch_norm_op:
        self.op_slice_dict[self.batch_norm_op] = [
            batch_norm_op_slice_0_5,
            batch_norm_op_slice_5_8,
            batch_norm_op_slice_8_11,
            batch_norm_op_slice_11_18]
      if op == self.concat_op:
        self.op_slice_dict[self.concat_op] = [
            concat_op_slice_0_5,
            concat_op_slice_5_8,
            concat_op_slice_8_11,
            concat_op_slice_11_18]

    self.mock_op_reg_manager.slice_op.side_effect = slice_op

    # Call handler to assign grouping.
    handler = concat_op_handler.ConcatOpHandler()
    handler.assign_grouping(self.concat_op, self.mock_op_reg_manager)

    # Verify manager looks up OpSlice for ops of interest.
    self.mock_op_reg_manager.get_op_slices.assert_has_calls(
        # Checking for ops to process.
        [mock.call(self.relu1_op),
         mock.call(self.relu2_op),
         mock.call(self.relu3_op),
         mock.call(self.batch_norm_op),
         # Initial slice data.
         mock.call(self.concat_op),
         mock.call(self.relu1_op),
         mock.call(self.relu2_op),
         mock.call(self.relu3_op),
         mock.call(self.batch_norm_op),
         # Reslicing.
         mock.call(self.relu1_op),
         mock.call(self.relu2_op),
         mock.call(self.relu3_op),
         mock.call(self.batch_norm_op),
         mock.call(self.concat_op),
         # Refreshing slice data.
         mock.call(self.relu1_op),
         mock.call(self.relu2_op),
         mock.call(self.relu3_op),
         mock.call(self.batch_norm_op),
         # Group concat op.
         mock.call(self.concat_op)])

    # Verify manager slices ops that do not have aligned OpSlice sizes.
    self.mock_op_reg_manager.slice_op.assert_has_calls(
        [mock.call(self.batch_norm_op, [5, 3, 3, 7]),
         mock.call(self.concat_op, [5, 3, 3, 7])])

    # Verify manager groups the new slices.
    self.mock_op_reg_manager.group_op_slices.assert_has_calls(
        [mock.call([concat_op_slice_0_5, self.relu1_op_slice,
                    batch_norm_op_slice_0_5]),
         mock.call([concat_op_slice_5_8, relu2_op_slice_0_3,
                    batch_norm_op_slice_5_8]),
         mock.call([concat_op_slice_8_11, relu2_op_slice_3_6,
                    batch_norm_op_slice_8_11]),
         mock.call([concat_op_slice_11_18, self.relu3_op_slice,
                    batch_norm_op_slice_11_18])])

  def testAssignGrouping_InputsGrouped(self):
    # In this test, only the input ops are grouped.  The concat and batch norm
    # ops will be sliced according to the input sizes.

    # Map ops to slices.
    self.op_slice_dict = {
        self.relu1_op: [self.relu1_op_slice],
        self.relu2_op: [self.relu2_op_slice],
        self.relu3_op: [self.relu3_op_slice],
        self.concat_op: [self.concat_op_slice],
        self.batch_norm_op: [self.batch_norm_op_slice],
    }

    # Map each slice to a group.  Batch norm (output) is not grouped.
    self.op_group_dict = {
        self.relu1_op_slice: self.relu1_op_group,
        self.relu2_op_slice: self.relu2_op_group,
        self.relu3_op_slice: self.relu3_op_group,
        self.concat_op_slice_0_5: self.concat_op_group1,
        self.concat_op_slice_5_11: self.concat_op_group2,
        self.concat_op_slice_11_18: self.concat_op_group3,
    }

    # Call handler to assign grouping.
    handler = concat_op_handler.ConcatOpHandler()
    handler.assign_grouping(self.concat_op, self.mock_op_reg_manager)

    # Verify manager looks up OpSlice for ops of interest.
    self.mock_op_reg_manager.get_op_slices.assert_has_calls(
        # Checking for ops to process.
        [mock.call(self.relu1_op),
         mock.call(self.relu2_op),
         mock.call(self.relu3_op),
         mock.call(self.batch_norm_op),
         # Initial slice data.
         mock.call(self.concat_op),
         mock.call(self.relu1_op),
         mock.call(self.relu2_op),
         mock.call(self.relu3_op),
         mock.call(self.batch_norm_op),
         # Reslicing.
         mock.call(self.relu1_op),
         mock.call(self.relu2_op),
         mock.call(self.relu3_op),
         mock.call(self.batch_norm_op),
         mock.call(self.concat_op),
         # Refreshing slice data.
         mock.call(self.relu1_op),
         mock.call(self.relu2_op),
         mock.call(self.relu3_op),
         mock.call(self.batch_norm_op),
         # Group concat op.
         mock.call(self.concat_op)])

    # Verify manager slices ops that do not have aligned OpSlice sizes.
    self.mock_op_reg_manager.slice_op.assert_has_calls(
        [mock.call(self.batch_norm_op, [5, 6, 7]),
         mock.call(self.concat_op, [5, 6, 7])])

    # Verify manager groups the new slices.
    self.mock_op_reg_manager.group_op_slices.assert_has_calls(
        [mock.call([self.concat_op_slice_0_5, self.relu1_op_slice]),
         mock.call([self.concat_op_slice_5_11, self.relu2_op_slice]),
         mock.call([self.concat_op_slice_11_18, self.relu3_op_slice])])

    # Verify manager adds ops to processing queue.
    self.mock_op_reg_manager.process_ops.assert_called_once_with(
        [self.batch_norm_op])

  def testAssignGrouping_OutputsGrouped(self):
    # In this test, only the output ops are grouped.  The concat and batch norm
    # ops will be sliced according to the input sizes.

    # Map ops to slices.
    self.op_slice_dict = {
        self.relu1_op: [self.relu1_op_slice],
        self.relu2_op: [self.relu2_op_slice],
        self.relu3_op: [self.relu3_op_slice],
        self.concat_op: [self.concat_op_slice],
        self.batch_norm_op: [self.batch_norm_op_slice],
    }

    # Map each slice to a group.  Input ops (ReLU) are not grouped.
    self.op_group_dict = {
        self.concat_op_slice_0_5: self.concat_op_group1,
        self.concat_op_slice_5_11: self.concat_op_group2,
        self.concat_op_slice_11_18: self.concat_op_group3,
        self.batch_norm_op_slice: self.batch_norm_op_group,
        self.batch_norm_op_slice_0_5: self.batch_norm_op_group1,
        self.batch_norm_op_slice_5_11: self.batch_norm_op_group2,
        self.batch_norm_op_slice_11_18: self.batch_norm_op_group3,
    }

    # Call handler to assign grouping.
    handler = concat_op_handler.ConcatOpHandler()
    handler.assign_grouping(self.concat_op, self.mock_op_reg_manager)

    # Verify manager looks up OpSlice for ops of interest.
    self.mock_op_reg_manager.get_op_slices.assert_has_calls(
        # Checking for ops to process.
        [mock.call(self.relu1_op),
         mock.call(self.relu2_op),
         mock.call(self.relu3_op),
         mock.call(self.batch_norm_op),
         # Initial slice data.
         mock.call(self.concat_op),
         mock.call(self.relu1_op),
         mock.call(self.relu2_op),
         mock.call(self.relu3_op),
         mock.call(self.batch_norm_op),
         # Reslicing.
         mock.call(self.relu1_op),
         mock.call(self.relu2_op),
         mock.call(self.relu3_op),
         mock.call(self.batch_norm_op),
         mock.call(self.concat_op),
         # Refreshing slice data.
         mock.call(self.relu1_op),
         mock.call(self.relu2_op),
         mock.call(self.relu3_op),
         mock.call(self.batch_norm_op)])

    # Verify manager slices ops that do not have aligned OpSlice sizes.
    self.mock_op_reg_manager.slice_op.assert_has_calls(
        [mock.call(self.batch_norm_op, [5, 6, 7]),
         mock.call(self.concat_op, [5, 6, 7])])

    # Verify manager does not group ops.
    self.mock_op_reg_manager.group_op_slices.assert_not_called()

    # Verify manager adds ops to processing queue.
    self.mock_op_reg_manager.process_ops.assert_called_once_with(
        [self.relu1_op, self.relu2_op, self.relu3_op])
    self.mock_op_reg_manager.process_ops_last.assert_called_once_with(
        [self.concat_op])

  def testAssignGrouping_NoNeighborGroups(self):
    # In this test, both the inputs and outputs are missing groups.  The concat
    # and batch norm are sliced, but grouping does not happen until the inputs
    # and outputs are grouped.

    # Map ops to slices.
    self.op_slice_dict = {
        self.relu1_op: [self.relu1_op_slice],
        self.relu2_op: [self.relu2_op_slice],
        self.relu3_op: [self.relu3_op_slice],
        self.concat_op: [self.concat_op_slice],
        self.batch_norm_op: [self.batch_norm_op_slice],
    }

    # No neighbor slices are grouped.
    self.op_group_dict = {
        self.concat_op_slice_0_5: self.concat_op_group1,
        self.concat_op_slice_5_11: self.concat_op_group2,
        self.concat_op_slice_11_18: self.concat_op_group3,
    }

    # Call handler to assign grouping.
    handler = concat_op_handler.ConcatOpHandler()
    handler.assign_grouping(self.concat_op, self.mock_op_reg_manager)

    # Verify manager looks up OpSlice for ops of interest.
    self.mock_op_reg_manager.get_op_slices.assert_has_calls(
        # Checking for ops to process.
        [mock.call(self.relu1_op),
         mock.call(self.relu2_op),
         mock.call(self.relu3_op),
         mock.call(self.batch_norm_op),
         # Initial slice data.
         mock.call(self.concat_op),
         mock.call(self.relu1_op),
         mock.call(self.relu2_op),
         mock.call(self.relu3_op),
         mock.call(self.batch_norm_op),
         # Reslicing.
         mock.call(self.relu1_op),
         mock.call(self.relu2_op),
         mock.call(self.relu3_op),
         mock.call(self.batch_norm_op),
         mock.call(self.concat_op),
         # Refreshing slice data.
         mock.call(self.relu1_op),
         mock.call(self.relu2_op),
         mock.call(self.relu3_op),
         mock.call(self.batch_norm_op)])

    # Verify manager slices ops that do not have aligned OpSlice sizes.
    self.mock_op_reg_manager.slice_op.assert_has_calls(
        [mock.call(self.batch_norm_op, [5, 6, 7]),
         mock.call(self.concat_op, [5, 6, 7])])

    # Verify manager doesn't group anything.
    self.mock_op_reg_manager.group_op_slices.assert_not_called()

    # Verify manager adds ops to processing queue.
    self.mock_op_reg_manager.process_ops.assert_called_once_with(
        [self.batch_norm_op, self.relu1_op, self.relu2_op, self.relu3_op])
    self.mock_op_reg_manager.process_ops_last.assert_called_once_with(
        [self.concat_op])

  def testGetInputOutputOpSlices(self):
    # Map ops to slices.
    self.op_slice_dict = {
        self.relu1_op: [self.relu1_op_slice],
        self.relu2_op: [self.relu2_op_slice],
        self.relu3_op: [self.relu3_op_slice],
        self.concat_op: [self.concat_op_slice],
        self.batch_norm_op: [self.batch_norm_op_slice],
    }

    input_ops = [self.relu1_op, self.relu2_op, self.relu3_op, self.axis_op]
    output_ops = [self.batch_norm_op]

    expected_input_op_slices = [
        [self.relu1_op_slice, self.relu2_op_slice, self.relu3_op_slice]]
    expected_output_op_slices = [
        [self.batch_norm_op_slice]]

    # Instantiate handler.
    handler = concat_op_handler.ConcatOpHandler()

    self.assertEqual(
        (expected_input_op_slices, expected_output_op_slices),
        handler._get_input_output_op_slices(input_ops, output_ops,
                                            self.mock_op_reg_manager))


class GroupingConcatOpHandlerTest(tf.test.TestCase):

  def _get_scope(self):
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

    # This tests 3 Conv2D ops being concatenated.
    inputs = tf.zeros([2, 4, 4, 3])
    with tf.contrib.framework.arg_scope(self._get_scope()):
      c1 = layers.conv2d(inputs, num_outputs=6, kernel_size=3, scope='conv1')
      c2 = layers.conv2d(inputs, num_outputs=6, kernel_size=3, scope='conv2')
      c3 = layers.conv2d(inputs, num_outputs=6, kernel_size=3, scope='conv3')
      net = tf.concat([c1, c2, c3], axis=2)
      layers.batch_norm(net)

    g = tf.get_default_graph()

    # Declare OpSlice and OpGroup for ops of interest.
    self.concat_op = g.get_operation_by_name('concat')
    self.concat_op_slice = orm.OpSlice(self.concat_op, orm.Slice(0, 6))
    self.concat_op_group = orm.OpGroup(
        self.concat_op_slice,
        omit_source_op_slices=[self.concat_op_slice])

    self.relu1_op = g.get_operation_by_name('conv1/Relu')
    self.relu1_op_slice = orm.OpSlice(self.relu1_op, orm.Slice(0, 6))
    self.relu1_op_group = orm.OpGroup(
        self.relu1_op_slice, omit_source_op_slices=[self.relu1_op_slice])

    self.relu2_op = g.get_operation_by_name('conv2/Relu')
    self.relu2_op_slice = orm.OpSlice(self.relu2_op, orm.Slice(0, 6))
    self.relu2_op_group = orm.OpGroup(
        self.relu2_op_slice, omit_source_op_slices=[self.relu2_op_slice])

    self.relu3_op = g.get_operation_by_name('conv3/Relu')
    self.relu3_op_slice = orm.OpSlice(self.relu3_op, orm.Slice(0, 6))
    self.relu3_op_group = orm.OpGroup(
        self.relu3_op_slice, omit_source_op_slices=[self.relu3_op_slice])

    self.batch_norm_op = g.get_operation_by_name('BatchNorm/FusedBatchNormV3')
    self.batch_norm_op_slice = orm.OpSlice(self.batch_norm_op, orm.Slice(0, 6))
    self.batch_norm_op_group = orm.OpGroup(
        self.batch_norm_op_slice,
        omit_source_op_slices=[self.batch_norm_op_slice])

    self.concat_group = orm.OpGroup(
        op_slice=None,
        op_groups=[
            self.batch_norm_op_group, self.concat_op_group, self.relu1_op_group,
            self.relu2_op_group, self.relu3_op_group
        ])

    # Create mock OpRegularizerManager with custom mapping of OpSlice and
    # OpGroup.
    self.mock_op_reg_manager = mock.create_autospec(orm.OpRegularizerManager)

    def get_op_slices(op):
      return self.op_slice_dict.get(op, [])

    def get_op_group(op_slice):
      return self.op_group_dict.get(op_slice)

    self.mock_op_reg_manager.get_op_slices.side_effect = get_op_slices
    self.mock_op_reg_manager.get_op_group.side_effect = get_op_group
    self.mock_op_reg_manager.is_source_op.return_value = False
    self.mock_op_reg_manager.is_passthrough.return_value = True
    self.mock_op_reg_manager.ops = [
        self.concat_op, self.relu1_op, self.relu2_op, self.relu3_op,
        self.batch_norm_op]

  def test_AssignGroupingOfGroupingConcatNoSlicing(self):
    # In this test, the output op (batch norm) has size 6 and is not sliced.
    # and that input Conv2Ds are all of size 6, and are grouped.

    # Map ops to slices.  Batch norm op is composed of multiple slices.
    self.op_slice_dict = {
        self.relu1_op: [self.relu1_op_slice],
        self.relu2_op: [self.relu2_op_slice],
        self.relu3_op: [self.relu3_op_slice],
        self.concat_op: [self.concat_op_slice],
        self.batch_norm_op: [self.batch_norm_op_slice],
    }

    # Map each slice to a group.
    self.op_group_dict = {
        self.relu1_op_slice: self.relu1_op_group,
        self.relu2_op_slice: self.relu2_op_group,
        self.relu3_op_slice: self.relu3_op_group,
        self.batch_norm_op_slice: self.batch_norm_op_group
    }

    # Call handler to assign grouping.
    handler = concat_op_handler.ConcatOpHandler()
    handler.assign_grouping(self.concat_op, self.mock_op_reg_manager)

    # Verify manager looks up OpSlice for ops of interest.
    self.mock_op_reg_manager.get_op_slices.assert_has_calls(
        # Checking for ops to process.
        [mock.call(self.relu1_op),
         mock.call(self.relu2_op),
         mock.call(self.relu3_op),
         mock.call(self.batch_norm_op),
         # Initial slice data.
         mock.call(self.concat_op),
         mock.call(self.relu1_op),
         mock.call(self.relu2_op),
         mock.call(self.relu3_op),
         mock.call(self.batch_norm_op),
         # Reslicing.
         mock.call(self.relu1_op),
         mock.call(self.relu2_op),
         mock.call(self.relu3_op),
         mock.call(self.concat_op),
         mock.call(self.batch_norm_op),
         # Refreshing slice data.
         mock.call(self.relu1_op),
         mock.call(self.relu2_op),
         mock.call(self.relu3_op),
         mock.call(self.batch_norm_op),
         # Group concat op.
         mock.call(self.concat_op)])

    # Verify manager does not slices the concat op.
    self.mock_op_reg_manager.slice_op.assert_not_called()

    # Verify manager groups the new slices.
    self.mock_op_reg_manager.group_op_slices.assert_called_once_with([
        self.concat_op_slice, self.relu1_op_slice, self.relu2_op_slice,
        self.relu3_op_slice, self.batch_norm_op_slice
    ])

  def testGetConcatOpAxis(self):
    x = tf.zeros([7, 12, 12, 3])
    self.assertEqual(
        concat_op_handler._get_concat_op_axis(tf.concat([x, x], 3).op), 3)
    self.assertEqual(
        concat_op_handler._get_concat_op_axis(tf.concat([x, x, x], 1).op), 1)
    self.assertEqual(
        concat_op_handler._get_concat_op_axis(tf.concat([x, x, x], 2).op), 2)


if __name__ == '__main__':
  tf.test.main()
