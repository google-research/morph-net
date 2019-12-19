"""Tests for depth_to_space_op_handler."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import mock

from morph_net.framework import depth_to_space_op_handler
from morph_net.framework import op_regularizer_manager as orm
import tensorflow.compat.v1 as tf


class DepthToSpaceOpHandlerTest(tf.test.TestCase):

  def setUp(self):
    super(DepthToSpaceOpHandlerTest, self).setUp()
    # Test a Identity -> DepthToSpace -> Identity chain of ops.
    inputs = tf.zeros([2, 4, 4, 4])
    id1 = tf.identity(inputs)
    dts = tf.depth_to_space(id1, 2)
    tf.identity(dts)

    g = tf.get_default_graph()

    # Declare OpSlice and OpGroup for ops of interest.
    self.id1_op = g.get_operation_by_name('Identity')
    self.id1_op_slice = orm.OpSlice(self.id1_op, orm.Slice(0, 4))
    self.id1_op_group = orm.OpGroup(self.id1_op_slice,
                                    omit_source_op_slices=[self.id1_op_slice])
    self.id1_op_slice0 = orm.OpSlice(self.id1_op, orm.Slice(0, 1))
    self.id1_op_slice1 = orm.OpSlice(self.id1_op, orm.Slice(1, 1))
    self.id1_op_slice2 = orm.OpSlice(self.id1_op, orm.Slice(2, 1))
    self.id1_op_slice3 = orm.OpSlice(self.id1_op, orm.Slice(3, 1))

    self.dts_op = g.get_operation_by_name('DepthToSpace')
    self.dts_op_slice = orm.OpSlice(self.dts_op, orm.Slice(0, 1))
    self.dts_op_group = orm.OpGroup(self.dts_op_slice,
                                    omit_source_op_slices=[self.dts_op_slice])

    self.id2_op = g.get_operation_by_name('Identity_1')
    self.id2_op_slice = orm.OpSlice(self.id2_op, orm.Slice(0, 1))
    self.id2_op_group = orm.OpGroup(self.id2_op_slice,
                                    omit_source_op_slices=[self.id2_op_slice])

    # Create mock OpRegularizerManager with custom mapping of OpSlice and
    # OpGroup.
    self.mock_op_reg_manager = mock.create_autospec(orm.OpRegularizerManager)

    self.op_slice_dict = {
        self.id1_op: [self.id1_op_slice],
        self.dts_op: [self.dts_op_slice],
        self.id2_op: [self.id2_op_slice],
    }
    def get_op_slices(op):
      return self.op_slice_dict.get(op)

    def get_op_group(op_slice):
      return self.op_group_dict.get(op_slice)

    self.mock_op_reg_manager.get_op_slices.side_effect = get_op_slices
    self.mock_op_reg_manager.get_op_group.side_effect = get_op_group
    self.mock_op_reg_manager.is_source_op.return_value = False
    self.mock_op_reg_manager.ops = [self.id1_op, self.dts_op, self.id2_op]

  def test_assign_grouping_no_neighbor_groups(self):
    # No ops have groups.
    self.op_group_dict = {}

    # Call handler to assign grouping.
    handler = depth_to_space_op_handler.DepthToSpaceOpHandler()
    handler.assign_grouping(self.dts_op, self.mock_op_reg_manager)

    # Verify manager looks up OpSlice for ops of interest.
    self.mock_op_reg_manager.get_op_slices.assert_has_calls(
        [mock.call(self.id1_op),
         mock.call(self.id2_op)])

    # Verify manager does not group.
    self.mock_op_reg_manager.group_op_slices.assert_not_called()

    # Verify manager processes grouping for identity ops.
    self.mock_op_reg_manager.process_ops.assert_called_once_with(
        [self.id1_op])

  def test_assign_grouping_all_inputs_grouped(self):
    # Map ops to slices.
    self.op_slice_dict[self.id1_op] = [
        self.id1_op_slice0,
        self.id1_op_slice1,
        self.id1_op_slice2,
        self.id1_op_slice3]

    # All inputs have groups.
    self.op_group_dict = {
        self.id1_op_slice0: self.id1_op_group,
        self.id1_op_slice1: self.id1_op_group,
        self.id1_op_slice2: self.id1_op_group,
        self.id1_op_slice3: self.id1_op_group,
    }

    # Call handler to assign grouping.
    handler = depth_to_space_op_handler.DepthToSpaceOpHandler()
    handler.assign_grouping(self.dts_op, self.mock_op_reg_manager)

    # Verify manager looks up OpSlice for ops of interest.
    self.mock_op_reg_manager.get_op_slices.assert_has_calls(
        # Checking for ops to process.
        [mock.call(self.id1_op),
         mock.call(self.id2_op),
         # Reslicing.
         mock.call(self.id1_op),
         mock.call(self.dts_op),
         mock.call(self.id2_op),
         # Refreshing slice data.
         mock.call(self.dts_op),
         mock.call(self.id1_op)])

    # Verify manager groups DepthToSpace channel with individual input channels.
    self.mock_op_reg_manager.group_op_slices.assert_called_once_with(
        [self.id1_op_slice0, self.id1_op_slice1, self.id1_op_slice2,
         self.id1_op_slice3, self.dts_op_slice])

    # Verify manager processes grouping for identity ops.
    self.mock_op_reg_manager.process_ops.assert_called_once_with([self.id2_op])

  def test_assign_grouping_all_outputs_grouped(self):
    # All outputs have groups.
    self.op_group_dict = {
        self.id2_op_slice: self.id2_op_group,
    }

    # Call handler to assign grouping.
    handler = depth_to_space_op_handler.DepthToSpaceOpHandler()
    handler.assign_grouping(self.dts_op, self.mock_op_reg_manager)

    # Verify manager looks up OpSlice for ops of interest.
    self.mock_op_reg_manager.get_op_slices.assert_has_calls(
        # Checking for ops to process.
        [mock.call(self.id1_op),
         mock.call(self.id2_op)])

    # Verify manager does not group.
    self.mock_op_reg_manager.group_op_slices.assert_not_called()

    # Verify manager processes grouping for identity ops.
    self.mock_op_reg_manager.process_ops.assert_called_once_with(
        [self.id1_op])

  def test_assign_grouping_all_neighbors_grouped(self):
    # Map ops to slices.
    self.op_slice_dict[self.id1_op] = [
        self.id1_op_slice0,
        self.id1_op_slice1,
        self.id1_op_slice2,
        self.id1_op_slice3]

    # All neighbors have groups.
    self.op_group_dict = {
        self.id1_op_slice0: self.id1_op_group,
        self.id1_op_slice1: self.id1_op_group,
        self.id1_op_slice2: self.id1_op_group,
        self.id1_op_slice3: self.id1_op_group,
        self.id2_op_slice: self.id2_op_group,
    }

    # Call handler to assign grouping.
    handler = depth_to_space_op_handler.DepthToSpaceOpHandler()
    handler.assign_grouping(self.dts_op, self.mock_op_reg_manager)

    # Verify manager looks up OpSlice for ops of interest.
    self.mock_op_reg_manager.get_op_slices.assert_has_calls(
        # Checking for ops to process.
        [mock.call(self.id1_op),
         mock.call(self.id2_op),
         # Reslicing.
         mock.call(self.id1_op),
         mock.call(self.dts_op),
         mock.call(self.id2_op),
         # Refreshing slice data.
         mock.call(self.dts_op),
         mock.call(self.id1_op)])

    # Verify manager groups DepthToSpace channel with individual input channels.
    self.mock_op_reg_manager.group_op_slices.assert_called_once_with(
        [self.id1_op_slice0, self.id1_op_slice1, self.id1_op_slice2,
         self.id1_op_slice3, self.dts_op_slice])

    # Verify manager processes grouping for identity ops.
    self.mock_op_reg_manager.process_ops.assert_not_called()

  def test_assign_grouping_all_neighbors_grouped_same_group(self):
    # Map ops to slices.
    self.op_slice_dict[self.id1_op] = [
        self.id1_op_slice0,
        self.id1_op_slice1,
        self.id1_op_slice2,
        self.id1_op_slice3]

    # All neighbors have the same group.
    self.op_group_dict = {
        self.id1_op_slice0: self.id1_op_group,
        self.id1_op_slice1: self.id1_op_group,
        self.id1_op_slice2: self.id1_op_group,
        self.id1_op_slice3: self.id1_op_group,
        self.id2_op_slice: self.id1_op_group,
    }

    # Call handler to assign grouping.
    handler = depth_to_space_op_handler.DepthToSpaceOpHandler()
    handler.assign_grouping(self.dts_op, self.mock_op_reg_manager)

    # Verify manager looks up OpSlice for ops of interest.
    self.mock_op_reg_manager.get_op_slices.assert_has_calls(
        # Checking for ops to process.
        [mock.call(self.id1_op),
         mock.call(self.id2_op),
         # Reslicing.
         mock.call(self.id1_op),
         mock.call(self.dts_op),
         mock.call(self.id2_op),
         # Refreshing slice data.
         mock.call(self.dts_op),
         mock.call(self.id1_op)])

    # Verify manager groups DepthToSpace channel with individual input channels.
    self.mock_op_reg_manager.group_op_slices.assert_called_once_with(
        [self.id1_op_slice0, self.id1_op_slice1, self.id1_op_slice2,
         self.id1_op_slice3, self.dts_op_slice])

    # Verify manager processes grouping for identity ops.
    self.mock_op_reg_manager.process_ops.assert_not_called()


if __name__ == '__main__':
  tf.test.main()
