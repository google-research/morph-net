"""Tests for op_handler_util."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import mock
from morph_net.framework import op_handler_util
from morph_net.framework import op_regularizer_manager as orm
import tensorflow as tf

arg_scope = tf.contrib.framework.arg_scope
layers = tf.contrib.layers


class OpHandlerUtilTest(tf.test.TestCase):

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

    # This tests 3 Conv2D ops being concatenated before a batch normalization.
    c2 = layers.conv2d(inputs, num_outputs=5, kernel_size=3, scope='conv2')
    c3 = layers.conv2d(inputs, num_outputs=6, kernel_size=3, scope='conv3')
    c4 = layers.conv2d(inputs, num_outputs=7, kernel_size=3, scope='conv4')
    net = tf.concat([c2, c3, c4], axis=3)
    layers.batch_norm(net)

    g = tf.get_default_graph()

    # Declare OpSlice and OpGroup for ops in the first test network.
    self.batch_norm_op = g.get_operation_by_name(
        'conv1/BatchNorm/FusedBatchNormV3')
    self.batch_norm_op_slice = orm.OpSlice(self.batch_norm_op, None)
    self.batch_norm_op_group = orm.OpGroup(self.batch_norm_op_slice)

    self.conv_op = g.get_operation_by_name('conv1/Conv2D')
    self.conv_op_slice = orm.OpSlice(self.conv_op, None)
    self.conv_op_group = orm.OpGroup(
        self.conv_op_slice, omit_source_op_slices=[self.conv_op_slice])

    self.gamma_op = g.get_operation_by_name('conv1/BatchNorm/gamma/read')
    self.beta_op = g.get_operation_by_name('conv1/BatchNorm/beta/read')
    self.decay_op = g.get_operation_by_name('conv1/BatchNorm/Const')
    self.epsilon_op = g.get_operation_by_name('conv1/BatchNorm/Const_1')
    self.mean_op = g.get_operation_by_name(
        'conv1/BatchNorm/AssignMovingAvg/sub_1')
    self.std_op = g.get_operation_by_name(
        'conv1/BatchNorm/AssignMovingAvg_1/sub_1')

    self.relu_op = g.get_operation_by_name('conv1/Relu')
    self.relu_op_slice = orm.OpSlice(self.relu_op, None)
    self.relu_op_group = orm.OpGroup(
        self.relu_op_slice, omit_source_op_slices=[self.relu_op_slice])

    # Declare OpSlice and OpGroup for ops in the second test network.
    self.relu2_op = g.get_operation_by_name('conv2/Relu')
    self.relu2_op_slice = orm.OpSlice(self.relu2_op, orm.Slice(0, 5))
    self.relu2_op_group = orm.OpGroup(
        self.relu2_op_slice, omit_source_op_slices=[self.relu2_op_slice])

    self.relu3_op = g.get_operation_by_name('conv3/Relu')
    self.relu3_op_slice = orm.OpSlice(self.relu3_op, orm.Slice(0, 6))
    self.relu3_op_group = orm.OpGroup(
        self.relu3_op_slice, omit_source_op_slices=[self.relu3_op_slice])

    self.relu4_op = g.get_operation_by_name('conv4/Relu')
    self.relu4_op_slice = orm.OpSlice(self.relu4_op, orm.Slice(0, 7))
    self.relu4_op_group = orm.OpGroup(
        self.relu4_op_slice, omit_source_op_slices=[self.relu4_op_slice])

    self.unfused_batch_norm_op = g.get_operation_by_name(
        'BatchNorm/FusedBatchNormV3')
    self.unfused_batch_norm_op_slice = orm.OpSlice(
        self.unfused_batch_norm_op, orm.Slice(0, 18))

    self.concat_op = g.get_operation_by_name('concat')
    self.concat_op_slice = orm.OpSlice(self.concat_op, orm.Slice(0, 18))
    self.concat_op_group = orm.OpGroup(
        self.concat_op_slice, omit_source_op_slices=[self.concat_op_slice])

    # Create mock OpRegularizerManager with custom mapping of OpSlice and
    # OpGroup.
    self.mock_op_reg_manager = mock.create_autospec(orm.OpRegularizerManager)

    def get_op_slices(op):
      return self.op_slice_dict.get(op, [])

    def get_op_group(op_slice):
      return self.op_group_dict.get(op_slice)

    def is_passthrough(op):
      return op in self._passthrough_ops

    self.mock_op_reg_manager.get_op_slices.side_effect = get_op_slices
    self.mock_op_reg_manager.get_op_group.side_effect = get_op_group
    self.mock_op_reg_manager.is_passthrough.side_effect = is_passthrough
    self.mock_op_reg_manager.ops = [
        self.batch_norm_op, self.gamma_op, self.beta_op, self.decay_op,
        self.epsilon_op, self.mean_op, self.std_op, self.conv_op, self.relu_op,
        self.relu2_op, self.relu3_op, self.relu4_op, self.unfused_batch_norm_op,
        self.concat_op]

  def testGetInputOps(self):
    # For batch norm, the expected inputs are Conv2D, gamma, and beta.  The
    # decay and epsilon are excluded because they are scalars.
    expected_inputs = [self.conv_op, self.gamma_op, self.beta_op]

    # Check for expected input ops.
    input_ops = op_handler_util.get_input_ops(self.batch_norm_op,
                                              self.mock_op_reg_manager)
    self.assertEqual(expected_inputs, input_ops)
    self.assertNotIn(self.decay_op, input_ops)
    self.assertNotIn(self.epsilon_op, input_ops)

  def testGetOutputOps(self):
    # For batch norm, the expected outputs are mean, std, and ReLU.
    expected_outputs = [self.relu_op, self.mean_op, self.std_op]

    # Check for expected output ops.
    self.assertEqual(
        expected_outputs,
        op_handler_util.get_output_ops(self.batch_norm_op,
                                       self.mock_op_reg_manager))

  def testGetOpsWithoutGroups(self):
    # For a list of ops, verify that ops without groups are returned.
    self.op_slice_dict = {
        self.batch_norm_op: [self.batch_norm_op_slice],
        self.conv_op: [self.conv_op_slice],
        self.gamma_op: [orm.OpSlice(self.gamma_op, None)],
        self.beta_op: [orm.OpSlice(self.beta_op, None)],
        self.decay_op: [orm.OpSlice(self.decay_op, None)],
        self.epsilon_op: [orm.OpSlice(self.epsilon_op, None)],
    }

    # Only batch norm and conv ops have groups.
    self.op_group_dict = {
        self.batch_norm_op_slice: self.batch_norm_op_group,
        self.conv_op_slice: self.conv_op_group
    }

    all_ops = [self.batch_norm_op, self.conv_op, self.gamma_op, self.beta_op,
               self.decay_op, self.epsilon_op]
    # Batch norm and conv ops have groups.  The other ops do not have groups.
    expected_ops = [self.gamma_op, self.beta_op, self.decay_op, self.epsilon_op]
    self.assertEqual(
        expected_ops,
        op_handler_util.get_ops_without_groups(
            all_ops, self.mock_op_reg_manager))

  def testRemoveNonPassthroughOps(self):
    self._passthrough_ops = (self.gamma_op, self.decay_op, self.std_op)

    all_ops = [self.batch_norm_op, self.conv_op, self.gamma_op, self.beta_op,
               self.decay_op, self.epsilon_op, self.mean_op]
    expected_ops = [self.gamma_op, self.decay_op]

    self.assertListEqual(
        expected_ops,
        op_handler_util.remove_non_passthrough_ops(all_ops,
                                                   self.mock_op_reg_manager))

  def testGroupOpWithInputsAndOutputs_SingleSlice(self):
    # For the single slice case, verify that batch norm is grouped with its
    # output (ReLU) and its input (Conv2D).
    aligned_op_slice_sizes = [5]

    self.op_slice_dict = {
        self.batch_norm_op: [self.batch_norm_op_slice],
        self.conv_op: [self.conv_op_slice],
        self.relu_op: [self.relu_op_slice]
    }
    # All ops have groups.
    self.op_group_dict = {
        self.batch_norm_op_slice: self.batch_norm_op_group,
        self.conv_op_slice: self.conv_op_group,
        self.relu_op_slice: self.relu_op_group
    }

    ops_grouped = op_handler_util.group_op_with_inputs_and_outputs(
        self.batch_norm_op, [[self.conv_op_slice]], [[self.relu_op_slice]],
        aligned_op_slice_sizes, self.mock_op_reg_manager)

    # Verify manager looks up op slice for ops of interest.
    self.mock_op_reg_manager.get_op_slices.assert_any_call(self.batch_norm_op)

    # Verify manager groups batch norm with Conv2D and ReLU ops.
    self.assertTrue(ops_grouped)
    self.mock_op_reg_manager.group_op_slices.assert_has_calls(
        [mock.call([self.batch_norm_op_slice, self.relu_op_slice]),
         mock.call([self.batch_norm_op_slice, self.conv_op_slice],
                   omit_source_op_slices=[])])

  def testGroupOpWithInputsAndOutputs_MultipleSlices(self):
    # For the multiple slice case, verify that batch norm slices are grouped
    # with output slices (ReLU) and input slices (Conv2D).
    batch_norm_op_slice_0_2 = orm.OpSlice(
        self.batch_norm_op, orm.OpSlice(0, 2))
    batch_norm_op_slice_2_5 = orm.OpSlice(
        self.batch_norm_op, orm.OpSlice(2, 3))
    batch_norm_op_group1 = orm.OpGroup(
        batch_norm_op_slice_0_2)
    batch_norm_op_group2 = orm.OpGroup(
        batch_norm_op_slice_2_5)

    conv_op_slice_0_2 = orm.OpSlice(
        self.conv_op, orm.OpSlice(0, 2))
    conv_op_slice_2_5 = orm.OpSlice(
        self.conv_op, orm.OpSlice(2, 3))
    conv_op_group1 = orm.OpGroup(
        conv_op_slice_0_2, omit_source_op_slices=[conv_op_slice_0_2])
    conv_op_group2 = orm.OpGroup(
        conv_op_slice_2_5, omit_source_op_slices=[conv_op_slice_2_5])

    relu_op_slice_0_2 = orm.OpSlice(
        self.relu_op, orm.OpSlice(0, 2))
    relu_op_slice_2_5 = orm.OpSlice(
        self.relu_op, orm.OpSlice(2, 3))
    relu_op_group1 = orm.OpGroup(relu_op_slice_0_2)
    relu_op_group2 = orm.OpGroup(relu_op_slice_2_5)

    aligned_op_slice_sizes = [2, 3]

    self.op_slice_dict = {
        self.batch_norm_op: [batch_norm_op_slice_0_2, batch_norm_op_slice_2_5],
        self.conv_op: [conv_op_slice_0_2, conv_op_slice_2_5],
        self.relu_op: [relu_op_slice_0_2, relu_op_slice_2_5],
    }

    # All ops have groups.
    self.op_group_dict = {
        batch_norm_op_slice_0_2: batch_norm_op_group1,
        batch_norm_op_slice_2_5: batch_norm_op_group2,
        conv_op_slice_0_2: conv_op_group1,
        conv_op_slice_2_5: conv_op_group2,
        relu_op_slice_0_2: relu_op_group1,
        relu_op_slice_2_5: relu_op_group2,
    }

    ops_grouped = op_handler_util.group_op_with_inputs_and_outputs(
        self.batch_norm_op, [[conv_op_slice_0_2, conv_op_slice_2_5]],
        [[relu_op_slice_0_2, relu_op_slice_2_5]], aligned_op_slice_sizes,
        self.mock_op_reg_manager)

    # Verify manager looks up op slice for ops of interest.
    self.mock_op_reg_manager.get_op_slices.assert_any_call(self.batch_norm_op)

    # Verify manager groups batch norm with Conv2D and ReLU ops.
    self.assertTrue(ops_grouped)
    self.mock_op_reg_manager.group_op_slices.assert_has_calls(
        [mock.call([batch_norm_op_slice_0_2, relu_op_slice_0_2]),
         mock.call([batch_norm_op_slice_0_2, conv_op_slice_0_2],
                   omit_source_op_slices=[]),
         mock.call([batch_norm_op_slice_2_5, relu_op_slice_2_5]),
         mock.call([batch_norm_op_slice_2_5, conv_op_slice_2_5],
                   omit_source_op_slices=[])])

  def testGetConcatInputOpSlices(self):
    # For concat, the input op slices are the concatenation of op slices of each
    # input op.

    # Map ops to slices.
    self.op_slice_dict = {
        self.relu2_op: [self.relu2_op_slice],
        self.relu3_op: [self.relu3_op_slice],
        self.relu4_op: [self.relu4_op_slice],
    }

    # The concat input is relu2, relu3, and relu4.
    expected_input_op_slices = [
        [self.relu2_op_slice, self.relu3_op_slice, self.relu4_op_slice]]

    input_ops = op_handler_util.get_input_ops(
        self.concat_op, self.mock_op_reg_manager)
    self.assertEqual(
        expected_input_op_slices,
        op_handler_util.get_concat_input_op_slices(
            input_ops, self.mock_op_reg_manager))

  def testGetOpSlices(self):
    # Generic ops are treated as a concatenation of their constituent OpSlice.
    batch_norm_op_slice_0_5 = orm.OpSlice(
        self.unfused_batch_norm_op, orm.Slice(0, 5))
    batch_norm_op_slice_5_11 = orm.OpSlice(
        self.unfused_batch_norm_op, orm.Slice(5, 6))
    batch_norm_op_slice_11_18 = orm.OpSlice(
        self.unfused_batch_norm_op, orm.Slice(11, 7))

    # Map ops to slices.
    self.op_slice_dict = {
        self.unfused_batch_norm_op: [
            batch_norm_op_slice_0_5, batch_norm_op_slice_5_11,
            batch_norm_op_slice_11_18],
    }

    # A nested list composed of a list of OpSlice for each output op.  In this
    # case, there is just one output op (i.e. batch norm).
    expected_output_op_slices = [[
        batch_norm_op_slice_0_5,
        batch_norm_op_slice_5_11,
        batch_norm_op_slice_11_18]]

    output_ops = op_handler_util.get_output_ops(
        self.concat_op, self.mock_op_reg_manager)
    self.assertEqual(
        expected_output_op_slices,
        op_handler_util.get_op_slices(output_ops, self.mock_op_reg_manager))

  def testGetOpSlices_FilterEmptySlices(self):
    # No slices are mapped to ops.
    self.op_slice_dict = {}

    # Verify that empty slices are removed.
    input_ops = op_handler_util.get_input_ops(
        self.batch_norm_op, self.mock_op_reg_manager)
    self.assertListEqual([], op_handler_util.get_op_slices(
        input_ops, self.mock_op_reg_manager))

  def testGetOpSliceSizes(self):
    relu3_op_slice_0_3 = orm.OpSlice(
        self.relu2_op, orm.Slice(0, 3))
    relu3_op_slice_3_6 = orm.OpSlice(
        self.relu2_op, orm.Slice(3, 3))

    batch_norm_op_slice_0_5 = orm.OpSlice(
        self.unfused_batch_norm_op, orm.Slice(0, 5))
    batch_norm_op_slice_5_8 = orm.OpSlice(
        self.unfused_batch_norm_op, orm.Slice(5, 3))
    batch_norm_op_slice_8_11 = orm.OpSlice(
        self.unfused_batch_norm_op, orm.Slice(8, 3))
    batch_norm_op_slice_11_18 = orm.OpSlice(
        self.unfused_batch_norm_op, orm.Slice(11, 7))

    # Map ops to slices.
    self.op_slice_dict = {
        self.relu2_op: [self.relu2_op_slice],
        self.relu3_op: [relu3_op_slice_0_3, relu3_op_slice_3_6],
        self.relu4_op: [self.relu4_op_slice],
        self.unfused_batch_norm_op: [
            batch_norm_op_slice_0_5, batch_norm_op_slice_5_8,
            batch_norm_op_slice_8_11, batch_norm_op_slice_11_18],
    }

    expected_op_slice_sizes = [
        [5],  # c2 has size 5.
        [3, 3],  # c3 has size 6, but in 2 slices of size 3.
        [7],  # c4 has size 7.
        [5, 3, 3, 7]]  # batch norm has size 18, but slice sizes of c1, c2, c3.

    self.assertEqual(
        expected_op_slice_sizes,
        op_handler_util.get_op_slice_sizes([
            [self.relu2_op_slice],
            [relu3_op_slice_0_3, relu3_op_slice_3_6],
            [self.relu4_op_slice],
            [batch_norm_op_slice_0_5, batch_norm_op_slice_5_8,
             batch_norm_op_slice_8_11, batch_norm_op_slice_11_18]]))

  def testGetAlignedOpSliceSizes(self):
    expected_op_slice_sizes = [5, 4, 2, 2, 5]
    self.assertEqual(
        expected_op_slice_sizes,
        op_handler_util.get_aligned_sizes([
            [5, 4, 2, 7],
            [9, 4, 5],
            [18]]))

    expected_op_slice_sizes = [1, 2, 2, 1, 3, 1, 2, 2, 1]
    self.assertEqual(
        expected_op_slice_sizes,
        op_handler_util.get_aligned_sizes([
            [1, 2, 3, 4, 5],
            [5, 4, 3, 2, 1]]))

    expected_op_slice_sizes = [1, 1, 1, 1, 1]
    self.assertEqual(
        expected_op_slice_sizes,
        op_handler_util.get_aligned_sizes([
            [5],
            [1, 1, 1, 1, 1]]))

    expected_op_slice_sizes = [10]
    self.assertEqual(
        expected_op_slice_sizes,
        op_handler_util.get_aligned_sizes([[10]]))

    # Raise exception for empty input.
    with self.assertRaises(ValueError):
      op_handler_util.get_aligned_sizes([])

    # Raise exception if total sizes do not match.
    with self.assertRaises(ValueError):
      op_handler_util.get_aligned_sizes([[1, 2], [4]])

  def testGetNumSlices(self):
    self.assertEqual(
        5, op_handler_util._get_num_slices([[1, 2, 3, 4, 5], [6, 7], [8]]))
    self.assertEqual(
        2, op_handler_util._get_num_slices([[6, 7], [8]]))
    self.assertEqual(
        1, op_handler_util._get_num_slices([[8]]))

  def testResliceConcatOps_Aligned(self):
    # Map ops to slices.
    self.op_slice_dict = {
        self.relu2_op: [self.relu2_op_slice],
        self.relu3_op: [self.relu3_op_slice],
        self.relu4_op: [self.relu4_op_slice],
    }

    op_handler_util.reslice_concat_ops(
        [self.relu2_op, self.relu3_op, self.relu4_op],
        [5, 6, 7], self.mock_op_reg_manager)

    # Verify manager does not slice any ops.
    self.mock_op_reg_manager.slice_op.assert_not_called()

  def testResliceConcatOps_NotAligned(self):
    relu3_op_slice_0_3 = orm.OpSlice(
        self.relu3_op, orm.Slice(0, 3))
    relu3_op_slice_3_6 = orm.OpSlice(
        self.relu3_op, orm.Slice(3, 3))

    # Map ops to slices.  The op c3 is composed of multiple slices.
    self.op_slice_dict = {
        self.relu2_op: [self.relu2_op_slice],
        self.relu3_op: [relu3_op_slice_0_3, relu3_op_slice_3_6],
        self.relu4_op: [self.relu4_op_slice],
    }

    op_handler_util.reslice_concat_ops(
        [self.relu2_op, self.relu3_op, self.relu4_op],
        [5, 4, 2, 2, 5], self.mock_op_reg_manager)

    # Verify manager slices input ops.
    self.mock_op_reg_manager.slice_op.assert_has_calls(
        [mock.call(self.relu3_op, [4, 2]),
         mock.call(self.relu4_op, [2, 5])])

  def testGetTotalSliceSize(self):
    op_slice_sizes = [1, 2, 3, 4, 5, 6, 7, 8, 9]

    self.assertEqual(
        15, op_handler_util.get_total_slice_size(op_slice_sizes, 0, 5))
    self.assertEqual(
        15, op_handler_util.get_total_slice_size(op_slice_sizes, 3, 3))
    self.assertEqual(
        30, op_handler_util.get_total_slice_size(op_slice_sizes, 5, 4))
    self.assertEqual(
        3, op_handler_util.get_total_slice_size(op_slice_sizes, 2, 1))

  def testResliceOps(self):
    # Map ops to slices
    self.op_slice_dict = {
        self.concat_op: [self.concat_op_slice],
        self.unfused_batch_norm_op: [self.unfused_batch_norm_op_slice],
    }

    op_handler_util.reslice_ops(
        [self.concat_op, self.unfused_batch_norm_op],
        [5, 4, 2, 2, 5], self.mock_op_reg_manager)

    # Verify manager slices input ops.
    self.mock_op_reg_manager.slice_op.assert_has_calls(
        [mock.call(self.concat_op, [5, 4, 2, 2, 5]),
         mock.call(self.unfused_batch_norm_op, [5, 4, 2, 2, 5])])

  def testGetSourceOpSlices(self):
    op_slices = [self.batch_norm_op_slice, self.conv_op_slice,
                 self.relu_op_slice]
    self.op_group_dict = {
        self.batch_norm_op_slice: self.batch_norm_op_group,
        self.conv_op_slice: self.conv_op_group,
        self.relu_op_slice: self.relu_op_group,
    }
    expected_source_op_slices = [self.batch_norm_op_slice]

    self.assertEqual(
        expected_source_op_slices,
        op_handler_util._get_source_op_slices(
            op_slices, self.mock_op_reg_manager))

  def testGetInputSourceOpsToOmit_NotSource(self):
    input_op_slices = [
        self.relu2_op_slice, self.relu3_op_slice, self.relu4_op_slice]
    # ReLU3 is a source ops now.
    relu3_op_group = orm.OpGroup(self.relu3_op_slice)
    self.op_group_dict = {
        self.relu2_op_slice: self.relu2_op_group,
        self.relu3_op_slice: relu3_op_group,
        self.relu4_op_slice: self.relu4_op_group,
        self.concat_op_slice: self.concat_op_group,
    }
    expected_ops_to_omit = []

    self.assertEqual(
        expected_ops_to_omit,
        op_handler_util._get_input_source_ops_to_omit(
            input_op_slices, self.concat_op_slice, self.mock_op_reg_manager))

  def testGetInputSourceOpsToOmit_IsSource(self):
    input_op_slices = [
        self.relu2_op_slice, self.relu3_op_slice, self.relu4_op_slice]
    # ReLU3 and concat are source ops now.
    relu3_op_group = orm.OpGroup(self.relu3_op_slice)
    concat_op_group = orm.OpGroup(self.concat_op_slice)
    self.op_group_dict = {
        self.relu2_op_slice: self.relu2_op_group,
        self.relu3_op_slice: relu3_op_group,
        self.relu4_op_slice: self.relu4_op_group,
        self.concat_op_slice: concat_op_group,
    }
    expected_ops_to_omit = [self.relu3_op_slice]

    self.assertEqual(
        expected_ops_to_omit,
        op_handler_util._get_input_source_ops_to_omit(
            input_op_slices, self.concat_op_slice, self.mock_op_reg_manager))

  def testGroupAlignedInputOutputSlices_InputsOutputsGrouped(self):
    self.op_slice_dict = {
        self.batch_norm_op: [self.batch_norm_op_slice],
        self.conv_op: [self.conv_op_slice],
        self.relu_op: [self.relu_op_slice]
    }
    self.op_group_dict = {
        self.batch_norm_op_slice: self.batch_norm_op_group,
        self.conv_op_slice: self.conv_op_group,
        self.relu_op_slice: self.relu_op_group
    }
    input_op_slices = [[self.conv_op_slice]]
    output_op_slices = [[self.relu_op_slice]]
    aligned_op_slice_sizes = [5]

    op_handler_util.group_aligned_input_output_slices(
        self.batch_norm_op, [], [], input_op_slices, output_op_slices,
        aligned_op_slice_sizes, self.mock_op_reg_manager)

    self.mock_op_reg_manager.group_op_slices.assert_has_calls(
        [mock.call([self.batch_norm_op_slice, self.relu_op_slice]),
         mock.call([self.batch_norm_op_slice, self.conv_op_slice],
                   omit_source_op_slices=[])])
    self.mock_op_reg_manager.process_ops.assert_not_called()

  def testGroupAlignedInputOutputSlices_InputsGrouped(self):
    self.op_slice_dict = {
        self.batch_norm_op: [self.batch_norm_op_slice],
        self.conv_op: [self.conv_op_slice],
        self.relu_op: [self.relu_op_slice]
    }
    self.op_group_dict = {
        self.batch_norm_op_slice: self.batch_norm_op_group,
        self.conv_op_slice: self.conv_op_group,
        self.relu_op_slice: self.relu_op_group
    }
    input_op_slices = [[self.conv_op_slice]]
    output_op_slices = [[self.relu_op_slice]]
    aligned_op_slice_sizes = [5]

    op_handler_util.group_aligned_input_output_slices(
        self.batch_norm_op, [], [self.relu_op], input_op_slices,
        output_op_slices, aligned_op_slice_sizes, self.mock_op_reg_manager)

    self.mock_op_reg_manager.group_op_slices.assert_called_once_with(
        [self.batch_norm_op_slice, self.conv_op_slice],
        omit_source_op_slices=[])
    self.mock_op_reg_manager.process_ops.assert_called_once_with([self.relu_op])

  def testGroupAlignedInputOutputSlices_OutputsGrouped(self):
    self.op_slice_dict = {
        self.batch_norm_op: [self.batch_norm_op_slice],
        self.conv_op: [self.conv_op_slice],
        self.relu_op: [self.relu_op_slice]
    }
    self.op_group_dict = {
        self.batch_norm_op_slice: self.batch_norm_op_group,
        self.conv_op_slice: self.conv_op_group,
        self.relu_op_slice: self.relu_op_group
    }
    input_op_slices = [[self.conv_op_slice]]
    output_op_slices = [[self.relu_op_slice]]
    aligned_op_slice_sizes = [5]

    op_handler_util.group_aligned_input_output_slices(
        self.batch_norm_op, [self.conv_op], [], input_op_slices,
        output_op_slices, aligned_op_slice_sizes, self.mock_op_reg_manager)

    self.mock_op_reg_manager.group_op_slices.assert_not_called()
    self.mock_op_reg_manager.process_ops.assert_called_once_with(
        [self.conv_op])
    self.mock_op_reg_manager.process_ops_last.assert_called_once_with(
        [self.batch_norm_op])

  def testGroupAlignedInputOutputSlices_NoGroups(self):
    self.op_slice_dict = {
        self.batch_norm_op: [self.batch_norm_op_slice],
        self.conv_op: [self.conv_op_slice],
        self.relu_op: [self.relu_op_slice]
    }
    self.op_group_dict = {
        self.batch_norm_op_slice: self.batch_norm_op_group,
        self.conv_op_slice: self.conv_op_group,
        self.relu_op_slice: self.relu_op_group
    }
    input_op_slices = [[self.conv_op_slice]]
    output_op_slices = [[self.relu_op_slice]]
    aligned_op_slice_sizes = [5]

    op_handler_util.group_aligned_input_output_slices(
        self.batch_norm_op, [self.conv_op], [self.relu_op], input_op_slices,
        output_op_slices, aligned_op_slice_sizes, self.mock_op_reg_manager)

    self.mock_op_reg_manager.group_op_slices.assert_not_called()
    self.mock_op_reg_manager.process_ops.assert_called_once_with(
        [self.relu_op, self.conv_op])
    self.mock_op_reg_manager.process_ops_last.assert_called_once_with(
        [self.batch_norm_op])

  def testGetOpSize(self):
    # Verify correct size for regular ops.
    self.assertEqual(5, op_handler_util.get_op_size(self.relu2_op))
    self.assertEqual(6, op_handler_util.get_op_size(self.relu3_op))
    self.assertEqual(7, op_handler_util.get_op_size(self.relu4_op))

    # Verify correct size for ops with multiple outputs.
    split = tf.split(self.conv_op.outputs[0], [2, 3], axis=3)
    self.assertEqual(5, op_handler_util.get_op_size(split[0].op))

  def testSeparateSameSizeOps(self):
    op1 = tf.zeros([2, 4, 4, 3])
    op2 = tf.zeros([2, 4, 4, 3])
    op3 = tf.zeros([2, 4, 4, 5])
    op4 = tf.zeros([])
    op5 = tf.zeros([2, 4, 4, 3])
    op6 = tf.zeros([2, 4, 4, 2])
    all_ops = [op2.op, op3.op, op4.op, op5.op, op6.op]

    # Op2 and Op5 have matching sizes.  Op3 and Op6 have different sizes.  Op4
    # has size 0 and is dropped.
    expected_same_size_ops = [op2.op, op5.op]
    expected_different_size_ops = [op3.op, op6.op]

    same_size_ops, different_size_ops = (
        op_handler_util.separate_same_size_ops(op1.op, all_ops))

    # Verify lists of same size ops and different size ops.
    self.assertListEqual(expected_same_size_ops, same_size_ops)
    self.assertListEqual(expected_different_size_ops, different_size_ops)

  def testOpAssumptions(self):
    # Verify that op assumptions are true.  For example, verify that specific
    # inputs are at expected indices.
    conv_transpose = layers.conv2d_transpose(
        self.batch_norm_op.outputs[0], num_outputs=8, kernel_size=3,
        scope='conv_transpose')
    layers.separable_conv2d(
        conv_transpose, num_outputs=9, kernel_size=3, scope='dwise_conv')
    layers.fully_connected(tf.zeros([1, 7]), 10, scope='fc')

    g = tf.get_default_graph()

    # Verify that FusedBatchNormV3 has gamma as inputs[1].
    self.assertEqual('conv1/BatchNorm/gamma/read:0',
                     self.batch_norm_op.inputs[1].name)

    # Verify that Conv2D has weights at expected index.
    index = op_handler_util.WEIGHTS_INDEX_DICT[self.conv_op.type]
    self.assertEqual('conv1/weights/read:0',
                     self.conv_op.inputs[index].name)

    # Verify that Conv2DBackpropInput has weights at expected index.
    conv_transpose_op = g.get_operation_by_name(
        'conv_transpose/conv2d_transpose')
    index = op_handler_util.WEIGHTS_INDEX_DICT[conv_transpose_op.type]
    self.assertEqual('conv_transpose/weights/read:0',
                     conv_transpose_op.inputs[index].name)

    # Verify that DepthwiseConv2dNative has weights at expected index.
    depthwise_conv_op = g.get_operation_by_name(
        'dwise_conv/separable_conv2d/depthwise')
    index = op_handler_util.WEIGHTS_INDEX_DICT[depthwise_conv_op.type]
    self.assertEqual('dwise_conv/depthwise_weights/read:0',
                     depthwise_conv_op.inputs[index].name)

    # Verify that MatMul has weights at expected index.
    matmul_op = g.get_operation_by_name('fc/MatMul')
    index = op_handler_util.WEIGHTS_INDEX_DICT[matmul_op.type]
    self.assertEqual('fc/weights/read:0',
                     matmul_op.inputs[index].name)

  def testGroupMatch(self):
    # Verify that regex matches an op in the group.
    regex = 'BatchNorm'
    op_slices = [self.batch_norm_op_slice, self.conv_op_slice,
                 self.relu_op_slice]

    # Regex matches the batch norm.
    self.assertTrue(op_handler_util.group_match(regex, op_slices))

    # Remove the matching batch norm op.
    op_slices.pop(0)
    self.assertFalse(op_handler_util.group_match(regex, op_slices))

  def testGroupMatch_EmptyRegex(self):
    # Verify that empty regex does not match.
    regex = ''
    op_slices = [self.batch_norm_op_slice, self.conv_op_slice,
                 self.relu_op_slice]

    self.assertFalse(op_handler_util.group_match(regex, op_slices))


if __name__ == '__main__':
  tf.test.main()
