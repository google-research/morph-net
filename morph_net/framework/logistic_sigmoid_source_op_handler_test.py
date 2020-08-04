"""Tests for logistic_sigmoid_source_op_handler.

As the grouping functionality is virtually identical to other grouping op
handlers, such as the batch norm one, here we don't replicate all of the same
tests. We mainly focus on making sure the regularization vector is created
correctly.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl.testing import parameterized
import mock

from morph_net.framework import logistic_sigmoid_source_op_handler as ls_source_op_handler
from morph_net.framework import op_regularizer_manager as orm
from morph_net.probabilistic_gating import activation_gating

import scipy as sp
import tensorflow.compat.v1 as tf

from tensorflow.contrib import framework as contrib_framework
from tensorflow.contrib import layers as contrib_layers

arg_scope = contrib_framework.arg_scope
layers = contrib_layers


class LogisticSigmoidSourceOpHandlerTest(
    parameterized.TestCase, tf.test.TestCase):

  def setUp(self):
    super(LogisticSigmoidSourceOpHandlerTest, self).setUp()
    tf.reset_default_graph()

    # This tests a Conv2D -> ReLU -> LogisticSigmoidGating chain of ops.
    inputs = tf.zeros([2, 4, 4, 3])
    c1 = layers.conv2d(inputs, num_outputs=5, kernel_size=3, scope='conv1')
    activation_gating.logistic_sigmoid_gating(
        c1, axis=3, is_training=True)

    g = tf.get_default_graph()

    # Declare OpSlice and OpGroup for ops that are created in the test network.
    self.conv_op = g.get_operation_by_name('conv1/Conv2D')
    self.conv_op_slice = orm.OpSlice(self.conv_op, orm.Slice(0, 5))
    self.conv_op_group = orm.OpGroup(
        self.conv_op_slice, omit_source_op_slices=[self.conv_op_slice])

    self.relu_op = g.get_operation_by_name('conv1/Relu')
    self.relu_op_slice = orm.OpSlice(self.relu_op, orm.Slice(0, 5))
    self.relu_op_group = orm.OpGroup(self.relu_op_slice)

    self.activation_gating_op = g.get_operation_by_name(
        'logistic_sigmoid_gating/LogisticSigmoidGating')
    self.activation_gating_op_slice = orm.OpSlice(
        self.activation_gating_op, orm.Slice(0, 5))
    self.activation_gating_op_group = orm.OpGroup(
        self.activation_gating_op_slice)

    self.mask_logits_op = g.get_operation_by_name(
        'logistic_sigmoid_gating/mask_logits/Read/ReadVariableOp')
    self.mask_logits_op_slice = orm.OpSlice(
        self.mask_logits_op, orm.Slice(0, 5))
    self.mask_logits_op_group = orm.OpGroup(
        self.mask_logits_op_slice,
        omit_source_op_slices=[self.mask_logits_op_slice])

    self.multiply_op = g.get_operation_by_name(
        'logistic_sigmoid_gating/Identity')
    self.multiply_op_slice = orm.OpSlice(
        self.multiply_op, orm.Slice(0, 5))
    self.multiply_op_group = orm.OpGroup(
        self.multiply_op_slice,
        omit_source_op_slices=[self.multiply_op_slice])

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
        self.conv_op, self.relu_op, self.activation_gating_op, self.multiply_op]

  def testAssignGrouping_NoNeighborGroups(self):
    self.op_slice_dict = {
        self.activation_gating_op: [self.activation_gating_op_slice],
        self.conv_op: [self.conv_op_slice],
        self.multiply_op: [self.multiply_op_slice],
        self.relu_op: [self.relu_op_slice],
        self.mask_logits_op: [self.mask_logits_op_slice],
    }

    # No neighbor ops have groups.
    self.op_group_dict = {
        self.activation_gating_op_slice: self.activation_gating_op_group,
    }

    # Call handler to assign grouping.
    handler = ls_source_op_handler.LogisticSigmoidSourceOpHandler()
    handler.assign_grouping(self.activation_gating_op, self.mock_op_reg_manager)

    # Verify manager looks up op slice for ops of interest.
    self.mock_op_reg_manager.get_op_slices.assert_any_call(
        self.activation_gating_op)
    self.mock_op_reg_manager.get_op_slices.assert_any_call(self.multiply_op)

    # Verify manager creates OpGroup for activation gating op.
    self.mock_op_reg_manager.create_op_group_for_op_slice(
        self.activation_gating_op_slice)

    # Verify manager groups activation gating with inputs.
    self.mock_op_reg_manager.group_op_slices.assert_not_called()

    # Verify manager processes grouping for input ops.
    self.mock_op_reg_manager.process_ops.assert_called_once_with(
        [self.multiply_op])

  def testCreateRegularizerOnMask(self):
    # Call handler to create regularizer.
    handler = ls_source_op_handler.LogisticSigmoidSourceOpHandler(
        regularize_on_mask=True)
    regularizer = handler.create_regularizer(
        self.activation_gating_op_slice)

    # Verify regularizer is the mask tensor.
    g = tf.get_default_graph()
    mask_tensor = g.get_tensor_by_name(
        'logistic_sigmoid_gating/LogisticSigmoidGating:0')
    self.assertEqual(mask_tensor, regularizer._regularization_vector)

  def testCreateRegularizerOnLogits(self):
    # Call handler to create regularizer.
    handler = ls_source_op_handler.LogisticSigmoidSourceOpHandler(
        regularize_on_mask=False)
    regularizer = handler.create_regularizer(
        self.activation_gating_op_slice)

    # Verify regularizer is the mask tensor.
    g = tf.get_default_graph()
    logits_tensor = g.get_tensor_by_name(
        'logistic_sigmoid_gating/mask_logits/Read/ReadVariableOp:0')

    with self.cached_session() as sess:
      sess.run(tf.global_variables_initializer())
      logits_vec, reg_vec = sess.run(
          [logits_tensor, regularizer._regularization_vector])

      # Verify regularizer is the probability tensor induced by the logits.
      self.assertAllEqual(sp.special.expit(logits_vec), reg_vec)

  def testCreateRegularizer_OnMask_Sliced(self):
    # Call handler to create regularizer.
    handler = ls_source_op_handler.LogisticSigmoidSourceOpHandler(
        regularize_on_mask=True)
    activation_gating_op_slice = orm.OpSlice(
        self.activation_gating_op, orm.Slice(0, 3))
    regularizer = handler.create_regularizer(
        activation_gating_op_slice)

    g = tf.get_default_graph()
    mask_tensor = g.get_tensor_by_name(
        'logistic_sigmoid_gating/LogisticSigmoidGating:0')
    with self.cached_session() as sess:
      sess.run(tf.global_variables_initializer())
      mask, reg_vec = sess.run(
          [mask_tensor, regularizer._regularization_vector])

      # Verify regularizer is the sliced mask tensor.
      self.assertAllEqual(mask[0:3], reg_vec)

  def testCreateRegularizer_OnLogits_Sliced(self):
    # Call handler to create regularizer.
    handler = ls_source_op_handler.LogisticSigmoidSourceOpHandler(
        regularize_on_mask=False)
    activation_gating_op_slice = orm.OpSlice(
        self.activation_gating_op, orm.Slice(0, 3))
    regularizer = handler.create_regularizer(
        activation_gating_op_slice)

    g = tf.get_default_graph()
    logits_tensor = g.get_tensor_by_name(
        'logistic_sigmoid_gating/mask_logits/Read/ReadVariableOp:0')
    with self.cached_session() as sess:
      sess.run(tf.global_variables_initializer())
      logits, reg_vec = sess.run(
          [logits_tensor, regularizer._regularization_vector])

      # Verify regularizer is the sliced probability tensor.
      self.assertAllEqual(sp.special.expit(logits[0:3]), reg_vec)

if __name__ == '__main__':
  tf.test.main()
