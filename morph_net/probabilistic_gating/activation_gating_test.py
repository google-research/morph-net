"""Tests for probabilistic_gating.activation_gating."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl.testing import parameterized
from morph_net.probabilistic_gating import activation_gating

import numpy as np

import tensorflow.compat.v1 as tf
# pylint: disable=g-direct-tensorflow-import
from tensorflow.contrib import layers as contrib_layers
# pylint: enable=g-direct-tensorflow-import

ag = activation_gating


class ActivationGatingTest(parameterized.TestCase, tf.test.TestCase):

  def _build_activation(self):
    activation_shape = [1, 17, 19, 4]
    activation = tf.ones(shape=activation_shape)
    return activation

  def test_no_trainable(self):
    activation = self._build_activation()
    gated_activation = ag.logistic_sigmoid_gating(
        activation, 3, is_training=False, log_odds_init=1000.0)

    with self.session() as sess:
      sess.run(tf.global_variables_initializer())
      gated_activation, = sess.run([gated_activation])

    self.assertAllEqual(gated_activation, activation, msg='gated activation')

  def test_op_type_name(self):
    activation = self._build_activation()
    _ = ag.logistic_sigmoid_gating(
        activation, axis=3, is_training=True)

    g = tf.get_default_graph()
    ops = [
        op for op in g.get_operations() if op.type == 'LogisticSigmoidGating']
    # Verify only one such OP exists in the graph.
    self.assertLen(ops, 1)

  def test_inputs_outputs(self):
    axis = 3
    activation = self._build_activation()
    _ = ag.logistic_sigmoid_gating(
        activation, axis=axis, is_training=True)

    g = tf.get_default_graph()
    op = g.get_operation_by_name(
        'logistic_sigmoid_gating/LogisticSigmoidGating')

    # Verify Variable is created.
    logits = op.inputs[0]
    logits_shape = logits.shape.as_list()
    self.assertLen(logits_shape, 1)
    self.assertEqual(logits_shape[0], activation.shape.as_list()[axis])

    # Verify output
    mask_shape = op.outputs[0].shape.as_list()
    self.assertLen(mask_shape, 1)
    self.assertEqual(mask_shape[0], activation.shape.as_list()[axis])

  @parameterized.named_parameters(('_1', 5.5, 0.996, True),
                                  ('_2', 0.5, 0.6, True),
                                  ('_3', -5.5, 0.004, True),
                                  ('_4', 5.5, 0.996, False),
                                  ('_5', 0.5, 0.6, False),
                                  ('_6', -5.5, 0.004, False),
                                  ('_7', -1000.0, 0.0, False),
                                  ('_8', 1000.0, 1.0, False))
  def test_sampled_mask(self, logits, expected_percent_on, straight_through):
    logits = tf.constant(logits, shape=[1000])
    mask = ag._logistic_sigmoid_sample(
        logits, 0.01, straight_through=straight_through)

    with self.session() as sess:
      sess.run(tf.global_variables_initializer())
      mask = mask.eval()

    avg_mask = np.average(mask)
    self.assertAlmostEqual(avg_mask, expected_percent_on, delta=0.07)

  @parameterized.named_parameters(('_st', True),
                                  ('_no_st', False))
  def test_add_gating_to_fn(self, straight_through):
    activation = tf.reshape(tf.eye(16, 16), shape=[1, 16, 16, 1])
    activation = tf.concat([activation, activation, activation], axis=3)

    gated_bn = activation_gating.gated_batch_norm(
        log_odds_init=100, straight_through=straight_through)
    gated_bn_activation = gated_bn(activation)

    bn_activation = contrib_layers.batch_norm(activation)

    g = tf.get_default_graph()
    op_names = [op.name for op in g.get_operations()]
    # Asserting that the correct ops have been plaed in the graph.
    self.assertIn(
        'BatchNorm/FusedBatchNormV3', op_names, msg='batch_norm_op_check')
    self.assertIn(
        'logistic_sigmoid_gating/LogisticSigmoidGating',
        op_names, msg='batch_norm_op_check')

    # Check values.
    with self.session() as sess:
      sess.run(tf.global_variables_initializer())
      self.assertAllClose(bn_activation, gated_bn_activation)


if __name__ == '__main__':
  tf.test.main()
