"""Tests for regularizers.framework.group_lasso_regularizer."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl.testing import parameterized
from morph_net.op_regularizers import group_lasso_regularizer
import numpy as np
import tensorflow.compat.v1 as tf
from tensorflow.contrib import framework as contrib_framework
from tensorflow.contrib import layers as contrib_layers

layers = contrib_layers

ALIVE_THRESHOLD = 1.0


def assert_not_all_are_alive_or_dead(alive_vector):
  assert not all(alive_vector), (
      'All activations are alive, test case is trivial. Increase threshold')
  assert any(alive_vector), (
      'All activations are dead, test case is trivial. Decrease threshold')


class GroupLassoRegularizerTest(parameterized.TestCase, tf.test.TestCase):

  def setUp(self):
    tf.reset_default_graph()
    tf.set_random_seed(7907)
    with contrib_framework.arg_scope(
        [layers.conv2d, layers.conv2d_transpose],
        weights_initializer=tf.random_normal_initializer):
      self.BuildModel()
    with self.cached_session():
      tf.global_variables_initializer().run()

  def BuildModel(self):
    image = tf.constant(0.0, shape=[1, 17, 19, 3])
    conv = layers.conv2d(image, 13, [7, 5], padding='SAME', scope='conv')
    layers.conv2d_transpose(conv, 11, [5, 5], scope='convt')

  # For Conv2D the reduction indices for group lasso are (0, 1, 2).
  # For Conv2DBackpropInput (aka conv2d transpose) they are (0, 1, 3).
  @parameterized.named_parameters(
      ('_regular_conv', 'conv/Conv2D', (0, 1, 2), 0.0),
      ('_transpose_conv', 'convt/conv2d_transpose', (0, 1, 3), 0.0),
      ('_regular_conv_l10.5', 'conv/Conv2D', (0, 1, 2), 0.5))
  def testOp(self, op_name, reduce_dims, l1_fraction):
    op = tf.get_default_graph().get_operation_by_name(op_name)
    with self.cached_session():
      weights = op.inputs[1].eval()
    l1_reg_vector = np.mean(np.abs(weights), axis=reduce_dims)
    l2_reg_vector = np.sqrt(np.mean(weights**2, axis=reduce_dims))
    expected_reg_vector = (
        l1_fraction * l1_reg_vector + (1.0 - l1_fraction) * l2_reg_vector)

    # We choose the threshold at the expectation value, so that some activations
    # end up above threshold and others end up below. The weights are normally
    # distributed, so the L2 norm is 1.0, and the L1 norm is sqrt(2/pi).
    # With a general l1_fraction, we compute a weighted average of the two:
    threshold = (1.0 - l1_fraction) + l1_fraction * np.sqrt(2 / np.pi)
    expected_alive = expected_reg_vector > threshold
    assert_not_all_are_alive_or_dead(expected_alive)

    conv_reg = (
        group_lasso_regularizer.GroupLassoRegularizer(
            weight_tensor=op.inputs[1], reduce_dims=reduce_dims,
            threshold=threshold, l1_fraction=l1_fraction))

    with self.cached_session():
      actual_reg_vector = conv_reg.regularization_vector.eval()
      actual_alive = conv_reg.alive_vector.eval()

    self.assertAllClose(expected_reg_vector, actual_reg_vector)
    self.assertAllEqual(expected_alive, actual_alive)


class GroupLassoRegularizerMatMulTest(parameterized.TestCase, tf.test.TestCase):
  """Unit tests for MatMul op with GroupLassoRegularizer."""

  def setUp(self):
    tf.reset_default_graph()
    tf.set_random_seed(7907)
    with contrib_framework.arg_scope(
        [layers.fully_connected],
        weights_initializer=tf.random_normal_initializer):
      # Build the model.
      input_data = tf.constant(0.0, shape=[1, 3])
      layers.fully_connected(input_data, 4, scope='fc')
    with self.cached_session():
      tf.global_variables_initializer().run()

  @parameterized.named_parameters(('_fully_connected_no_l1', 0.0),
                                  ('_fully_connected_l1_0.5', 0.5))
  def testMatMulOp(self, l1_fraction):
    op = tf.get_default_graph().get_operation_by_name('fc/MatMul')
    with self.cached_session():
      weights = op.inputs[1].eval()
    l1_reg_vector = np.mean(np.abs(weights), axis=0)
    l2_reg_vector = np.sqrt(np.mean(weights**2, axis=0))
    expected_reg_vector = (
        l1_fraction * l1_reg_vector + (1.0 - l1_fraction) * l2_reg_vector)
    # We choose the threshold at the expectation value, so that some activations
    # end up above threshold and others end up below. The weights are normally
    # distributed, so the L2 norm is 1.0, and the L1 norm is sqrt(2/pi).
    # With a general l1_fraction, we compute a weighted average of the two:
    threshold = (1.0 - l1_fraction) + l1_fraction * np.sqrt(2 / np.pi)
    expected_alive = expected_reg_vector > threshold
    assert_not_all_are_alive_or_dead(expected_alive)

    matmul_reg = (
        group_lasso_regularizer.GroupLassoRegularizer(
            weight_tensor=op.inputs[1], reduce_dims=(0,),
            threshold=threshold, l1_fraction=l1_fraction))
    with self.cached_session():
      actual_reg_vector = matmul_reg.regularization_vector.eval()
      actual_alive = matmul_reg.alive_vector.eval()

    self.assertAllClose(actual_reg_vector, expected_reg_vector)
    self.assertAllEqual(actual_alive, expected_alive)


if __name__ == '__main__':
  tf.test.main()
