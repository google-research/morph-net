"""Tests for morph_net.network_regularizers.activation_regularizer."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl.testing import parameterized
from morph_net.network_regularizers import activation_regularizer
import numpy as np
import tensorflow.compat.v1 as tf
from tensorflow.contrib import slim as contrib_slim

slim = contrib_slim


class ActivationLossTest(parameterized.TestCase, tf.test.TestCase):

  def setUp(self):
    super(ActivationLossTest, self).setUp()
    tf.reset_default_graph()
    self.build_with_batch_norm()
    with self.cached_session():
      self.init()

  def build_with_batch_norm(self):
    params = {
        'trainable': True,
        'normalizer_fn': slim.batch_norm,
        'normalizer_params': {
            'scale': True
        }
    }

    with slim.arg_scope([slim.layers.conv2d], **params):
      self.build_model()

  def build_model(self):
    image = tf.constant(0.0, shape=[1, 17, 19, 3])
    conv1 = slim.layers.conv2d(image, 13, [7, 5], padding='SAME', scope='conv1')
    conv2 = slim.layers.conv2d(image, 23, [1, 1], padding='SAME', scope='conv2')
    concat = tf.concat([conv1, conv2], 3)
    self.conv3 = slim.layers.conv2d(
        concat, 29, [3, 3], stride=2, padding='SAME', scope='conv3')
    self.conv4 = slim.layers.conv2d(
        concat, 31, [1, 1], stride=1, padding='SAME', scope='conv4')
    self.name_to_var = {v.op.name: v for v in tf.global_variables()}

    self.gamma_activation_reg = (
        activation_regularizer.GammaActivationRegularizer(
            [self.conv3.op, self.conv4.op], gamma_threshold=0.45))

  def get_conv(self, name):
    return tf.get_default_graph().get_operation_by_name(name +  '/Conv2D')

  def init(self):
    tf.global_variables_initializer().run()
    gamma1 = self.name_to_var['conv1/BatchNorm/gamma']
    gamma1.assign([0.8] * 7 + [0.2] * 6).eval()
    gamma2 = self.name_to_var['conv2/BatchNorm/gamma']
    gamma2.assign([-0.7] * 11 + [0.1] * 12).eval()
    gamma3 = self.name_to_var['conv3/BatchNorm/gamma']
    gamma3.assign([0.6] * 10 + [-0.3] * 19).eval()
    gamma4 = self.name_to_var['conv4/BatchNorm/gamma']
    gamma4.assign([-0.5] * 17 + [-0.4] * 14).eval()

  def cost(self, conv):
    with self.cached_session():
      return self.gamma_activation_reg.get_cost(conv).eval()

  def loss(self, conv):
    with self.cached_session():
      try:
        return self.gamma_activation_reg.get_regularization_term(conv).eval()
      except AttributeError:
        print (str(conv))
        print ('getloss', self.gamma_activation_reg.get_regularization_term(
            conv))

  @parameterized.named_parameters(
      ('_1', 'conv1', 7),  # 7 is the number of values > 0.45 in conv1.
      ('_2', 'conv2', 11),  # 11 is the number of values > 0.45 in conv2.
      ('_3', 'conv3', 10),  # 10 is the number of values > 0.45 in conv3.
      ('_4', 'conv4', 17))  # 17 is the number of values > 0.45 in conv4.
  def test_cost(self, conv_name, expected_cost):
    conv = self.get_conv(conv_name)
    self.assertEqual(expected_cost, self.cost([conv]))

  def test_list_cost(self):
    # Test that passing a list of convs sums their contributions:
    convs = [self.get_conv('conv3'), self.get_conv('conv4')]
    self.assertEqual(
        self.cost(convs[1:]) + self.cost(convs[:1]), self.cost(convs))

  @parameterized.named_parameters(  # Expected values are sum(abs(gamma)).
      ('_1', 'conv1', 6.8),
      ('_2', 'conv2', 8.9),
      ('_3', 'conv3', 11.7),
      ('_4', 'conv4', 14.1))
  def test_loss(self, conv_name, expected_loss):
    self.assertAllClose(expected_loss, self.loss([self.get_conv(conv_name)]))

  def test_loss_gradient_conv2(self):
    loss = self.gamma_activation_reg.get_regularization_term(
        [self.get_conv('conv2')])
    expected_grad = np.array([-1.0] * 11 + [1.0] * 12)
    gammas = [
        self.name_to_var['conv%d/BatchNorm/gamma' % i] for i in range(1, 5)
    ]

    # Although the loss associated with conv2 depends on the gammas of conv2,
    # conv3 and conv4, only gamma2 should receive graients. The other gammas are
    # compared to the threshold to see if they are alive or not, but this should
    # not send gradients to them.
    grads = tf.gradients(loss, gammas)
    self.assertEqual(None, grads[0])
    self.assertEqual(None, grads[2])
    self.assertEqual(None, grads[3])

    # Regarding gamma2, it receives a -1 or a +1 gradient, depending on whether
    # the gamma is negative or positive, since the loss is |gamma|. This is
    # multiplied by expected_coeff.
    with self.cached_session():
      self.assertAllClose(expected_grad, grads[1].eval())

  def test_conv_has_no_gamma(self):
    conv5 = slim.layers.conv2d(
        self.conv3, 11, [1, 1], stride=1, padding='SAME', scope='conv5')
    self.gamma_activation_reg = (
        activation_regularizer.GammaActivationRegularizer(
            [conv5.op], gamma_threshold=0.45))

    # Sanity check regarding conv3.
    self.assertAllClose(11.7, self.loss([self.get_conv('conv3')]))

    # Conv5 has 11 outputs.
    conv = self.get_conv('conv5')
    self.assertEqual(11, self.cost([conv]))


if __name__ == '__main__':
  tf.test.main()
