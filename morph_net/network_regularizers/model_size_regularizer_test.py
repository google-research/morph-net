"""Tests for network_regularizers.model_size_regularizer."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from absl.testing import parameterized
from morph_net.network_regularizers import model_size_regularizer
from morph_net.network_regularizers import resource_function
from morph_net.testing import dummy_decorator
import tensorflow.compat.v1 as tf
from tensorflow.contrib import slim as contrib_slim

slim = contrib_slim

_coeff = resource_function.num_weights_coeff
NUM_CHANNELS = 3


class GammaModelSizeDecoratedTest(parameterized.TestCase, tf.test.TestCase):
  """Test op regularizer decorator with model size regularizer."""

  def cost(self, conv):
    with self.cached_session():
      return self.gamma_flop_reg.get_cost(conv).eval()

  def loss(self, conv):
    with self.cached_session():
      return self.gamma_flop_reg.get_regularization_term(conv).eval()

  def get_conv(self, name):
    return tf.get_default_graph().get_operation_by_name(name + '/Conv2D')

  def testLossCostDecorated(self):
    params = {'trainable': True, 'normalizer_fn': slim.batch_norm,
              'normalizer_params': {'scale': True}}

    with slim.arg_scope([slim.layers.conv2d], **params):
      image = tf.constant(0.0, shape=[1, 1, 1, NUM_CHANNELS])
      conv1 = slim.layers.conv2d(
          image, 2, [1, 1], padding='SAME', scope='conv1')
    with self.cached_session():
      tf.global_variables_initializer().run()
      name_to_var = {v.op.name: v for v in tf.global_variables()}
      gamma1 = name_to_var['conv1/BatchNorm/gamma']
      gamma1.assign([1] * 2).eval()

    self.gamma_flop_reg = model_size_regularizer.GammaModelSizeRegularizer(
        [conv1.op],
        gamma_threshold=0.1,
        regularizer_decorator=dummy_decorator.DummyDecorator,
        decorator_parameters={'scale': 0.5})

    conv = self.get_conv('conv1')
    self.assertEqual(_coeff(conv) * 3 * 1, self.loss([conv]))
    self.assertEqual(_coeff(conv) * 2 * NUM_CHANNELS, self.cost([conv]))

if __name__ == '__main__':
  tf.test.main()
