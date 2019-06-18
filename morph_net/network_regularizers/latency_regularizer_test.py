"""Tests for network_regularizers.latency_regularizer."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from absl.testing import parameterized
from morph_net.network_regularizers import flop_regularizer
from morph_net.network_regularizers import latency_regularizer
from morph_net.network_regularizers import model_size_regularizer
from morph_net.network_regularizers import resource_function
import tensorflow as tf


from tensorflow.contrib.slim.nets import inception

slim = tf.contrib.slim

NUM_CHANNELS = 3
HARDWARE = 'P100'


class LatencyRegularizerTest(parameterized.TestCase, tf.test.TestCase):

  def build_with_batch_norm(self, fused):
    params = {
        'trainable': True,
        'normalizer_fn': slim.batch_norm,
        'normalizer_params': {
            'scale': True,
            'fused': fused,
        }
    }

    with slim.arg_scope([slim.layers.conv2d], **params):
      self.build_model()
    with self.cached_session():
      self.init()

  def build_model(self):
    # Our test model is:
    #
    #         -> conv1 --+     -> conv3 -->
    #        /           |    /
    #  image          [concat]
    #        \           |    \
    #         -> conv2 --+     -> conv4 -->
    #
    # (the model has two "outputs", conv3 and conv4).
    #
    image = tf.constant(0.0, shape=[1, 17, 19, NUM_CHANNELS])
    conv1 = slim.layers.conv2d(image, 13, [7, 5], padding='SAME', scope='conv1')
    conv2 = slim.layers.conv2d(image, 23, [1, 1], padding='SAME', scope='conv2')
    concat = tf.concat([conv1, conv2], 3)
    self.conv3 = slim.layers.conv2d(
        concat, 29, [3, 3], stride=2, padding='SAME', scope='conv3')
    self.conv4 = slim.layers.conv2d(
        concat, 31, [1, 1], stride=1, padding='SAME', scope='conv4')
    self.name_to_var = {v.op.name: v for v in tf.global_variables()}

    self.regularizer = latency_regularizer.GammaLatencyRegularizer(
        [self.conv3.op, self.conv4.op],
        gamma_threshold=0.45, hardware=HARDWARE)

  def get_conv(self, name):
    return tf.get_default_graph().get_operation_by_name(name + '/Conv2D')

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

  def get_cost(self, conv):
    with self.cached_session():
      return self.regularizer.get_cost(conv).eval()

  def get_loss(self, conv):
    with self.cached_session():
      return self.regularizer.get_regularization_term(conv).eval()

  def test_cost(self):
    self.build_with_batch_norm(True)
    # Conv1 has 7 gammas above 0.45, and NUM_CHANNELS inputs (from the image).
    conv = self.get_conv('conv1')
    # FLOPs = 2 * NHWRSCK
    expected_flops = 2 * 17 * 19 * 7 * 5 * 3 * 7
    expected_cost = (
        float(expected_flops) / resource_function.PEAK_COMPUTE[HARDWARE])
    self.assertAllClose(expected_cost, self.get_cost([conv]))
    self.assertAllClose(51.0548400879, self.get_cost([conv]))

    # Conv2 has 11 gammas above 0.45, and NUM_CHANNELS inputs (from the image).
    conv = self.get_conv('conv2')
    # Memory = (input_size + weight_size + output_size) * dtype_size
    #        = (NHWC + RSCK + NHWK) * 4
    expected_memory = (17 * 19 * 3 + 3 * 11 + 17 * 19 * 11) * 4
    expected_cost = (
        float(expected_memory) / resource_function.MEMORY_BANDWIDTH[HARDWARE])
    self.assertAllClose(expected_cost, self.get_cost([conv]))
    self.assertAllClose(24.8907108307, self.get_cost([conv]))

    # Conv3 has 10 gammas above 0.45, and 7 + 11 inputs from conv1 and conv2.
    conv = self.get_conv('conv3')
    # Memory = (input_size + weight_size + output_size) * dtype_size
    #        = (NHWC + RSCK + NHWK) * 4
    expected_memory = (17 * 19 * 18 + 3 * 3 * 18 * 10 + 9 * 10 * 10) * 4
    expected_cost3 = (
        float(expected_memory) / resource_function.MEMORY_BANDWIDTH[HARDWARE])
    self.assertAllClose(expected_cost3, self.get_cost([conv]))
    self.assertAllClose(45.5409851074, self.get_cost([conv]))

    # Conv4 has 17 gammas above 0.45, and 7 + 11 inputs from conv1 and conv2.
    conv = self.get_conv('conv4')
    # Memory = (input_size + weight_size + output_size) * dtype_size
    #        = (NHWC + RSCK + NHWK) * 4
    expected_memory = (17 * 19 * 18 + 18 * 17 + 17 * 19 * 17) * 4
    expected_cost4 = (
        float(expected_memory) / resource_function.MEMORY_BANDWIDTH[HARDWARE])
    self.assertAllClose(expected_cost4, self.get_cost([conv]))
    self.assertAllClose(63.4480857849, self.get_cost([conv]))

    # Test that passing a list of convs sums their contributions:
    convs = [self.get_conv('conv3'), self.get_conv('conv4')]
    self.assertAllClose(expected_cost3 + expected_cost4, self.get_cost(convs))
    self.assertAllClose(108.989074707, self.get_cost(convs))

  def testLossRegression(self):
    self.build_with_batch_norm(True)
    g = tf.get_default_graph()
    all_convs = [o for o in g.get_operations() if o.type == 'Conv2D']

    # FLOP loss = 2 * NHWRS * (alive_in * reg_out + alive_out * reg_in)
    conv1_loss = (
        2 * 17 * 19 * 7 * 5 * (3 * 6.8 + 7 * 0)
        / resource_function.PEAK_COMPUTE[HARDWARE])
    self.assertAllClose(conv1_loss, self.get_loss([self.get_conv('conv1')]))

    # Memory loss = (HW * reg_in + HW * reg_out) * dtype_size
    # Note that reg_in = reg_out for pass-through ops such as Relu.
    relu1_loss = ((17 * 19 * 6.8 + 17 * 19 * 6.8) * 4
                  / resource_function.MEMORY_BANDWIDTH[HARDWARE])
    batch_norm1_loss = relu1_loss
    relu1_op = g.get_operation_by_name('conv1/Relu')
    self.assertAllClose(relu1_loss, self.get_loss([relu1_op]))

    # Memory loss = input_tensor + weight_tensor + output_tensor
    #             = (HW * reg_in
    #                + RS * (alive_in * reg_out + alive_out * reg_in)
    #                + HW * reg_out) * dtype_size
    conv2_loss = ((17 * 19 * 0 + (3 * 8.9 + 11 * 0) + 17 * 19 * 8.9) * 4
                  / resource_function.MEMORY_BANDWIDTH[HARDWARE])
    self.assertAllClose(conv2_loss, self.get_loss([self.get_conv('conv2')]))

    relu2_loss = ((17 * 19 * 8.9) * 2 * 4
                  / resource_function.MEMORY_BANDWIDTH[HARDWARE])
    batch_norm2_loss = relu2_loss
    relu2_op = g.get_operation_by_name('conv2/Relu')
    self.assertAllClose(relu2_loss, self.get_loss([relu2_op]))

    # Memory loss = input_tensor + output_tensor
    #             = HW * reg_in + HW * reg_out
    # Note that reg_in = reg_out for concat.
    concat_loss = ((17 * 19 * 15.7 + 17 * 19 * 15.7) * 4
                   / resource_function.MEMORY_BANDWIDTH[HARDWARE])
    concat_op = g.get_operation_by_name('concat')
    self.assertAllClose(concat_loss, self.get_loss([concat_op]))

    conv3_loss = ((17 * 19 * 15.7
                   + 3 * 3 * (18 * 11.7 + 10 * 15.7)
                   + 9 * 10 * 11.7) * 4
                  / resource_function.MEMORY_BANDWIDTH[HARDWARE])
    self.assertAllClose(conv3_loss, self.get_loss([self.get_conv('conv3')]))

    relu3_loss = ((9 * 10 * 11.7) * 2 * 4
                  / resource_function.MEMORY_BANDWIDTH[HARDWARE])
    batch_norm3_loss = relu3_loss
    relu3_op = g.get_operation_by_name('conv3/Relu')
    self.assertAllClose(relu3_loss, self.get_loss([relu3_op]))

    conv4_loss = ((17 * 19 * 15.7
                   + (18 * 14.1 + 17 * 15.7)
                   + 17 * 19 * 14.1) * 4
                  / resource_function.MEMORY_BANDWIDTH[HARDWARE])
    self.assertAllClose(conv4_loss, self.get_loss([self.get_conv('conv4')]))

    relu4_loss = ((17 * 19 * 14.1) * 2 * 4
                  / resource_function.MEMORY_BANDWIDTH[HARDWARE])
    batch_norm4_loss = relu4_loss
    relu4_op = g.get_operation_by_name('conv4/Relu')
    self.assertAllClose(relu4_loss, self.get_loss([relu4_op]))

    conv_losses = conv1_loss + conv2_loss + conv3_loss + conv4_loss
    other_losses = (concat_loss
                    + relu1_loss + batch_norm1_loss
                    + relu2_loss + batch_norm2_loss
                    + relu3_loss + batch_norm3_loss
                    + relu4_loss + batch_norm4_loss)
    self.assertAllClose(conv_losses, self.get_loss(all_convs))
    self.assertAllClose(conv_losses + other_losses, self.get_loss([]))

  @parameterized.named_parameters(
      ('_P100', 'P100'),
      ('_V100', 'V100'))
  def testInceptionV2(self, hardware):
    image = tf.zeros([1, 112, 112, 3])
    net, _ = inception.inception_v2_base(image)
    g = tf.get_default_graph()
    self.regularizer = latency_regularizer.GammaLatencyRegularizer(
        [net.op], gamma_threshold=0.5, hardware=hardware)

    # Compute-bound convolution.
    op = g.get_operation_by_name(
        'InceptionV2/Mixed_3c/Branch_2/Conv2d_0c_3x3/Conv2D')
    # FLOP cost = 2 * NHWRSCK
    expected_cost = (2 * 14 * 14 * 3 * 3 * 96 * 96
                     / resource_function.PEAK_COMPUTE[hardware])
    self.assertAllClose(expected_cost, self.get_cost([op]))

    # Memory-bound convolution.
    op = g.get_operation_by_name(
        'InceptionV2/Conv2d_1a_7x7/separable_conv2d')
    # Memory cost = input_tensor + weight_tensor + output_tensor
    #             = NHWC + RSCK + NHWK
    # Note that this is a pointwise convolution with kernel 1x1.
    for i in op.inputs:
      print(i)
    expected_cost = ((56 * 56 * 24 + 24 * 64 + 56 * 56 * 64) * 4
                     / resource_function.MEMORY_BANDWIDTH[hardware])
    self.assertAllClose(expected_cost, self.get_cost([op]))

  def testInceptionV2_TotalCost(self):
    conv_params = {
        'activation_fn': tf.nn.relu6,
        'weights_regularizer': tf.contrib.layers.l2_regularizer(0.00004),
        'weights_initializer': tf.random_normal_initializer(stddev=0.03),
        'trainable': True,
        'biases_initializer': tf.constant_initializer(0.0),
        'normalizer_fn': tf.contrib.layers.batch_norm,
        'normalizer_params': {
            'is_training': False,
            'decay': 0.9997,
            'scale': True,
            'epsilon': 0.001,
        }
    }

    tf.reset_default_graph()
    with slim.arg_scope([slim.layers.conv2d, slim.layers.separable_conv2d],
                        **conv_params):
      # Build model.
      image = tf.zeros([1, 224, 224, 3])
      net, _ = inception.inception_v2_base(image)
      logits = slim.layers.fully_connected(
          net,
          1001,
          activation_fn=None,
          scope='logits',
          weights_initializer=tf.random_normal_initializer(stddev=1e-3),
          biases_initializer=tf.constant_initializer(0.0))

    # Instantiate regularizers.
    flop_reg = flop_regularizer.GammaFlopsRegularizer(
        [logits.op], gamma_threshold=0.5)
    p100_reg = latency_regularizer.GammaLatencyRegularizer(
        [logits.op], gamma_threshold=0.5, hardware='P100')
    v100_reg = latency_regularizer.GammaLatencyRegularizer(
        [logits.op], gamma_threshold=0.5, hardware='V100')
    model_size_reg = model_size_regularizer.GammaModelSizeRegularizer(
        [logits.op], gamma_threshold=0.5)

    with self.cached_session():
      tf.global_variables_initializer().run()

    # Verify costs are expected.
    self.assertAllClose(3.86972e+09, flop_reg.get_cost())
    self.assertAllClose(517536.0, p100_reg.get_cost())
    self.assertAllClose(173330.453125, v100_reg.get_cost())
    self.assertAllClose(1.11684e+07, model_size_reg.get_cost())


if __name__ == '__main__':
  tf.test.main()
