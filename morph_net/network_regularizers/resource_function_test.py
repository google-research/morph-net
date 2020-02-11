"""Tests for network_regularizers.resource_function."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from absl.testing import parameterized
from morph_net.network_regularizers import resource_function
import numpy as np
import tensorflow.compat.v1 as tf
from tensorflow.contrib import layers as contrib_layers
# pylint: disable=g-direct-tensorflow-import
from tensorflow.python.framework import ops

layers = contrib_layers


class ResourceFunctionTest(parameterized.TestCase, tf.test.TestCase):

  def assertNearRelatively(self, expected, actual):
    self.assertNear(expected, actual, expected * 1e-6)

  def setUp(self):
    super(ResourceFunctionTest, self).setUp()

    self.image_shape = (1, 11, 13, 17)
    self.image = tf.placeholder(tf.float32, shape=[1, None, None, 17])
    net = layers.conv2d(
        self.image, 19, [7, 5], stride=2, padding='SAME', scope='conv1')
    layers.conv2d_transpose(
        self.image, 29, [7, 5], stride=2, padding='SAME', scope='convt2')
    net = tf.reduce_mean(net, axis=(1, 2))
    layers.fully_connected(net, 23, scope='FC')
    net = layers.conv2d(
        self.image, 10, [7, 5], stride=2, padding='SAME', scope='conv2')
    layers.separable_conv2d(
        net, None, [3, 2], depth_multiplier=1, padding='SAME', scope='dw1')

    self.video_shape = (1, 11, 9, 13, 17)
    self.video = tf.placeholder(tf.float32, shape=[1, None, None, None, 17])
    net = layers.conv3d(
        self.video, 19, [7, 3, 5], stride=2, padding='SAME', scope='vconv1')
    g = tf.get_default_graph()
    self.conv_op = g.get_operation_by_name('conv1/Conv2D')
    self.convt_op = g.get_operation_by_name(
        'convt2/conv2d_transpose')
    self.matmul_op = g.get_operation_by_name('FC/MatMul')
    self.dw_op = g.get_operation_by_name('dw1/depthwise')
    self.conv3d_op = g.get_operation_by_name(
        'vconv1/Conv3D')

  @parameterized.named_parameters(
      ('_BatchSize1_AliveIn17_AliveOut19', 1, 17, 19),
      ('_BatchSize32_AliveIn4_AliveOut9', 32, 4, 9))
  def testConvFlopFunction_Cost(
      self, batch_size, num_alive_inputs, num_alive_outputs):
    flop_cost_tensor = resource_function.flop_function(
        self.conv_op, False, num_alive_inputs, num_alive_outputs, 17, 19,
        batch_size)
    with self.session() as sess:
      sess.run(tf.global_variables_initializer())
      flop_cost, _ = sess.run(
          [flop_cost_tensor, self.image],
          feed_dict={self.image: np.zeros(self.image_shape)})

    # Expected FLOP cost =
    # 2 * batch_size * feature_map_width * feature_map_height
    # * kernel_width * kernel_height * input_depth * output_depth
    expected_flop_cost = (
        2 * batch_size * 6 * 7 * 7 * 5 * num_alive_inputs * num_alive_outputs)
    self.assertEqual(expected_flop_cost, flop_cost)

  @parameterized.named_parameters(
      ('_BatchSize1_AliveIn17_AliveOut19_RegIn17_RegOut19',
       1, 17, 19, 17, 19),
      ('_BatchSize32_AliveIn4_AliveOut9_RegIn3_RegOut7',
       32, 4, 9, 3, 7))
  def testConvFlopFunction_Regularization(
      self, batch_size, num_alive_inputs, num_alive_outputs, reg_inputs,
      reg_outputs):
    flop_loss_tensor = resource_function.flop_function(
        self.conv_op, True, num_alive_inputs, num_alive_outputs, reg_inputs,
        reg_outputs, batch_size)
    with self.session() as sess:
      sess.run(tf.global_variables_initializer())
      flop_loss, _ = sess.run(
          [flop_loss_tensor, self.image],
          feed_dict={self.image: np.zeros(self.image_shape)})

    # Expected FLOP regularization loss =
    # 2 * batch_size * feature_map_width * feature_map_height
    # * kernel_width * kernel_height
    # * (num_alive_inputs * reg_outputs + num_alive_outputs * reg_inputs)
    expected_flop_loss = (
        2 * batch_size * 6 * 7 * 7 * 5 * (
            num_alive_inputs * reg_outputs + num_alive_outputs * reg_inputs))
    self.assertEqual(expected_flop_loss, flop_loss)

  @parameterized.named_parameters(
      ('_BatchSize1_AliveIn17_AliveOut19', 1, 17, 19),
      ('_BatchSize32_AliveIn4_AliveOut9', 32, 4, 9))
  def testConvMemoryFunction_Cost(
      self, batch_size, num_alive_inputs, num_alive_outputs):
    memory_cost_tensor = resource_function.memory_function(
        self.conv_op, False, num_alive_inputs, num_alive_outputs, 17, 19,
        batch_size)
    with self.session() as sess:
      sess.run(tf.global_variables_initializer())
      memory_cost, _ = sess.run(
          [memory_cost_tensor, self.image],
          feed_dict={self.image: np.zeros(self.image_shape)})

    # Expected memory cost = input_feature + weights + output_feature =
    # (batch_size * feature_map_width * feature_map_height * num_alive_inputs
    # + kernel_width * kernel_height * num_alive_inputs * num_alive_outputs
    # + batch_size * feature_map_width * feature_map_height * num_alive_outputs)
    # * dtype.size
    expected_memory_cost = (
        batch_size * 11 * 13 * num_alive_inputs
        + 7 * 5 * num_alive_inputs * num_alive_outputs
        + batch_size * 6 * 7 * num_alive_outputs) * self.image.dtype.size
    self.assertEqual(expected_memory_cost, memory_cost)

  @parameterized.named_parameters(
      ('_BatchSize1_AliveIn17_AliveOut19_RegIn17_RegOut19',
       1, 17, 19, 17, 19),
      ('_BatchSize32_AliveIn4_AliveOut9_RegIn3_RegOut7',
       32, 4, 9, 3, 7))
  def testConvMemoryFunction_Regularization(
      self, batch_size, num_alive_inputs, num_alive_outputs, reg_inputs,
      reg_outputs):
    memory_loss_tensor = resource_function.memory_function(
        self.conv_op, True, num_alive_inputs, num_alive_outputs, reg_inputs,
        reg_outputs, batch_size)
    with self.session() as sess:
      sess.run(tf.global_variables_initializer())
      memory_loss, _ = sess.run(
          [memory_loss_tensor, self.image],
          feed_dict={self.image: np.zeros(self.image_shape)})

    # Expected memory loss = input_feature + weights + output_feature =
    # (batch_size * feature_map_width * feature_map_height * num_alive_inputs
    # + kernel_width * kernel_height * (
    #   num_alive_inputs * reg_outputs + num_alive_outputs * reg_inputs)
    # + batch_size * feature_map_width * feature_map_height * num_alive_outputs)
    # * dtype.size
    expected_memory_loss = (
        batch_size * 11 * 13 * reg_inputs
        + 7 * 5 * (
            num_alive_inputs * reg_outputs + num_alive_outputs * reg_inputs)
        + batch_size * 6 * 7 * reg_outputs) * self.image.dtype.size
    self.assertEqual(expected_memory_loss, memory_loss)

  @parameterized.named_parameters(
      ('_BatchSize1_AliveIn17_AliveOut19', 1, 17, 19, 1, 1),
      ('_BatchSize32_AliveIn4_AliveOut9_ComputeBound', 32, 4, 9, 1000, 2000),
      ('_BatchSize32_AliveIn4_AliveOut9_MemoryBound', 32, 4, 9, 1000, 20))
  def testConvLatencyFunction_Cost(
      self, batch_size, num_alive_inputs, num_alive_outputs, peak_compute,
      memory_bandwidth):
    latency_cost_tensor = resource_function.latency_function(
        self.conv_op, False, num_alive_inputs, num_alive_outputs, 17, 19,
        peak_compute, memory_bandwidth, batch_size)
    with self.session() as sess:
      sess.run(tf.global_variables_initializer())
      latency_cost, _ = sess.run(
          [latency_cost_tensor, self.image],
          feed_dict={self.image: np.zeros(self.image_shape)})

    # Expected latency cost = max(compute_cost, memory_cost)
    expected_compute_cost = (
        2 * batch_size * 6 * 7 * 7 * 5 * num_alive_inputs * num_alive_outputs
        / peak_compute)
    expected_memory_cost = (
        (batch_size * 11 * 13 * num_alive_inputs
         + 7 * 5 * num_alive_inputs * num_alive_outputs
         + batch_size * 6 * 7 * num_alive_outputs)
        * self.image.dtype.size / memory_bandwidth)
    expected_latency_cost = max(expected_compute_cost, expected_memory_cost)
    self.assertNearRelatively(expected_latency_cost, latency_cost)

  @parameterized.named_parameters(
      ('_BatchSize1_AliveIn17_AliveOut19_RegIn17_RegOut19',
       1, 17, 19, 17, 19, 1, 1),
      ('_BatchSize32_AliveIn4_AliveOut9_RegIn3_RegOut7_ComputeBound',
       32, 4, 9, 3, 7, 1000, 2000),
      ('_BatchSize32_AliveIn4_AliveOut9_RegIn3_RegOut7_MemoryBound',
       32, 4, 9, 3, 7, 1000, 20))
  def testConvLatencyFunction_Regularization(
      self, batch_size, num_alive_inputs, num_alive_outputs, reg_inputs,
      reg_outputs, peak_compute, memory_bandwidth):
    latency_loss_tensor = resource_function.latency_function(
        self.conv_op, True, num_alive_inputs, num_alive_outputs, reg_inputs,
        reg_outputs, peak_compute, memory_bandwidth, batch_size)
    with self.session() as sess:
      sess.run(tf.global_variables_initializer())
      latency_loss, _ = sess.run(
          [latency_loss_tensor, self.image],
          feed_dict={self.image: np.zeros(self.image_shape)})

    # Expected latency loss = max(compute_loss, memory_loss)
    expected_compute_cost = (
        2 * batch_size * 6 * 7 * 7 * 5 * num_alive_inputs * num_alive_outputs
        / peak_compute)
    expected_memory_cost = (
        (batch_size * 11 * 13 * num_alive_inputs
         + 7 * 5 * num_alive_inputs * num_alive_outputs
         + batch_size * 6 * 7 * num_alive_outputs)
        * self.image.dtype.size / memory_bandwidth)
    expected_compute_loss = (
        2 * batch_size * 6 * 7 * 7 * 5 * (
            num_alive_inputs * reg_outputs + num_alive_outputs * reg_inputs)
        / peak_compute)
    expected_memory_loss = (
        (batch_size * 11 * 13 * reg_inputs
         + 7 * 5 * (
             num_alive_inputs * reg_outputs + num_alive_outputs * reg_inputs)
         + batch_size * 6 * 7 * reg_outputs)
        * self.image.dtype.size / memory_bandwidth)
    if expected_memory_cost > expected_compute_cost:
      expected_latency_loss = expected_memory_loss
    else:
      expected_latency_loss = expected_compute_loss
    self.assertNearRelatively(expected_latency_loss, latency_loss)

  @parameterized.named_parameters(
      ('_BatchSize1_AliveIn17_AliveOut19', 1, 17, 19),
      ('_BatchSize32_AliveIn4_AliveOut9', 32, 4, 9))
  def testConvModelSizeFunction_Cost(
      self, batch_size, num_alive_inputs, num_alive_outputs):
    model_size_cost = resource_function.model_size_function(
        self.conv_op, False, num_alive_inputs, num_alive_outputs, 17, 19,
        batch_size)

    # Expected model size cost =
    # kernel_width * kernel_height * input_depth * output_depth
    expected_model_size_cost = 7 * 5 * num_alive_inputs * num_alive_outputs
    self.assertEqual(expected_model_size_cost, model_size_cost)

  @parameterized.named_parameters(
      ('_BatchSize1_AliveIn17_AliveOut19_RegIn17_RegOut19',
       1, 17, 19, 17, 19),
      ('_BatchSize32_AliveIn4_AliveOut9_RegIn3_RegOut7',
       32, 4, 9, 3, 7))
  def testConvModelSizeFunction_Regularization(
      self, batch_size, num_alive_inputs, num_alive_outputs, reg_inputs,
      reg_outputs):
    model_size_loss = resource_function.model_size_function(
        self.conv_op, True, num_alive_inputs, num_alive_outputs, reg_inputs,
        reg_outputs, batch_size)

    # Expected model size regularization loss =
    # kernel_width * kernel_height
    # * (num_alive_inputs * reg_outputs + num_alive_outputs * reg_inputs)
    expected_model_size_loss = (
        7 * 5 * (
            num_alive_inputs * reg_outputs + num_alive_outputs * reg_inputs))
    self.assertEqual(expected_model_size_loss, model_size_loss)

  @parameterized.named_parameters(
      ('_BatchSize1_AliveIn17_AliveOut19', 1, 17, 19),
      ('_BatchSize32_AliveIn4_AliveOut9', 32, 4, 9))
  def testConvActivationCountFunction_Cost(
      self, batch_size, num_alive_inputs, num_alive_outputs):
    activation_count_cost = resource_function.activation_count_function(
        self.conv_op, False, num_alive_inputs, num_alive_outputs, 17, 19,
        batch_size)

    # Expected activation count cost = output_depth
    expected_activation_count_cost = num_alive_outputs
    self.assertEqual(expected_activation_count_cost, activation_count_cost)

  @parameterized.named_parameters(
      ('_BatchSize1_AliveIn17_AliveOut19_RegIn17_RegOut19',
       1, 17, 19, 17, 19),
      ('_BatchSize32_AliveIn4_AliveOut9_RegIn3_RegOut7',
       32, 4, 9, 3, 7))
  def testConvActivationCountFunction_Regularization(
      self, batch_size, num_alive_inputs, num_alive_outputs, reg_inputs,
      reg_outputs):
    activation_count_loss = resource_function.activation_count_function(
        self.conv_op, True, num_alive_inputs, num_alive_outputs, reg_inputs,
        reg_outputs, batch_size)

    # Expected model size regularization loss = reg_outputs
    expected_activation_count_loss = reg_outputs
    self.assertEqual(expected_activation_count_loss, activation_count_loss)

  @parameterized.named_parameters(
      ('_BatchSize1_AliveIn17_AliveOut29', 1, 17, 29),
      ('_BatchSize32_AliveIn4_AliveOut9', 32, 4, 9))
  def testConvTransposeFlopFunction_Cost(
      self, batch_size, num_alive_inputs, num_alive_outputs):
    flop_cost_tensor = resource_function.flop_function(
        self.convt_op, False, num_alive_inputs, num_alive_outputs, 17, 29,
        batch_size)
    with self.session() as sess:
      sess.run(tf.global_variables_initializer())
      flop_cost, _ = sess.run(
          [flop_cost_tensor, self.image],
          feed_dict={self.image: np.zeros(self.image_shape)})

    # Expected FLOP cost =
    # 2 * batch_size * feature_map_width * feature_map_height
    # * kernel_width * kernel_height * input_depth * output_depth
    expected_flop_cost = (
        2 * batch_size * 11 * 13 * 7 * 5 * num_alive_inputs * num_alive_outputs)
    self.assertEqual(expected_flop_cost, flop_cost)

  @parameterized.named_parameters(
      ('_BatchSize1_AliveIn17_AliveOut19_RegIn17_RegOut29',
       1, 17, 19, 17, 29),
      ('_BatchSize32_AliveIn4_AliveOut9_RegIn3_RegOut7',
       32, 4, 9, 3, 7))
  def testConvTransposeFlopFunction_Regularization(
      self, batch_size, num_alive_inputs, num_alive_outputs, reg_inputs,
      reg_outputs):
    flop_loss_tensor = resource_function.flop_function(
        self.convt_op, True, num_alive_inputs, num_alive_outputs, reg_inputs,
        reg_outputs, batch_size)
    with self.session() as sess:
      sess.run(tf.global_variables_initializer())
      flop_loss, _ = sess.run(
          [flop_loss_tensor, self.image],
          feed_dict={self.image: np.zeros(self.image_shape)})

    # Expected FLOP regularization loss =
    # 2 * batch_size * feature_map_width * feature_map_height
    # * kernel_width * kernel_height
    # * (num_alive_inputs * reg_outputs + num_alive_outputs * reg_inputs)
    expected_flop_loss = (
        2 * batch_size * 11 * 13 * 7 * 5 * (
            num_alive_inputs * reg_outputs + num_alive_outputs * reg_inputs))
    self.assertEqual(expected_flop_loss, flop_loss)

  @parameterized.named_parameters(
      ('_BatchSize1_AliveIn17_AliveOut29', 1, 17, 29),
      ('_BatchSize32_AliveIn4_AliveOut9', 32, 4, 9))
  def testConvTransposeMemoryFunction_Cost(
      self, batch_size, num_alive_inputs, num_alive_outputs):
    memory_cost_tensor = resource_function.memory_function(
        self.convt_op, False, num_alive_inputs, num_alive_outputs, 17, 29,
        batch_size)
    with self.session() as sess:
      sess.run(tf.global_variables_initializer())
      memory_cost, _ = sess.run(
          [memory_cost_tensor, self.image],
          feed_dict={self.image: np.zeros(self.image_shape)})

    # Expected memory cost = input_feature + weights + output_feature =
    # (batch_size * feature_map_width * feature_map_height * num_alive_inputs
    # + kernel_width * kernel_height * num_alive_inputs * num_alive_outputs
    # + batch_size * feature_map_width * feature_map_height * num_alive_outputs)
    # * dtype.size
    expected_memory_cost = (
        batch_size * 11 * 13 * num_alive_inputs
        + 7 * 5 * num_alive_inputs * num_alive_outputs
        + batch_size * 22 * 26 * num_alive_outputs) * self.image.dtype.size
    self.assertEqual(expected_memory_cost, memory_cost)

  @parameterized.named_parameters(
      ('_BatchSize1_AliveIn17_AliveOut19_RegIn17_RegOut29',
       1, 17, 29, 17, 29),
      ('_BatchSize32_AliveIn4_AliveOut9_RegIn3_RegOut7',
       32, 4, 9, 3, 7))
  def testConvTransposeMemoryFunction_Regularization(
      self, batch_size, num_alive_inputs, num_alive_outputs, reg_inputs,
      reg_outputs):
    memory_loss_tensor = resource_function.memory_function(
        self.convt_op, True, num_alive_inputs, num_alive_outputs, reg_inputs,
        reg_outputs, batch_size)
    with self.session() as sess:
      sess.run(tf.global_variables_initializer())
      memory_loss, _ = sess.run(
          [memory_loss_tensor, self.image],
          feed_dict={self.image: np.zeros(self.image_shape)})

    # Expected memory loss = input_feature + weights + output_feature =
    # (batch_size * feature_map_width * feature_map_height * num_alive_inputs
    # + kernel_width * kernel_height * (
    #   num_alive_inputs * reg_outputs + num_alive_outputs * reg_inputs)
    # + batch_size * feature_map_width * feature_map_height * num_alive_outputs)
    # * dtype.size
    expected_memory_loss = (
        batch_size * 11 * 13 * reg_inputs
        + 7 * 5 * (
            num_alive_inputs * reg_outputs + num_alive_outputs * reg_inputs)
        + batch_size * 22 * 26 * reg_outputs) * self.image.dtype.size
    self.assertEqual(expected_memory_loss, memory_loss)

  @parameterized.named_parameters(
      ('_BatchSize1_AliveIn17_AliveOut29', 1, 17, 29, 1, 1),
      ('_BatchSize32_AliveIn4_AliveOut9_ComputeBound', 32, 4, 9, 1000, 2000),
      ('_BatchSize32_AliveIn4_AliveOut9_MemoryBound', 32, 4, 9, 1000, 20))
  def testConvTransposeLatencyFunction_Cost(
      self, batch_size, num_alive_inputs, num_alive_outputs, peak_compute,
      memory_bandwidth):
    latency_cost_tensor = resource_function.latency_function(
        self.convt_op, False, num_alive_inputs, num_alive_outputs, 17, 29,
        peak_compute, memory_bandwidth, batch_size)
    with self.session() as sess:
      sess.run(tf.global_variables_initializer())
      latency_cost, _ = sess.run(
          [latency_cost_tensor, self.image],
          feed_dict={self.image: np.zeros(self.image_shape)})

    # Expected latency cost = max(compute_cost, memory_cost)
    expected_compute_cost = (
        2 * batch_size * 11 * 13 * 7 * 5 * num_alive_inputs * num_alive_outputs
        / peak_compute)
    expected_memory_cost = (
        (batch_size * 11 * 13 * num_alive_inputs
         + 7 * 5 * num_alive_inputs * num_alive_outputs
         + batch_size * 22 * 26 * num_alive_outputs)
        * self.image.dtype.size / memory_bandwidth)
    expected_latency_cost = max(expected_compute_cost, expected_memory_cost)
    self.assertNearRelatively(expected_latency_cost, latency_cost)

  @parameterized.named_parameters(
      ('_BatchSize1_AliveIn17_AliveOut19_RegIn17_RegOut29',
       1, 17, 19, 17, 29, 1, 1),
      ('_BatchSize32_AliveIn4_AliveOut9_RegIn3_RegOut7_ComputeBound',
       32, 4, 9, 3, 7, 1000, 2000),
      ('_BatchSize32_AliveIn4_AliveOut9_RegIn3_RegOut7_MemoryBound',
       32, 4, 9, 3, 7, 1000, 20))
  def testConvTransposeLatencyFunction_Regularization(
      self, batch_size, num_alive_inputs, num_alive_outputs, reg_inputs,
      reg_outputs, peak_compute, memory_bandwidth):
    latency_loss_tensor = resource_function.latency_function(
        self.convt_op, True, num_alive_inputs, num_alive_outputs, reg_inputs,
        reg_outputs, peak_compute, memory_bandwidth, batch_size)
    with self.session() as sess:
      sess.run(tf.global_variables_initializer())
      latency_loss, _ = sess.run(
          [latency_loss_tensor, self.image],
          feed_dict={self.image: np.zeros(self.image_shape)})

    # Expected latency loss = max(compute_loss, memory_loss)
    expected_compute_cost = (
        2 * batch_size * 11 * 13 * 7 * 5 * num_alive_inputs * num_alive_outputs
        / peak_compute)
    expected_memory_cost = (
        (batch_size * 11 * 13 * num_alive_inputs
         + 7 * 5 * num_alive_inputs * num_alive_outputs
         + batch_size * 22 * 26 * num_alive_outputs)
        * self.image.dtype.size / memory_bandwidth)
    expected_compute_loss = (
        2 * batch_size * 11 * 13 * 7 * 5 * (
            num_alive_inputs * reg_outputs + num_alive_outputs * reg_inputs)
        / peak_compute)
    expected_memory_loss = (
        (batch_size * 11 * 13 * reg_inputs
         + 7 * 5 * (
             num_alive_inputs * reg_outputs + num_alive_outputs * reg_inputs)
         + batch_size * 22 * 26 * reg_outputs)
        * self.image.dtype.size / memory_bandwidth)
    if expected_memory_cost > expected_compute_cost:
      expected_latency_loss = expected_memory_loss
    else:
      expected_latency_loss = expected_compute_loss
    self.assertNearRelatively(expected_latency_loss, latency_loss)

  @parameterized.named_parameters(
      ('_BatchSize1_AliveIn17_AliveOut29', 1, 17, 29),
      ('_BatchSize32_AliveIn4_AliveOut9', 32, 4, 9))
  def testConvTransposeModelSizeFunction_Cost(
      self, batch_size, num_alive_inputs, num_alive_outputs):
    model_size_cost = resource_function.model_size_function(
        self.convt_op, False, num_alive_inputs, num_alive_outputs, 17, 29,
        batch_size)

    # Expected model size cost =
    # kernel_width * kernel_height * input_depth * output_depth
    expected_model_size_cost = 7 * 5 * num_alive_inputs * num_alive_outputs
    self.assertEqual(expected_model_size_cost, model_size_cost)

  @parameterized.named_parameters(
      ('_BatchSize1_AliveIn17_AliveOut19_RegIn17_RegOut29',
       1, 17, 19, 17, 29),
      ('_BatchSize32_AliveIn4_AliveOut9_RegIn3_RegOut7',
       32, 4, 9, 3, 7))
  def testConvTransposeModelSizeFunction_Regularization(
      self, batch_size, num_alive_inputs, num_alive_outputs, reg_inputs,
      reg_outputs):
    model_size_loss = resource_function.model_size_function(
        self.convt_op, True, num_alive_inputs, num_alive_outputs, reg_inputs,
        reg_outputs, batch_size)

    # Expected FLOP regularization loss =
    # kernel_width * kernel_height
    # * (num_alive_inputs * reg_outputs + num_alive_outputs * reg_inputs)
    expected_model_size_loss = (
        7 * 5 * (
            num_alive_inputs * reg_outputs + num_alive_outputs * reg_inputs))
    self.assertEqual(expected_model_size_loss, model_size_loss)

  @parameterized.named_parameters(
      ('_BatchSize1_AliveIn17_AliveOut29', 1, 17, 29),
      ('_BatchSize32_AliveIn4_AliveOut9', 32, 4, 9))
  def testConvTransposeActivationCountFunction_Cost(
      self, batch_size, num_alive_inputs, num_alive_outputs):
    activation_count_cost = resource_function.activation_count_function(
        self.convt_op, False, num_alive_inputs, num_alive_outputs, 17, 29,
        batch_size)

    # Expected model size cost = output_depth
    expected_activation_count_cost = num_alive_outputs
    self.assertEqual(expected_activation_count_cost, activation_count_cost)

  @parameterized.named_parameters(
      ('_BatchSize1_AliveIn17_AliveOut19_RegIn17_RegOut29',
       1, 17, 19, 17, 29),
      ('_BatchSize32_AliveIn4_AliveOut9_RegIn3_RegOut7',
       32, 4, 9, 3, 7))
  def testConvTransposeActivationCountFunction_Regularization(
      self, batch_size, num_alive_inputs, num_alive_outputs, reg_inputs,
      reg_outputs):
    activation_count_loss = resource_function.activation_count_function(
        self.convt_op, True, num_alive_inputs, num_alive_outputs, reg_inputs,
        reg_outputs, batch_size)

    # Expected FLOP regularization loss = reg_outputs
    expected_activation_count_loss = reg_outputs
    self.assertEqual(expected_activation_count_loss, activation_count_loss)

  @parameterized.named_parameters(
      ('_BatchSize1_AliveIn17_AliveOut19', 1, 17, 19),
      ('_BatchSize32_AliveIn4_AliveOut9', 32, 4, 9))
  def testMatMulFlopFunction_Cost(
      self, batch_size, num_alive_inputs, num_alive_outputs):
    flop_cost = resource_function.flop_function(
        self.matmul_op, False, num_alive_inputs, num_alive_outputs, 17, 19,
        batch_size)

    # Expected FLOP cost =
    # 2 * batch_size * input_depth * output_depth
    expected_flop_cost = 2 * batch_size * num_alive_inputs * num_alive_outputs
    self.assertEqual(expected_flop_cost, flop_cost)

  @parameterized.named_parameters(
      ('_BatchSize1_AliveIn17_AliveOut19_RegIn17_RegOut19',
       1, 17, 19, 17, 19),
      ('_BatchSize32_AliveIn4_AliveOut9_RegIn3_RegOut7',
       32, 4, 9, 3, 7))
  def testMatMulFlopFunction_Regularization(
      self, batch_size, num_alive_inputs, num_alive_outputs, reg_inputs,
      reg_outputs):
    flop_loss = resource_function.flop_function(
        self.matmul_op, True, num_alive_inputs, num_alive_outputs, reg_inputs,
        reg_outputs, batch_size)

    # Expected FLOP regularization loss =
    # 2 * batch_size
    # * (num_alive_inputs * reg_outputs + num_alive_outputs * reg_inputs)
    expected_flop_loss = (
        2 * batch_size * (
            num_alive_inputs * reg_outputs + num_alive_outputs * reg_inputs))
    self.assertEqual(expected_flop_loss, flop_loss)

  @parameterized.named_parameters(
      ('_BatchSize1_AliveIn17_AliveOut19', 1, 17, 19),
      ('_BatchSize32_AliveIn4_AliveOut9', 32, 4, 9))
  def testMatMulMemoryFunction_Cost(
      self, batch_size, num_alive_inputs, num_alive_outputs):
    memory_cost_tensor = resource_function.memory_function(
        self.matmul_op, False, num_alive_inputs, num_alive_outputs, 17, 19,
        batch_size)
    with self.session() as sess:
      sess.run(tf.global_variables_initializer())
      memory_cost, _ = sess.run(
          [memory_cost_tensor, self.image],
          feed_dict={self.image: np.zeros(self.image_shape)})

    # Expected memory cost = input_feature + weights + output_feature =
    # (batch_size * num_alive_inputs
    # + num_alive_inputs * num_alive_outputs
    # + batch_size * num_alive_outputs) * dtype.size
    expected_memory_cost = (
        batch_size * num_alive_inputs
        + num_alive_inputs * num_alive_outputs
        + batch_size * num_alive_outputs) * self.image.dtype.size
    self.assertEqual(expected_memory_cost, memory_cost)

  @parameterized.named_parameters(
      ('_BatchSize1_AliveIn17_AliveOut19_RegIn17_RegOut19',
       1, 17, 19, 17, 19),
      ('_BatchSize32_AliveIn4_AliveOut9_RegIn3_RegOut7',
       32, 4, 9, 3, 7))
  def testMatMulMemoryFunction_Regularization(
      self, batch_size, num_alive_inputs, num_alive_outputs, reg_inputs,
      reg_outputs):
    memory_loss_tensor = resource_function.memory_function(
        self.matmul_op, True, num_alive_inputs, num_alive_outputs, reg_inputs,
        reg_outputs, batch_size)
    with self.session() as sess:
      sess.run(tf.global_variables_initializer())
      memory_loss, _ = sess.run(
          [memory_loss_tensor, self.image],
          feed_dict={self.image: np.zeros(self.image_shape)})

    # Expected memory loss = input_feature + weights + output_feature =
    # (batch_size * feature_map_width * feature_map_height * num_alive_inputs
    # + kernel_width * kernel_height * (
    #   num_alive_inputs * reg_outputs + num_alive_outputs * reg_inputs)
    # + batch_size * feature_map_width * feature_map_height * num_alive_outputs)
    # * dtype.size
    expected_memory_loss = (
        batch_size * reg_inputs
        + (num_alive_inputs * reg_outputs + num_alive_outputs * reg_inputs)
        + batch_size * reg_outputs) * self.image.dtype.size
    self.assertEqual(expected_memory_loss, memory_loss)

  @parameterized.named_parameters(
      ('_BatchSize1_AliveIn17_AliveOut19', 1, 17, 19, 1, 1),
      ('_BatchSize32_AliveIn4_AliveOut9_ComputeBound', 32, 4, 9, 1000, 2000),
      ('_BatchSize32_AliveIn4_AliveOut9_MemoryBound', 32, 4, 9, 1000, 20))
  def testMatMulLatencyFunction_Cost(
      self, batch_size, num_alive_inputs, num_alive_outputs, peak_compute,
      memory_bandwidth):
    latency_cost_tensor = resource_function.latency_function(
        self.matmul_op, False, num_alive_inputs, num_alive_outputs, 17, 19,
        peak_compute, memory_bandwidth, batch_size)
    with self.session() as sess:
      sess.run(tf.global_variables_initializer())
      latency_cost, _ = sess.run(
          [latency_cost_tensor, self.image],
          feed_dict={self.image: np.zeros(self.image_shape)})

    # Expected latency cost = max(compute_cost, memory_cost)
    expected_compute_cost = (
        2 * batch_size * num_alive_inputs * num_alive_outputs / peak_compute)
    expected_memory_cost = (
        (batch_size * num_alive_inputs
         + num_alive_inputs * num_alive_outputs
         + batch_size * num_alive_outputs)
        * self.image.dtype.size / memory_bandwidth)
    expected_latency_cost = max(expected_compute_cost, expected_memory_cost)
    self.assertNearRelatively(expected_latency_cost, latency_cost)

  @parameterized.named_parameters(
      ('_BatchSize1_AliveIn17_AliveOut19_RegIn17_RegOut19',
       1, 17, 19, 17, 19, 1, 1),
      ('_BatchSize32_AliveIn4_AliveOut9_RegIn3_RegOut7_ComputeBound',
       32, 4, 9, 3, 7, 1000, 2000),
      ('_BatchSize32_AliveIn4_AliveOut9_RegIn3_RegOut7_MemoryBound',
       32, 4, 9, 3, 7, 1000, 20))
  def testMatMulLatencyFunction_Regularization(
      self, batch_size, num_alive_inputs, num_alive_outputs, reg_inputs,
      reg_outputs, peak_compute, memory_bandwidth):
    latency_loss_tensor = resource_function.latency_function(
        self.matmul_op, True, num_alive_inputs, num_alive_outputs, reg_inputs,
        reg_outputs, peak_compute, memory_bandwidth, batch_size)
    with self.session() as sess:
      sess.run(tf.global_variables_initializer())
      latency_loss, _ = sess.run(
          [latency_loss_tensor, self.image],
          feed_dict={self.image: np.zeros(self.image_shape)})

    # Expected latency loss = max(compute_loss, memory_loss)
    expected_compute_cost = (
        2 * batch_size * num_alive_inputs * num_alive_outputs / peak_compute)
    expected_memory_cost = (
        (batch_size * num_alive_inputs
         + num_alive_inputs * num_alive_outputs
         + batch_size * num_alive_outputs)
        * self.image.dtype.size / memory_bandwidth)
    expected_compute_loss = (
        2 * batch_size * (
            num_alive_inputs * reg_outputs + num_alive_outputs * reg_inputs)
        / peak_compute)
    expected_memory_loss = (
        (batch_size * reg_inputs
         + (num_alive_inputs * reg_outputs + num_alive_outputs * reg_inputs)
         + batch_size * reg_outputs)
        * self.image.dtype.size / memory_bandwidth)
    if expected_memory_cost > expected_compute_cost:
      expected_latency_loss = expected_memory_loss
    else:
      expected_latency_loss = expected_compute_loss
    self.assertNearRelatively(expected_latency_loss, latency_loss)

  @parameterized.named_parameters(
      ('_BatchSize1_AliveIn17_AliveOut19', 1, 17, 19),
      ('_BatchSize32_AliveIn4_AliveOut9', 32, 4, 9))
  def testMatMulModelSizeFunction_Cost(
      self, batch_size, num_alive_inputs, num_alive_outputs):
    model_size_cost = resource_function.model_size_function(
        self.matmul_op, False, num_alive_inputs, num_alive_outputs, 17, 19,
        batch_size)

    # Expected model size cost = input_depth * output_depth
    expected_model_size_cost = num_alive_inputs * num_alive_outputs
    self.assertEqual(expected_model_size_cost, model_size_cost)

  @parameterized.named_parameters(
      ('_BatchSize1_AliveIn17_AliveOut19_RegIn17_RegOut19',
       1, 17, 19, 17, 19),
      ('_BatchSize32_AliveIn4_AliveOut9_RegIn3_RegOut7',
       32, 4, 9, 3, 7))
  def testMatMulModelSizeFlopFunction_Regularization(
      self, batch_size, num_alive_inputs, num_alive_outputs, reg_inputs,
      reg_outputs):
    model_size_loss = resource_function.model_size_function(
        self.matmul_op, True, num_alive_inputs, num_alive_outputs, reg_inputs,
        reg_outputs, batch_size)

    # Expected model size regularization loss =
    # num_alive_inputs * reg_outputs + num_alive_outputs * reg_inputs
    expected_model_size_loss = (
        num_alive_inputs * reg_outputs + num_alive_outputs * reg_inputs)
    self.assertEqual(expected_model_size_loss, model_size_loss)

  @parameterized.named_parameters(
      ('_BatchSize1_AliveIn17_AliveOut19', 1, 17, 19),
      ('_BatchSize32_AliveIn4_AliveOut9', 32, 4, 9))
  def testMatMulActivationCountFunction_Cost(
      self, batch_size, num_alive_inputs, num_alive_outputs):
    activation_count_cost = resource_function.activation_count_function(
        self.matmul_op, False, num_alive_inputs, num_alive_outputs, 17, 19,
        batch_size)

    # Expected model size cost = output_depth
    expected_activation_count_cost = num_alive_outputs
    self.assertEqual(expected_activation_count_cost, activation_count_cost)

  @parameterized.named_parameters(
      ('_BatchSize1_AliveIn17_AliveOut19_RegIn17_RegOut19',
       1, 17, 19, 17, 19),
      ('_BatchSize32_AliveIn4_AliveOut9_RegIn3_RegOut7',
       32, 4, 9, 3, 7))
  def testMatMulActivationCountFlopFunction_Regularization(
      self, batch_size, num_alive_inputs, num_alive_outputs, reg_inputs,
      reg_outputs):
    activation_count_loss = resource_function.activation_count_function(
        self.matmul_op, True, num_alive_inputs, num_alive_outputs, reg_inputs,
        reg_outputs, batch_size)

    # Expected model size regularization loss = reg_outputs
    expected_activation_count_loss = reg_outputs
    self.assertEqual(expected_activation_count_loss, activation_count_loss)

  @parameterized.named_parameters(
      ('_BatchSize1_AliveIn10_AliveOut10', 1, 10, 10),
      ('_BatchSize32_AliveIn4_AliveOut9', 32, 4, 9))
  def testDepthwiseConvFlopFunction_Cost(
      self, batch_size, num_alive_inputs, num_alive_outputs):
    flop_cost_tensor = resource_function.flop_function(
        self.dw_op, False, num_alive_inputs, num_alive_outputs, 10, 10,
        batch_size)
    with self.session() as sess:
      sess.run(tf.global_variables_initializer())
      flop_cost, _ = sess.run(
          [flop_cost_tensor, self.image],
          feed_dict={self.image: np.zeros(self.image_shape)})

    # Expected FLOP cost =
    # 2 * batch_size * feature_map_width * feature_map_height
    # * kernel_width * kernel_height * output_depth
    expected_flop_cost = (
        2 * batch_size * 6 * 7 * 3 * 2 * num_alive_outputs)
    self.assertEqual(expected_flop_cost, flop_cost)

  @parameterized.named_parameters(
      ('_BatchSize1_AliveIn10_AliveOut10_RegIn10_RegOut10',
       1, 10, 10, 10, 10),
      ('_BatchSize32_AliveIn4_AliveOut9_RegIn3_RegOut7',
       32, 4, 9, 3, 7))
  def testDepthwiseConvFlopFunction_Regularization(
      self, batch_size, num_alive_inputs, num_alive_outputs, reg_inputs,
      reg_outputs):
    flop_loss_tensor = resource_function.flop_function(
        self.dw_op, True, num_alive_inputs, num_alive_outputs, reg_inputs,
        reg_outputs, batch_size)
    with self.session() as sess:
      sess.run(tf.global_variables_initializer())
      flop_loss, _ = sess.run(
          [flop_loss_tensor, self.image],
          feed_dict={self.image: np.zeros(self.image_shape)})

    # Expected FLOP regularization loss =
    # 2 * batch_size * feature_map_width * feature_map_height
    # * kernel_width * kernel_height * (reg_inputs + reg_outputs)
    expected_flop_loss = (
        2 * batch_size * 6 * 7 * 3 * 2 * (reg_inputs + reg_outputs))
    self.assertEqual(expected_flop_loss, flop_loss)

  @parameterized.named_parameters(
      ('_BatchSize1_AliveIn10_AliveOut10', 1, 10, 10),
      ('_BatchSize32_AliveIn4_AliveOut9', 32, 4, 9))
  def testDepthwiseConvMemoryFunction_Cost(
      self, batch_size, num_alive_inputs, num_alive_outputs):
    memory_cost_tensor = resource_function.memory_function(
        self.conv_op, False, num_alive_inputs, num_alive_outputs, 10, 10,
        batch_size)
    with self.session() as sess:
      sess.run(tf.global_variables_initializer())
      memory_cost, _ = sess.run(
          [memory_cost_tensor, self.image],
          feed_dict={self.image: np.zeros(self.image_shape)})

    # Expected memory cost = input_feature + weights + output_feature =
    # (batch_size * feature_map_width * feature_map_height * num_alive_inputs
    # + kernel_width * kernel_height * num_alive_inputs * num_alive_outputs
    # + batch_size * feature_map_width * feature_map_height * num_alive_outputs)
    # * dtype.size
    expected_memory_cost = (
        batch_size * 11 * 13 * num_alive_inputs
        + 7 * 5 * num_alive_inputs * num_alive_outputs
        + batch_size * 6 * 7 * num_alive_outputs) * self.image.dtype.size
    self.assertEqual(expected_memory_cost, memory_cost)

  @parameterized.named_parameters(
      ('_BatchSize1_AliveIn10_AliveOut10_RegIn10_RegOut10',
       1, 10, 10, 10, 10),
      ('_BatchSize32_AliveIn4_AliveOut9_RegIn3_RegOut7',
       32, 4, 9, 3, 7))
  def testDepthwiseConvMemoryFunction_Regularization(
      self, batch_size, num_alive_inputs, num_alive_outputs, reg_inputs,
      reg_outputs):
    memory_loss_tensor = resource_function.memory_function(
        self.conv_op, True, num_alive_inputs, num_alive_outputs, reg_inputs,
        reg_outputs, batch_size)
    with self.session() as sess:
      sess.run(tf.global_variables_initializer())
      memory_loss, _ = sess.run(
          [memory_loss_tensor, self.image],
          feed_dict={self.image: np.zeros(self.image_shape)})

    # Expected memory loss = input_feature + weights + output_feature =
    # (batch_size * feature_map_width * feature_map_height * num_alive_inputs
    # + kernel_width * kernel_height * (
    #   num_alive_inputs * reg_outputs + num_alive_outputs * reg_inputs)
    # + batch_size * feature_map_width * feature_map_height * num_alive_outputs)
    # * dtype.size
    expected_memory_loss = (
        batch_size * 11 * 13 * reg_inputs
        + 7 * 5 * (
            num_alive_inputs * reg_outputs + num_alive_outputs * reg_inputs)
        + batch_size * 6 * 7 * reg_outputs) * self.image.dtype.size
    self.assertEqual(expected_memory_loss, memory_loss)

  @parameterized.named_parameters(
      ('_BatchSize1_AliveIn10_AliveOut10', 1, 10, 10, 1, 1),
      ('_BatchSize32_AliveIn4_AliveOut9_ComputeBound', 32, 4, 9, 1000, 2000),
      ('_BatchSize32_AliveIn4_AliveOut9_MemoryBound', 32, 4, 9, 1000, 20))
  def testDepthwiseConvLatencyFunction_Cost(
      self, batch_size, num_alive_inputs, num_alive_outputs, peak_compute,
      memory_bandwidth):
    latency_cost_tensor = resource_function.latency_function(
        self.conv_op, False, num_alive_inputs, num_alive_outputs, 10, 10,
        peak_compute, memory_bandwidth, batch_size)
    with self.session() as sess:
      sess.run(tf.global_variables_initializer())
      latency_cost, _ = sess.run(
          [latency_cost_tensor, self.image],
          feed_dict={self.image: np.zeros(self.image_shape)})

    # Expected latency cost = max(compute_cost, memory_cost)
    expected_compute_cost = (
        2 * batch_size * 6 * 7 * 7 * 5 * num_alive_inputs * num_alive_outputs
        / peak_compute)
    expected_memory_cost = (
        (batch_size * 11 * 13 * num_alive_inputs
         + 7 * 5 * num_alive_inputs * num_alive_outputs
         + batch_size * 6 * 7 * num_alive_outputs)
        * self.image.dtype.size / memory_bandwidth)
    expected_latency_cost = max(expected_compute_cost, expected_memory_cost)
    self.assertNearRelatively(expected_latency_cost, latency_cost)

  @parameterized.named_parameters(
      ('_BatchSize1_AliveIn10_AliveOut10_RegIn10_RegOut10',
       1, 10, 10, 10, 10, 1, 1),
      ('_BatchSize32_AliveIn4_AliveOut9_RegIn3_RegOut7_ComputeBound',
       32, 4, 9, 3, 7, 1000, 2000),
      ('_BatchSize32_AliveIn4_AliveOut9_RegIn3_RegOut7_MemoryBound',
       32, 4, 9, 3, 7, 1000, 20))
  def testDepthwiseConvLatencyFunction_Regularization(
      self, batch_size, num_alive_inputs, num_alive_outputs, reg_inputs,
      reg_outputs, peak_compute, memory_bandwidth):
    latency_loss_tensor = resource_function.latency_function(
        self.conv_op, True, num_alive_inputs, num_alive_outputs, reg_inputs,
        reg_outputs, peak_compute, memory_bandwidth, batch_size)
    with self.session() as sess:
      sess.run(tf.global_variables_initializer())
      latency_loss, _ = sess.run(
          [latency_loss_tensor, self.image],
          feed_dict={self.image: np.zeros(self.image_shape)})

    # Expected latency loss = max(compute_loss, memory_loss)
    expected_compute_cost = (
        2 * batch_size * 6 * 7 * 7 * 5 * num_alive_inputs * num_alive_outputs
        / peak_compute)
    expected_memory_cost = (
        (batch_size * 11 * 13 * num_alive_inputs
         + 7 * 5 * num_alive_inputs * num_alive_outputs
         + batch_size * 6 * 7 * num_alive_outputs)
        * self.image.dtype.size / memory_bandwidth)
    expected_compute_loss = (
        2 * batch_size * 6 * 7 * 7 * 5 * (
            num_alive_inputs * reg_outputs + num_alive_outputs * reg_inputs)
        / peak_compute)
    expected_memory_loss = (
        (batch_size * 11 * 13 * reg_inputs
         + 7 * 5 * (
             num_alive_inputs * reg_outputs + num_alive_outputs * reg_inputs)
         + batch_size * 6 * 7 * reg_outputs)
        * self.image.dtype.size / memory_bandwidth)
    if expected_memory_cost > expected_compute_cost:
      expected_latency_loss = expected_memory_loss
    else:
      expected_latency_loss = expected_compute_loss
    self.assertNearRelatively(expected_latency_loss, latency_loss)

  @parameterized.named_parameters(
      ('_BatchSize1_AliveIn10_AliveOut10', 1, 10, 10),
      ('_BatchSize32_AliveIn4_AliveOut9', 32, 4, 9))
  def testDepthwiseConvModelSizeFunction_Cost(
      self, batch_size, num_alive_inputs, num_alive_outputs):
    model_size_cost = resource_function.model_size_function(
        self.dw_op, False, num_alive_inputs, num_alive_outputs, 10, 10,
        batch_size)

    # Expected model size cost =
    # kernel_width * kernel_height * output_depth
    expected_model_size_cost = 3 * 2 * num_alive_outputs
    self.assertEqual(expected_model_size_cost, model_size_cost)

  @parameterized.named_parameters(
      ('_BatchSize1_AliveIn10_AliveOut10_RegIn10_RegOut10',
       1, 10, 10, 10, 10),
      ('_BatchSize32_AliveIn4_AliveOut9_RegIn3_RegOut7',
       32, 4, 9, 3, 7))
  def testDepthwiseConvModelSizeFlopFunction_Regularization(
      self, batch_size, num_alive_inputs, num_alive_outputs, reg_inputs,
      reg_outputs):
    model_size_loss = resource_function.model_size_function(
        self.dw_op, True, num_alive_inputs, num_alive_outputs, reg_inputs,
        reg_outputs, batch_size)

    # Expected model size regularization loss =
    # kernel_width * kernel_height * (reg_inputs + reg_outputs)
    expected_model_size_loss = 3 * 2 * (reg_inputs + reg_outputs)
    self.assertEqual(expected_model_size_loss, model_size_loss)

  @parameterized.named_parameters(
      ('_BatchSize1_AliveIn10_AliveOut10', 1, 10, 10),
      ('_BatchSize32_AliveIn4_AliveOut9', 32, 4, 9))
  def testDepthwiseConvActivationCountFunction_Cost(
      self, batch_size, num_alive_inputs, num_alive_outputs):
    activation_count_cost = resource_function.activation_count_function(
        self.dw_op, False, num_alive_inputs, num_alive_outputs, 10, 10,
        batch_size)

    # Expected model size cost = output_depth
    expected_activation_count_cost = num_alive_outputs
    self.assertEqual(expected_activation_count_cost, activation_count_cost)

  @parameterized.named_parameters(
      ('_BatchSize1_AliveIn10_AliveOut10_RegIn10_RegOut10',
       1, 10, 10, 10, 10),
      ('_BatchSize32_AliveIn4_AliveOut9_RegIn3_RegOut7',
       32, 4, 9, 3, 7))
  def testDepthwiseConvActivationCountFlopFunction_Regularization(
      self, batch_size, num_alive_inputs, num_alive_outputs, reg_inputs,
      reg_outputs):
    activation_count_loss = resource_function.activation_count_function(
        self.dw_op, True, num_alive_inputs, num_alive_outputs, reg_inputs,
        reg_outputs, batch_size)

    # Expected model size regularization loss = reg_outputs
    expected_activation_count_loss = reg_outputs
    self.assertEqual(expected_activation_count_loss, activation_count_loss)

  @parameterized.named_parameters(
      ('_BatchSize1_AliveIn19_AliveIn11', 1, 19, 11),
      ('_BatchSize32_AliveIn15_AliveIn7', 32, 15, 7))
  def testConcatFlopFunction_Cost(
      self, batch_size, num_alive_inputs3, num_alive_inputs4):
    conv3 = layers.conv2d(
        self.image, 19, [7, 5], stride=2, padding='SAME', scope='conv3')
    conv4 = layers.conv2d(
        self.image, 11, [3, 7], stride=2, padding='SAME', scope='conv4')
    concat = tf.concat([conv3, conv4], axis=3)
    flop_cost = resource_function.flop_function(
        concat.op, False, num_alive_inputs3 + num_alive_inputs4,
        num_alive_inputs3 + num_alive_inputs4, 19, 11, batch_size)

    expected_flop_cost = 0
    self.assertEqual(expected_flop_cost, flop_cost)

  @parameterized.named_parameters(
      ('_BatchSize1_AliveIn17_AliveIn19_RegIn17_RegIn19',
       1, 19, 11, 19, 11),
      ('_BatchSize32_AliveIn4_AliveIn9_RegIn3_RegIn7',
       32, 15, 7, 12, 6))
  def testConcatFlopFunction_Regularization(
      self, batch_size, num_alive_inputs3, num_alive_inputs4, reg_inputs3,
      reg_inputs4):
    conv3 = layers.conv2d(
        self.image, 19, [7, 5], stride=2, padding='SAME', scope='conv3')
    conv4 = layers.conv2d(
        self.image, 11, [3, 7], stride=2, padding='SAME', scope='conv4')
    concat = tf.concat([conv3, conv4], axis=3)
    flop_loss = resource_function.flop_function(
        concat.op, True, num_alive_inputs3 + num_alive_inputs4,
        num_alive_inputs3 + num_alive_inputs4, reg_inputs3 + reg_inputs4,
        reg_inputs3 + reg_inputs4, batch_size)

    expected_flop_loss = 0
    self.assertEqual(expected_flop_loss, flop_loss)

  @parameterized.named_parameters(
      ('_BatchSize1_AliveIn19_AliveIn11', 1, 19, 11),
      ('_BatchSize32_AliveIn15_AliveIn7', 32, 15, 7))
  def testConcatMemoryFunction_Cost(
      self, batch_size, num_alive_inputs3, num_alive_inputs4):
    conv3 = layers.conv2d(
        self.image, 19, [7, 5], stride=2, padding='SAME', scope='conv3')
    conv4 = layers.conv2d(
        self.image, 11, [3, 7], stride=2, padding='SAME', scope='conv4')
    concat = tf.concat([conv3, conv4], axis=3)
    memory_cost_tensor = resource_function.memory_function(
        concat.op, False, num_alive_inputs3 + num_alive_inputs4,
        num_alive_inputs3 + num_alive_inputs4, 19, 11, batch_size)
    with self.session() as sess:
      sess.run(tf.global_variables_initializer())
      memory_cost, _ = sess.run(
          [memory_cost_tensor, self.image],
          feed_dict={self.image: np.zeros(self.image_shape)})

    # Expected memory cost = input_feature3 + input_feature4 + output_feature =
    # (batch_size * feature_map_width * feature_map_height * num_alive_inputs3
    # + batch_size * feature_map_width * feature_map_height * num_alive_inputs4
    # + batch_size * feature_map_width * feature_map_height * num_alive_outputs)
    # * dtype.size
    expected_memory_cost = (
        (batch_size * 6 * 7 * num_alive_inputs3
         + batch_size * 6 * 7 * num_alive_inputs4
         + batch_size * 6 * 7 * (num_alive_inputs3 + num_alive_inputs4))
        * self.image.dtype.size)
    self.assertEqual(expected_memory_cost, memory_cost)

  @parameterized.named_parameters(
      ('_BatchSize1_AliveIn19_AliveIn11_RegIn19_RegIn11', 1, 19, 11, 19, 11),
      ('_BatchSize32_AliveIn15_AliveIn7_RegIn12_RegIn6', 32, 15, 7, 12, 6))
  def testConcatMemoryFunction_Regularization(
      self, batch_size, num_alive_inputs3, num_alive_inputs4, reg_inputs3,
      reg_inputs4):
    conv3 = layers.conv2d(
        self.image, 19, [7, 5], stride=2, padding='SAME', scope='conv3')
    conv4 = layers.conv2d(
        self.image, 11, [3, 7], stride=2, padding='SAME', scope='conv4')
    concat = tf.concat([conv3, conv4], axis=3)
    memory_loss_tensor = resource_function.memory_function(
        concat.op, True, num_alive_inputs3 + num_alive_inputs4,
        num_alive_inputs3 + num_alive_inputs4, reg_inputs3 + reg_inputs4,
        reg_inputs3 + reg_inputs4, batch_size)
    with self.session() as sess:
      sess.run(tf.global_variables_initializer())
      memory_loss, _ = sess.run(
          [memory_loss_tensor, self.image],
          feed_dict={self.image: np.zeros(self.image_shape)})

    # Expected memory cost = input_feature3 + input_feature4 + output_feature =
    # (batch_size * feature_map_width * feature_map_height * reg_inputs3
    # + batch_size * feature_map_width * feature_map_height * reg_inputs4
    # + batch_size * feature_map_width * feature_map_height *
    # (reg_inputs3 + reg_inputs4)) * dtype.size
    expected_memory_loss = (
        (batch_size * 6 * 7 * reg_inputs3
         + batch_size * 6 * 7 * reg_inputs4
         + batch_size * 6 * 7 * (reg_inputs3 + reg_inputs4))
        * self.image.dtype.size)
    self.assertEqual(expected_memory_loss, memory_loss)

  @parameterized.named_parameters(
      ('_BatchSize1_AliveIn19_AliveIn11', 1, 19, 11),
      ('_BatchSize32_AliveIn15_AliveIn7', 32, 15, 7))
  def testConcatModelSizeFunction_Cost(
      self, batch_size, num_alive_inputs3, num_alive_inputs4):
    conv3 = layers.conv2d(
        self.image, 19, [7, 5], stride=2, padding='SAME', scope='conv3')
    conv4 = layers.conv2d(
        self.image, 11, [3, 7], stride=2, padding='SAME', scope='conv4')
    concat = tf.concat([conv3, conv4], axis=3)
    model_size_cost = resource_function.model_size_function(
        concat.op, False, num_alive_inputs3 + num_alive_inputs4,
        num_alive_inputs3 + num_alive_inputs4, 19, 11, batch_size)

    expected_model_size_cost = 0
    self.assertEqual(expected_model_size_cost, model_size_cost)

  @parameterized.named_parameters(
      ('_BatchSize1_AliveIn17_AliveIn19_RegIn17_RegIn19',
       1, 19, 11, 19, 11),
      ('_BatchSize32_AliveIn4_AliveIn9_RegIn3_RegIn7',
       32, 15, 7, 12, 6))
  def testConcatModelSizeFunction_Regularization(
      self, batch_size, num_alive_inputs3, num_alive_inputs4, reg_inputs3,
      reg_inputs4):
    conv3 = layers.conv2d(
        self.image, 19, [7, 5], stride=2, padding='SAME', scope='conv3')
    conv4 = layers.conv2d(
        self.image, 11, [3, 7], stride=2, padding='SAME', scope='conv4')
    concat = tf.concat([conv3, conv4], axis=3)
    model_size_loss = resource_function.model_size_function(
        concat.op, True, num_alive_inputs3 + num_alive_inputs4,
        num_alive_inputs3 + num_alive_inputs4, reg_inputs3 + reg_inputs4,
        reg_inputs3 + reg_inputs4, batch_size)

    expected_model_size_loss = 0
    self.assertEqual(expected_model_size_loss, model_size_loss)

  @parameterized.named_parameters(
      ('_BatchSize1_AliveIn19_AliveIn11', 1, 19, 11),
      ('_BatchSize32_AliveIn15_AliveIn7', 32, 15, 7))
  def testConcatActivationCountFunction_Cost(
      self, batch_size, num_alive_inputs3, num_alive_inputs4):
    conv3 = layers.conv2d(
        self.image, 19, [7, 5], stride=2, padding='SAME', scope='conv3')
    conv4 = layers.conv2d(
        self.image, 11, [3, 7], stride=2, padding='SAME', scope='conv4')
    concat = tf.concat([conv3, conv4], axis=3)
    activation_count_cost = resource_function.activation_count_function(
        concat.op, False, num_alive_inputs3 + num_alive_inputs4,
        num_alive_inputs3 + num_alive_inputs4, 19, 11, batch_size)

    expected_activation_count_cost = 0
    self.assertEqual(expected_activation_count_cost, activation_count_cost)

  @parameterized.named_parameters(
      ('_BatchSize1_AliveIn17_AliveIn19_RegIn17_RegIn19',
       1, 19, 11, 19, 11),
      ('_BatchSize32_AliveIn4_AliveIn9_RegIn3_RegIn7',
       32, 15, 7, 12, 6))
  def testConcatActivationCountFunction_Regularization(
      self, batch_size, num_alive_inputs3, num_alive_inputs4, reg_inputs3,
      reg_inputs4):
    conv3 = layers.conv2d(
        self.image, 19, [7, 5], stride=2, padding='SAME', scope='conv3')
    conv4 = layers.conv2d(
        self.image, 11, [3, 7], stride=2, padding='SAME', scope='conv4')
    concat = tf.concat([conv3, conv4], axis=3)
    activation_count_loss = resource_function.activation_count_function(
        concat.op, True, num_alive_inputs3 + num_alive_inputs4,
        num_alive_inputs3 + num_alive_inputs4, reg_inputs3 + reg_inputs4,
        reg_inputs3 + reg_inputs4, batch_size)

    expected_activation_count_loss = 0
    self.assertEqual(expected_activation_count_loss, activation_count_loss)

  def testBadHardware(self):
    with self.assertRaises(ValueError):
      _ = resource_function.latency_function_factory('apple', 66)
    with self.assertRaises(ValueError):
      _ = resource_function.latency_function_factory(None, 11)

  def testConvFlopsCoeff(self):
    tf.compat.v1.reset_default_graph()
    image = tf.constant(0.0, shape=[1, 11, 13, 17])
    layers.conv2d(image, 19, [7, 5], stride=2, padding='SAME', scope='conv1')
    conv_op = tf.get_default_graph().get_operation_by_name('conv1/Conv2D')
    # Divide by the input depth and the output depth to get the coefficient.
    expected_coeff = _flops(conv_op) / (17.0 * 19.0)
    actual_coeff = resource_function.flop_coeff(conv_op)
    self.assertNearRelatively(expected_coeff, actual_coeff)

  def testConvFlopsCoeffUnknownShape(self):
    tf.compat.v1.reset_default_graph()
    image = tf.placeholder(tf.float32, shape=[1, None, None, 17])
    net = layers.conv2d(
        image, 19, [7, 5], stride=2, padding='SAME', scope='conv1')
    self.conv_op = tf.get_default_graph().get_operation_by_name(
        'conv1/Conv2D')
    actual_coeff_tensor = resource_function.flop_coeff(self.conv_op)
    with self.cached_session() as sess:
      sess.run(tf.global_variables_initializer())
      actual_coeff, _ = sess.run([actual_coeff_tensor, net],
                                 feed_dict={image: np.zeros((1, 11, 13, 17))})
    # We cannot use the _flops function above because the shapes are not all
    # known in the graph.
    expected_coeff = 2940.0
    self.assertNearRelatively(expected_coeff, actual_coeff)

  def testConvTransposeFlopsCoeff(self):
    tf.compat.v1.reset_default_graph()
    image = tf.constant(0.0, shape=[1, 11, 13, 17])
    layers.conv2d_transpose(
        image, 29, [7, 5], stride=2, padding='SAME', scope='convt2')
    convt_op = tf.get_default_graph().get_operation_by_name(
        'convt2/conv2d_transpose')

    # Divide by the input depth and the output depth to get the coefficient.
    expected_coeff = _flops(convt_op) / (17.0 * 29.0)
    actual_coeff = resource_function.flop_coeff(convt_op)
    self.assertNearRelatively(expected_coeff, actual_coeff)

  def testFcFlopsCoeff(self):
    expected_coeff = _flops(self.matmul_op) / (19.0 * 23.0)
    actual_coeff = resource_function.flop_coeff(self.matmul_op)
    self.assertNearRelatively(expected_coeff, actual_coeff)

  def testConvNumWeightsCoeff(self):
    actual_coeff = resource_function.num_weights_coeff(self.conv_op)
    # The coefficient is just the filter size - 7 * 5 = 35:
    self.assertNearRelatively(35, actual_coeff)

  def testFcNumWeightsCoeff(self):
    actual_coeff = resource_function.num_weights_coeff(self.matmul_op)
    # The coefficient is 1.0, the number of weights is just inputs x outputs.
    self.assertNearRelatively(1.0, actual_coeff)

  def testDepthwiseConvFlopsCoeff(self):
    tf.compat.v1.reset_default_graph()
    image = tf.constant(0.0, shape=[1, 11, 13, 17])
    net = layers.conv2d(
        image, 10, [7, 5], stride=2, padding='SAME', scope='conv2')
    layers.separable_conv2d(
        net, None, [3, 2], depth_multiplier=1, padding='SAME', scope='dw1')
    dw_op = tf.get_default_graph().get_operation_by_name('dw1/depthwise')

    # Divide by the input depth (which is also the output depth) to get the
    # coefficient.
    expected_coeff = _flops(dw_op) / (10.0)
    actual_coeff = resource_function.flop_coeff(dw_op)
    self.assertNearRelatively(expected_coeff, actual_coeff)

  def test_conv3d_flops_coeff(self):
    tf.compat.v1.reset_default_graph()
    input_depth = 17
    output_depth = 10
    video = tf.zeros([1, 15, 12, 13, input_depth])
    _ = layers.conv3d(
        video, output_depth, [7, 5, 3], stride=2, padding='SAME', scope='conv')
    conv_op = tf.get_default_graph().get_operation_by_name('conv/Conv3D')
    # Divide by the input depth and the output depth to get the coefficient.
    expected_coeff = _flops(conv_op) / (input_depth * output_depth)
    actual_coeff = resource_function.flop_coeff(conv_op)
    self.assertNearRelatively(expected_coeff, actual_coeff)


def _flops(op):
  """Get the number of flops of a convolution, from the ops stats registry.

  Args:
    op: A tf.Operation object.

  Returns:
    The number os flops needed to evaluate conv_op.
  """
  return ops.get_stats_for_node_def(
      tf.get_default_graph(), op.node_def, 'flops').value


if __name__ == '__main__':
  tf.test.main()
