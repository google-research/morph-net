"""Tests for network_regularizers.flop_regularizer."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl.testing import parameterized
from morph_net.network_regularizers import flop_regularizer
from morph_net.network_regularizers import resource_function
from morph_net.testing import dummy_decorator

import numpy as np
import tensorflow.compat.v1 as tf

from tensorflow.contrib import layers as contrib_layers
from tensorflow.contrib import slim as contrib_slim
from tensorflow.contrib.slim.nets import resnet_v1

slim = contrib_slim

_coeff = resource_function.flop_coeff
NUM_CHANNELS = 3
SKIP_GAMMA_CONV3D = True  # TODO(e1) remove when gamma is supported.


class GammaFlopsTest(parameterized.TestCase, tf.test.TestCase):

  def BuildWithBatchNorm(self, fused):
    params = {
        'trainable': True,
        'normalizer_fn': slim.batch_norm,
        'normalizer_params': {
            'scale': True,
            'fused': fused,
        }
    }

    with slim.arg_scope([slim.layers.conv2d], **params):
      self.BuildModel()
    with self.cached_session():
      self.Init()

  def BuildModel(self):
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

    # op.name: 'Const'
    image = tf.constant(0.0, shape=[1, 17, 19, NUM_CHANNELS])
    # op.name: 'conv1/Conv2D'
    self.conv1 = slim.layers.conv2d(
        image, 13, [7, 5], padding='SAME', scope='conv1')
    self.conv2 = slim.layers.conv2d(
        image, 23, [1, 1], padding='SAME', scope='conv2')
    self.concat = tf.concat([self.conv1, self.conv2], 3)
    self.conv3 = slim.layers.conv2d(
        self.concat, 29, [3, 3], stride=2, padding='SAME', scope='conv3')
    self.conv4 = slim.layers.conv2d(
        self.concat, 31, [1, 1], stride=1, padding='SAME', scope='conv4')
    self.name_to_var = {v.op.name: v for v in tf.global_variables()}

  def AddRegularizer(self, input_boundary=None):
    self.gamma_flop_reg = flop_regularizer.GammaFlopsRegularizer(
        [self.conv3.op, self.conv4.op],
        gamma_threshold=0.45,
        input_boundary=input_boundary)

  def GetConv(self, name):
    return tf.get_default_graph().get_operation_by_name(name + '/Conv2D')

  def Init(self):
    tf.global_variables_initializer().run()
    gamma1 = self.name_to_var['conv1/BatchNorm/gamma']
    gamma1.assign([0.8] * 7 + [0.2] * 6).eval()
    gamma2 = self.name_to_var['conv2/BatchNorm/gamma']
    gamma2.assign([-0.7] * 11 + [0.1] * 12).eval()
    gamma3 = self.name_to_var['conv3/BatchNorm/gamma']
    gamma3.assign([0.6] * 10 + [-0.3] * 19).eval()
    gamma4 = self.name_to_var['conv4/BatchNorm/gamma']
    gamma4.assign([-0.5] * 17 + [-0.4] * 14).eval()

  def GetCost(self, conv):
    with self.cached_session():
      return self.gamma_flop_reg.get_cost(conv).eval()

  def GetLoss(self, conv):
    with self.cached_session():
      return self.gamma_flop_reg.get_regularization_term(conv).eval()

  def GetSourceOps(self):
    op_regularizer_manager = self.gamma_flop_reg.op_regularizer_manager
    return [
        op.name
        for op in op_regularizer_manager.ops
        if op_regularizer_manager.is_source_op(op)
    ]

  def testCost(self):
    self.BuildWithBatchNorm(fused=True)
    self.AddRegularizer(input_boundary=None)

    # Conv1 has 7 gammas above 0.45, and NUM_CHANNELS inputs (from the image).
    conv = self.GetConv('conv1')
    self.assertEqual(_coeff(conv) * 7 * NUM_CHANNELS, self.GetCost([conv]))

    # Conv2 has 11 gammas above 0.45, and NUM_CHANNELS inputs (from the image).
    conv = self.GetConv('conv2')
    self.assertEqual(_coeff(conv) * 11 * NUM_CHANNELS, self.GetCost([conv]))

    # Conv3 has 10 gammas above 0.45, and 7 + 11 inputs from conv1 and conv2.
    conv = self.GetConv('conv3')
    self.assertEqual(_coeff(conv) * 10 * 18, self.GetCost([conv]))

    # Conv4 has 17 gammas above 0.45, and 7 + 11 inputs from conv1 and conv2.
    conv = self.GetConv('conv4')
    self.assertEqual(_coeff(conv) * 17 * 18, self.GetCost([conv]))

    # Test that passing a list of convs sums their contributions:
    convs = [self.GetConv('conv3'), self.GetConv('conv4')]
    self.assertEqual(
        self.GetCost(convs[:1]) + self.GetCost(convs[1:]), self.GetCost(convs))

  def testInputBoundaryNone(self):
    self.BuildWithBatchNorm(fused=True)
    self.AddRegularizer(input_boundary=None)
    self.assertCountEqual(self.GetSourceOps(), [
        'conv1/BatchNorm/FusedBatchNormV3', 'conv2/BatchNorm/FusedBatchNormV3',
        'conv3/BatchNorm/FusedBatchNormV3', 'conv4/BatchNorm/FusedBatchNormV3'
    ])

  def testInputBoundaryConv3(self):
    # Only block one path, can still reach all other convolutions.
    self.BuildWithBatchNorm(fused=True)
    self.AddRegularizer(input_boundary=[self.conv3.op])
    self.assertCountEqual(self.GetSourceOps(), [
        'conv1/BatchNorm/FusedBatchNormV3', 'conv2/BatchNorm/FusedBatchNormV3',
        'conv4/BatchNorm/FusedBatchNormV3'
    ])

  def testInputBoundaryConv3And4(self):
    # Block both paths, can no longer reach Concat and earlier convolutions.
    self.BuildWithBatchNorm(fused=True)
    self.AddRegularizer(input_boundary=[self.conv3.op, self.conv4.op])
    self.assertCountEqual(self.GetSourceOps(), [])

  def testInputBoundaryConcat(self):
    # Block concat, can only see conv3 and conv4.
    self.BuildWithBatchNorm(fused=True)
    self.AddRegularizer(input_boundary=[self.concat.op])
    self.assertCountEqual(self.GetSourceOps(), [
        'conv3/BatchNorm/FusedBatchNormV3', 'conv4/BatchNorm/FusedBatchNormV3'
    ])

  def testLossDecorated(self):
    self.BuildWithBatchNorm(True)
    self.AddRegularizer()
    # Create network regularizer with DummyDecorator op regularization.
    self.gamma_flop_reg = flop_regularizer.GammaFlopsRegularizer(
        [self.conv3.op, self.conv4.op],
        gamma_threshold=0.45,
        regularizer_decorator=dummy_decorator.DummyDecorator,
        decorator_parameters={'scale': 0.5})

    all_convs = [
        o for o in tf.get_default_graph().get_operations() if o.type == 'Conv2D'
    ]
    total_reg_term = 1410376.375
    self.assertAllClose(total_reg_term * 0.5, self.GetLoss(all_convs))
    self.assertAllClose(total_reg_term * 0.5, self.GetLoss([]))


class GammaFlopDecoratedTest(parameterized.TestCase, tf.test.TestCase):
  """A simple test to check the op regularizer decorator with flop regularizer.
  """

  def testLossCostDecorated(self):
    params = {'trainable': True, 'normalizer_fn': slim.batch_norm,
              'normalizer_params': {'scale': True}}

    with slim.arg_scope([slim.layers.conv2d], **params):
      image = tf.constant(0.0, shape=[1, 3, 3, NUM_CHANNELS])
      conv1 = slim.layers.conv2d(
          image, 2, kernel_size=1, padding='SAME', scope='conv1')
    with self.cached_session():
      tf.global_variables_initializer().run()
      name_to_var = {v.op.name: v for v in tf.global_variables()}
      gamma1 = name_to_var['conv1/BatchNorm/gamma']
      gamma1.assign([1] * 2).eval()

    self.gamma_flop_reg = flop_regularizer.GammaFlopsRegularizer(
        [conv1.op],
        gamma_threshold=0.1,
        regularizer_decorator=dummy_decorator.DummyDecorator,
        decorator_parameters={'scale': 0.5})

    conv = tf.get_default_graph().get_operation_by_name('conv1/Conv2D')
    # we compare the computed cost and regularization calculated as follows:
    # reg_term = op_coeff * (number_of_inputs * (regularization=2 * 0.5) +
    # number_of_outputs * (input_regularization=0))
    # number_of_flops = coeff * number_of_inputs * number_of_outputs.
    with self.cached_session():
      predicted_reg = self.gamma_flop_reg.get_regularization_term([conv]).eval()
      self.assertEqual(_coeff(conv) * NUM_CHANNELS * 1, predicted_reg)
      predicted_cost = self.gamma_flop_reg.get_cost([conv]).eval()
      self.assertEqual(_coeff(conv) * 2 * NUM_CHANNELS, predicted_cost)


class GroupLassoFlopDecoratedTest(parameterized.TestCase, tf.test.TestCase):
  """A test to check the op regularizer decorator for group lasso regularizer.
  """

  def testLossCostDecorated(self):
    image = tf.constant(0.0, shape=[1, 3, 3, 3])
    kernel = tf.ones([1, 1, 3, 2])

    pred = tf.nn.conv2d(image, kernel, strides=[1, 1, 1, 1], padding='SAME')
    conv = pred.op

    self.group_lasso_reg = flop_regularizer.GroupLassoFlopsRegularizer(
        [conv],
        0.1,
        l1_fraction=0,
        regularizer_decorator=dummy_decorator.DummyDecorator,
        decorator_parameters={'scale': 0.5})
    # we compare the computed cost and regularization calculated as follows:
    # reg_term = op_coeff * (number_of_inputs * (regularization=2 * 0.5) +
    # number_of_outputs * (input_regularization=0))
    # number_of_flops = coeff * number_of_inputs * number_of_outputs.
    with self.cached_session():
      pred_reg = self.group_lasso_reg.get_regularization_term([conv]).eval()
      self.assertEqual(_coeff(conv) * 3 * 1, pred_reg)
      pred_cost = self.group_lasso_reg.get_cost([conv]).eval()
      self.assertEqual(_coeff(conv) * 2 * NUM_CHANNELS, pred_cost)


class GammaFlopsWithDepthwiseConvTestBase(tf.test.TestCase):
  """Test flop_regularizer for a network with depthwise convolutions."""

  def BuildWithBatchNorm(self):
    params = {
        'trainable': True,
        'normalizer_fn': slim.batch_norm,
        'normalizer_params': {
            'scale': True
        }
    }
    ops_with_batchnorm = [slim.layers.conv2d]
    if self._depthwise_use_batchnorm:
      ops_with_batchnorm.append(slim.layers.separable_conv2d)

    with slim.arg_scope(ops_with_batchnorm, **params):
      self.BuildModel()

  def BuildModel(self):
    # Our test model is:
    #
    #         -> dw1 --> conv1 --+
    #        /                   |
    #  image                     [concat] --> conv3
    #        \                   |
    #         -> conv2 --> dw2 --+
    #
    # (the model has two "outputs", conv3).
    #
    image = tf.constant(0.0, shape=[1, 17, 19, NUM_CHANNELS])
    dw1 = slim.layers.separable_conv2d(
        image, None, [3, 3], depth_multiplier=1, stride=1, scope='dw1')
    conv1 = slim.layers.conv2d(dw1, 13, [7, 5], padding='SAME', scope='conv1')
    conv2 = slim.layers.conv2d(image, 23, [1, 1], padding='SAME', scope='conv2')
    dw2 = slim.layers.separable_conv2d(
        conv2, None, [5, 5], depth_multiplier=1, stride=1, scope='dw2')
    concat = tf.concat([conv1, dw2], 3)
    self.conv3 = slim.layers.conv2d(
        concat, 29, [3, 3], stride=2, padding='SAME', scope='conv3')
    self.name_to_var = {v.op.name: v for v in tf.global_variables()}

    regularizer_blacklist = None
    if self._depthwise_use_batchnorm:
      regularizer_blacklist = ['dw1']
    self.gamma_flop_reg = flop_regularizer.GammaFlopsRegularizer(
        [self.conv3.op], gamma_threshold=0.45,
        regularizer_blacklist=regularizer_blacklist)

  def GetConv(self, name):
    return tf.get_default_graph().get_operation_by_name(
        name + ('/Conv2D' if 'conv' in name else '/depthwise'))

  def GetGammaAbsValue(self, name):
    gamma_op = tf.get_default_graph().get_operation_by_name(name +
                                                            '/BatchNorm/gamma')
    with self.cached_session():
      gamma = gamma_op.outputs[0].eval()
    return np.abs(gamma)

  def Init(self):
    tf.global_variables_initializer().run()
    gamma1 = self.name_to_var['conv1/BatchNorm/gamma']
    gamma1.assign([0.8] * 7 + [0.2] * 6).eval()
    gamma2 = self.name_to_var['conv2/BatchNorm/gamma']
    gamma2.assign([-0.7] * 11 + [0.1] * 12).eval()
    gamma3 = self.name_to_var['conv3/BatchNorm/gamma']
    gamma3.assign([0.6] * 10 + [-0.3] * 19).eval()
    # Initialize gamma for depthwise convs only if there are Batchnorm for them.
    if self._depthwise_use_batchnorm:
      gammad1 = self.name_to_var['dw1/BatchNorm/gamma']
      gammad1.assign([-0.3] * 1 + [-0.9] * 2).eval()
      gammad2 = self.name_to_var['dw2/BatchNorm/gamma']
      gammad2.assign([0.3] * 5 + [0.9] * 10 + [-0.1] * 8).eval()

  def cost(self, conv):  # pylint: disable=invalid-name
    with self.cached_session():
      cost = self.gamma_flop_reg.get_cost(conv)
      return cost.eval() if isinstance(cost, tf.Tensor) else cost

  def loss(self, conv):  # pylint: disable=invalid-name
    with self.cached_session():
      reg = self.gamma_flop_reg.get_regularization_term(conv)
      return reg.eval() if isinstance(reg, tf.Tensor) else reg


class GammaFlopsWithDepthwiseConvTest(GammaFlopsWithDepthwiseConvTestBase):
  """Test flop_regularizer for a network with depthwise convolutions."""

  def setUp(self):
    self._depthwise_use_batchnorm = True
    super(GammaFlopsWithDepthwiseConvTest, self).setUp()
    self.BuildWithBatchNorm()
    with self.cached_session():
      self.Init()

  def testCost(self):
    # Dw1 has 2 gammas above 0.45 out of NUM_CHANNELS inputs (from the image),
    # but because the input doesn't have a regularizer, it has no way of
    # removing the channels, so the channel count is still NUM_CHANNELS.
    conv = self.GetConv('dw1')
    self.assertEqual(_coeff(conv) * NUM_CHANNELS, self.cost([conv]))

    # Conv1 has 7 gammas above 0.45, and NUM_CHANNELS inputs (from dw1).
    conv = self.GetConv('conv1')
    self.assertEqual(_coeff(conv) * 7 * NUM_CHANNELS, self.cost([conv]))

    # Conv2 has 11 active + 12 inactive, while Dw2 has 5 inactive, 10 active and
    # 8 active. Their max (or) has 15 active and 8 inactive.
    # Conv2 has NUM_CHANNELS inputs (from the image).
    conv = self.GetConv('conv2')
    self.assertEqual(_coeff(conv) * 15 * NUM_CHANNELS, self.cost([conv]))

    # Dw2 has 15 out of 23 inputs (from the Conv2).
    conv = self.GetConv('dw2')
    self.assertEqual(_coeff(conv) * 15, self.cost([conv]))

    # Conv3 has 10 gammas above 0.45, and 7 + 15 inputs from conv1 and dw2.
    conv = self.GetConv('conv3')
    self.assertEqual(_coeff(conv) * 10 * 22, self.cost([conv]))

  def testRegularizer(self):
    # Dw1 depthwise convolution is connected to the input (no regularizer).
    conv = self.GetConv('dw1')
    # Although the effective regularizer for dw is computed as below:
    # gamma = self.GetGammaAbsValue('dw1')
    # expected_loss = _coeff(conv) * gamma.sum()
    # Since the input is not regularized, dw does not return a regularizer.
    expected_loss = 0.0
    self.assertNear(expected_loss, self.loss([conv]), expected_loss * 1e-5)

    # Conv1 takes Dw1 as input, its input regularizer is from dw1.
    conv = self.GetConv('conv1')
    gamma = self.GetGammaAbsValue('conv1')
    # The effective size for dw can be computed from its gamma, and
    # the loss may be computed as follows:
    # gamma_dw = self.GetGammaAbsValue('dw1')
    # expected_loss = _coeff(conv) * (
    #     gamma.sum() * (gamma_dw > 0.45).sum() + gamma_dw.sum() *
    #     (gamma > 0.45).sum())
    # However, since dw cannot change shape because its input doesn't have a
    # regularizer, the real loss we expect should be:
    expected_loss = _coeff(conv) * (gamma.sum() * NUM_CHANNELS)
    self.assertNear(expected_loss, self.loss([conv]), expected_loss * 1e-5)

    # Dw2 depthwise convolution is connected to conv2 (grouped regularizer).
    conv = self.GetConv('conv2')
    gamma_conv = self.GetGammaAbsValue('conv2')
    dw = self.GetConv('dw2')
    gamma_dw = self.GetGammaAbsValue('dw2')
    gamma = np.maximum(gamma_dw, gamma_conv).sum()
    expected_loss = _coeff(conv) * (gamma * 3 + (gamma > 0.45).sum() * 0)
    self.assertNear(expected_loss, self.loss([conv]), expected_loss * 1e-5)
    expected_loss = _coeff(dw) * gamma * 2
    self.assertNear(expected_loss, self.loss([dw]), expected_loss * 1e-5)


class GammaFlopsWithDepthwiseConvNoBatchNormTest(
    GammaFlopsWithDepthwiseConvTestBase):
  """Test flop_regularizer for un-batchnormed depthwise convolutions.

  This test is used to confirm that when depthwise convolution is not BNed, it
  will not be considered towards the regularizer, but it will be counted towards
  the cost.
  This design choice is for backward compatibility for users who did not
  regularize depthwise convolutions. However, the cost will be reported
  regardless in order to be faithful to the real computation complexity.
  """

  def setUp(self):
    self._depthwise_use_batchnorm = False
    super(GammaFlopsWithDepthwiseConvNoBatchNormTest, self).setUp()
    self.BuildWithBatchNorm()
    with self.cached_session():
      self.Init()

  def testCost(self):
    # Dw1 has NUM_CHANNELS inputs (from the image).
    conv = self.GetConv('dw1')
    self.assertEqual(_coeff(conv) * 3, self.cost([conv]))

    # Conv1 has 7 gammas above 0.45, and 3 inputs (from dw1).
    conv = self.GetConv('conv1')
    self.assertEqual(_coeff(conv) * 7 * 3, self.cost([conv]))

    # Conv2 has 11 active outputs and NUM_CHANNELS inputs (from the image).
    conv = self.GetConv('conv2')
    self.assertEqual(_coeff(conv) * 11 * NUM_CHANNELS, self.cost([conv]))

    # Dw2 has 11 inputs (pass-through from the Conv2).
    conv = self.GetConv('dw2')
    self.assertEqual(_coeff(conv) * 11, self.cost([conv]))

    # Conv3 has 10 gammas above 0.45, and 7 + 11 inputs from conv1 and dw2.
    conv = self.GetConv('conv3')
    self.assertEqual(_coeff(conv) * 10 * 18, self.cost([conv]))

  def testRegularizer(self):
    # Dw1 depthwise convolution is connected to the input (no regularizer).
    conv = self.GetConv('dw1')
    expected_loss = 0.0
    self.assertNear(expected_loss, self.loss([conv]), expected_loss * 1e-5)

    # Conv1 takes Dw1 as input, but it's not affected by dw1 because depthwise
    # is not BNed.
    conv = self.GetConv('conv1')
    gamma = self.GetGammaAbsValue('conv1')
    expected_loss = _coeff(conv) * (gamma.sum() * NUM_CHANNELS)
    self.assertNear(expected_loss, self.loss([conv]), expected_loss * 1e-5)

    # Dw2 depthwise convolution is connected to conv2 (pass through).
    dw = self.GetConv('dw2')
    gamma = self.GetGammaAbsValue('conv2')
    expected_loss = _coeff(dw) * gamma.sum() * 2
    self.assertNear(expected_loss, self.loss([dw]), expected_loss * 1e-5)


class GammaFlopResidualConnectionsLossTest(tf.test.TestCase):
  """Tests flop_regularizer for a network with residual connections."""

  def setUp(self):
    super(GammaFlopResidualConnectionsLossTest, self).setUp()
    tf.set_random_seed(7)
    self._threshold = 0.6

  def BuildModel(self, resnet_fn, block_fn):
    # We use this model as a test case because the slim.nets.resnet module is
    # used in some production.
    #
    # The model looks as follows:
    #
    # Image --> unit_1/shortcut
    # Image --> unit_1/conv1 --> unit_1/conv2 --> unit_1/conv3
    #
    # unit_1/shortcut + unit_1/conv3 --> unit_1 (residual connection)
    #
    # unit_1 --> unit_2/conv1  -> unit_2/conv2 --> unit_2/conv3
    #
    # unit_1 + unit_2/conv3 --> unit_2 (residual connection)
    #
    # In between, there are strided convolutions and pooling ops, but these
    # should not affect the regularizer.
    blocks = [
        block_fn('block1', base_depth=7, num_units=2, stride=2),
    ]
    image = tf.constant(0.0, shape=[1, 2, 2, NUM_CHANNELS])
    net = resnet_fn(
        image, blocks, include_root_block=False, is_training=False)[0]
    net = tf.reduce_mean(net, axis=(1, 2))
    return slim.layers.fully_connected(net, 23, scope='FC')

  def BuildGraphWithBatchNorm(self, resnet_fn, block_fn):
    params = {
        'trainable': True,
        'normalizer_fn': slim.batch_norm,
        'normalizer_params': {
            'scale': True
        }
    }

    with slim.arg_scope([slim.layers.conv2d, slim.layers.separable_conv2d],
                        **params):
      self.net = self.BuildModel(resnet_fn, block_fn)

  def InitGamma(self):
    assignments = []
    gammas = {}
    for v in tf.global_variables():
      if v.op.name.endswith('/gamma'):
        assignments.append(v.assign(tf.random_uniform(v.shape)))
        gammas[v.op.name] = v
    with self.cached_session() as s:
      s.run(assignments)
      self._gammas = s.run(gammas)

  def GetGamma(self, short_name):
    tokens = short_name.split('/')
    name = ('resnet_v1/block1/' + tokens[0] + '/bottleneck_v1/' + tokens[1] +
            '/BatchNorm/gamma')
    return self._gammas[name]

  def GetOp(self, short_name):
    if short_name == 'FC':
      return tf.get_default_graph().get_operation_by_name('FC/MatMul')
    tokens = short_name.split('/')
    name = ('resnet_v1/block1/' + tokens[0] + '/bottleneck_v1/' + tokens[1] +
            '/Conv2D')
    return tf.get_default_graph().get_operation_by_name(name)

  def NumAlive(self, short_name):
    return np.sum(self.GetGamma(short_name) > self._threshold)

  def GetCoeff(self, short_name):
    return _coeff(self.GetOp(short_name))

  def testCost(self):
    self.BuildGraphWithBatchNorm(resnet_v1.resnet_v1, resnet_v1.resnet_v1_block)
    self.InitGamma()
    res_alive = np.logical_or(
        np.logical_or(
            self.GetGamma('unit_1/shortcut') > self._threshold,
            self.GetGamma('unit_1/conv3') > self._threshold),
        self.GetGamma('unit_2/conv3') > self._threshold)

    self.gamma_flop_reg = flop_regularizer.GammaFlopsRegularizer(
        [self.net.op], self._threshold)

    expected = {}
    expected['unit_1/shortcut'] = (
        self.GetCoeff('unit_1/shortcut') * np.sum(res_alive) * NUM_CHANNELS)
    expected['unit_1/conv1'] = (
        self.GetCoeff('unit_1/conv1') * self.NumAlive('unit_1/conv1') *
        NUM_CHANNELS)
    expected['unit_1/conv2'] = (
        self.GetCoeff('unit_1/conv2') * self.NumAlive('unit_1/conv2') *
        self.NumAlive('unit_1/conv1'))
    expected['unit_1/conv3'] = (
        self.GetCoeff('unit_1/conv3') * np.sum(res_alive) *
        self.NumAlive('unit_1/conv2'))
    expected['unit_2/conv1'] = (
        self.GetCoeff('unit_2/conv1') * self.NumAlive('unit_2/conv1') *
        np.sum(res_alive))
    expected['unit_2/conv2'] = (
        self.GetCoeff('unit_2/conv2') * self.NumAlive('unit_2/conv2') *
        self.NumAlive('unit_2/conv1'))
    expected['unit_2/conv3'] = (
        self.GetCoeff('unit_2/conv3') * np.sum(res_alive) *
        self.NumAlive('unit_2/conv2'))
    expected['FC'] = 2.0 * np.sum(res_alive) * 23.0

    # TODO(e1): Is there a way to use Parametrized Tests to make this more
    # elegant?
    with self.cached_session():
      for short_name in expected:
        cost = self.gamma_flop_reg.get_cost([self.GetOp(short_name)]).eval()
        self.assertEqual(expected[short_name], cost)

      self.assertEqual(
          sum(expected.values()),
          self.gamma_flop_reg.get_cost().eval())


class GammaConv3DTest(parameterized.TestCase, tf.test.TestCase):

  @parameterized.named_parameters(
      ('_all_alive', 1e-3, 3),
      ('_one_alive', .35, 1),
      ('_all_dead', .99, 0),
  )
  def test_simple_conv3d(self, threshold, expected_alive):
    # TODO(e1) remove when gamma is supported.
    # This test works if reshape not set to be a handled by
    # leaf_op_handler.LeafOpHandler() in op_handlers.py. However this changes
    # brakes other tests for reasons to be investigated.
    if SKIP_GAMMA_CONV3D:
      return

    def fused_batch_norm3d(*args, **kwargs):
      if args:
        inputs = args[0]
        args = args[1:]
      else:
        inputs = kwargs.pop('inputs')
      shape = inputs.shape
      # inputs is assumed to be NHWTC (T is for time).
      batch_size = shape[0]
      # space x time cube is reshaped to be 2D with dims:
      # H, W, T --> H, W * T
      # the idea is that batch norm only needs this to collect spacial stats.
      target_shape = [batch_size, shape[1], shape[2] * shape[3], shape[4]]
      inputs = tf.reshape(inputs, target_shape, name='Reshape/to2d')
      normalized = slim.batch_norm(inputs, *args, **kwargs)
      return tf.reshape(normalized, shape, name='Reshape/to3d')

    gamma_val = [0.5, 0.3, 0.2]
    num_inputs = 4
    batch_size = 2
    video = tf.zeros([batch_size, 8, 8, 8, num_inputs])
    kernel = [5, 5, 5]
    num_outputs = 3
    net = slim.conv3d(
        video,
        num_outputs,
        kernel,
        padding='SAME',
        normalizer_fn=fused_batch_norm3d,
        normalizer_params={
            'scale': True,
            'fused': True
        },
        scope='vconv1')
    self.assertLen(net.shape.as_list(), 5)
    shape = net.shape.as_list()
    # The number of applications is the number of elements in the [HWT] tensor.
    num_applications = shape[1] * shape[2] * shape[3]
    application_cost = num_inputs * kernel[0] * kernel[1] * kernel[2]
    name_to_var = {v.op.name: v for v in tf.global_variables()}
    flop_reg = flop_regularizer.GammaFlopsRegularizer(
        [net.op,
         tf.get_default_graph().get_operation_by_name('vconv1/Conv3D')],
        threshold,
        force_group=['vconv1/Reshape/to3d|vconv1/Reshape/to2d|vconv1/Conv3D'])
    gamma = name_to_var['vconv1/BatchNorm/gamma']
    with self.session():
      tf.global_variables_initializer().run()
      gamma.assign(gamma_val).eval()

      self.assertAllClose(
          flop_reg.get_cost(),
          2 * expected_alive * num_applications * application_cost)

      raw_cost = 2 * num_outputs * num_applications * application_cost
      self.assertAllClose(flop_reg.get_regularization_term(),
                          raw_cost * np.mean(gamma_val))


class GroupLassoFlopRegTest(parameterized.TestCase, tf.test.TestCase):

  def assertNearRelatively(self, expected, actual):
    self.assertNear(expected, actual, expected * 1e-6)

  def testFlopRegularizer(self):
    tf.reset_default_graph()
    tf.set_random_seed(7907)
    with slim.arg_scope(
        [slim.layers.conv2d, slim.layers.conv2d_transpose],
        weights_initializer=tf.random_normal_initializer):
      # Our test model is:
      #
      #         -> conv1 --+
      #        /           |--[concat]
      #  image --> conv2 --+
      #        \
      #         -> convt
      #
      # (the model has two "outputs", convt and concat).
      #
      image = tf.constant(0.0, shape=[1, 17, 19, NUM_CHANNELS])
      conv1 = slim.layers.conv2d(
          image, 13, [7, 5], padding='SAME', scope='conv1')
      conv2 = slim.layers.conv2d(
          image, 23, [1, 1], padding='SAME', scope='conv2')
      self.concat = tf.concat([conv1, conv2], 3)
      self.convt = slim.layers.conv2d_transpose(
          image, 29, [7, 5], stride=3, padding='SAME', scope='convt')
      self.name_to_var = {v.op.name: v for v in tf.global_variables()}
    with self.cached_session():
      tf.global_variables_initializer().run()

    threshold = 1.0
    flop_reg = flop_regularizer.GroupLassoFlopsRegularizer(
        [self.concat.op, self.convt.op], threshold=threshold, l1_fraction=0)

    with self.cached_session() as s:
      evaluated_vars = s.run(self.name_to_var)

    def group_norm(weights, axis=(0, 1, 2)):  # pylint: disable=invalid-name
      return np.sqrt(np.mean(weights**2, axis=axis))

    reg_vectors = {
        'conv1': group_norm(evaluated_vars['conv1/weights'], (0, 1, 2)),
        'conv2': group_norm(evaluated_vars['conv2/weights'], (0, 1, 2)),
        'convt': group_norm(evaluated_vars['convt/weights'], (0, 1, 3))
    }

    num_alive = {k: np.sum(r > threshold) for k, r in reg_vectors.items()}
    total_outputs = (
        reg_vectors['conv1'].shape[0] + reg_vectors['conv2'].shape[0])
    total_alive_outputs = sum(num_alive.values())
    assert total_alive_outputs > 0, (
        'All outputs are dead - test is trivial. Decrease the threshold.')
    assert total_alive_outputs < total_outputs, (
        'All outputs are alive - test is trivial. Increase the threshold.')

    coeff1 = _coeff(_get_op('conv1/Conv2D'))
    coeff2 = _coeff(_get_op('conv2/Conv2D'))
    coefft = _coeff(_get_op('convt/conv2d_transpose'))

    expected_flop_cost = NUM_CHANNELS * (
        coeff1 * num_alive['conv1'] + coeff2 * num_alive['conv2'] +
        coefft * num_alive['convt'])
    expected_reg_term = NUM_CHANNELS * (
        coeff1 * np.sum(reg_vectors['conv1']) + coeff2 * np.sum(
            reg_vectors['conv2']) + coefft * np.sum(reg_vectors['convt']))
    with self.cached_session():
      self.assertEqual(
          round(expected_flop_cost), round(flop_reg.get_cost().eval()))
      self.assertNearRelatively(expected_reg_term,
                                flop_reg.get_regularization_term().eval())

  @parameterized.named_parameters(
      ('_matmul', False),
      ('_slim', True),
  )
  def testFlopRegularizerWithMatMul(self, use_contrib):
    """Test the MatMul op regularizer with FLOP network regularizer."""

    # Set up a two layer fully connected network.
    tf.reset_default_graph()
    # Create the variables, and corresponding values.
    x = tf.constant(1.0, shape=[2, 6], name='x', dtype=tf.float32)
    w = tf.get_variable('w', shape=(6, 4), dtype=tf.float32)
    b = tf.get_variable('b', shape=(4), dtype=tf.float32)
    w2 = tf.get_variable('w2', shape=(4, 1), dtype=tf.float32)
    b2 = tf.get_variable('b2', shape=(1), dtype=tf.float32)
    w_value = np.arange(24).reshape((6, 4)).astype('float32')
    b_value = np.arange(4).reshape(4).astype('float32')
    w2_value = np.arange(21, 25).reshape((4, 1)).astype('float32')
    b2_value = np.arange(1).astype('float32')
    fc1_name = 'matmul1/MatMul'
    fc2_name = 'matmul2/MatMul'

    # Build the test network model.
    if use_contrib:
      net = contrib_layers.fully_connected(
          x,
          4,
          scope='matmul1',
          weights_initializer=tf.constant_initializer(w_value),
          biases_initializer=tf.constant_initializer(b_value))
      output = contrib_layers.fully_connected(
          net,
          1,
          scope='matmul2',
          weights_initializer=tf.constant_initializer(w2_value),
          biases_initializer=tf.constant_initializer(b2_value))
      with self.cached_session():
        tf.global_variables_initializer().run()
    else:
      net = tf.nn.relu(tf.matmul(x, w, name=fc1_name) + b)
      output = tf.nn.relu(tf.matmul(net, w2, name=fc2_name) + b2)
      # Assign values to network parameters.
      with self.cached_session() as session:
        session.run([
            w.assign(w_value),
            b.assign(b_value),
            w2.assign(w2_value),
            b2.assign(b2_value)
        ])

    # Create FLOPs network regularizer.
    threshold = 32.0
    flop_reg = flop_regularizer.GroupLassoFlopsRegularizer([output.op],
                                                           threshold, 0)

    # Compute expected regularization vector and alive vector.
    def group_norm(weights, axis=(0, 1, 2)):  # pylint: disable=invalid-name
      return np.sqrt(np.mean(weights**2, axis=axis))
    expected_reg_vector1 = group_norm(w_value, axis=(0,))
    expected_reg_vector2 = group_norm(w2_value, axis=(0,))
    # Since the threshold is 32, and the L2 norm of columns in matrix w is
    # (29.66479301, 31.71750259, 33.82307053, 35.97220993). Thus, the alive
    # vector for w should be (0, 0, 1, 1). The alive vector is [1] since the L2
    # norm for w2_value is 45.055521 > 32.
    # Compute the expected FLOPs cost and expected regularization term.
    matmul1_live_input = 6
    matmul1_live_output = sum(expected_reg_vector1 > threshold)
    matmul2_live_output = sum(expected_reg_vector2 > threshold)
    expected_flop_cost = (
        _coeff(_get_op(fc1_name)) * matmul1_live_input * matmul1_live_output +
        _coeff(_get_op(fc2_name)) * matmul1_live_output * matmul2_live_output)
    regularizer1 = np.sum(expected_reg_vector1)
    regularizer2 = np.sum(expected_reg_vector2)
    expected_reg_term = (
        _coeff(_get_op(fc1_name)) * matmul1_live_input * regularizer1 +
        _coeff(_get_op(fc2_name)) * (matmul1_live_output * regularizer2 +
                                     matmul2_live_output * regularizer1))
    with self.cached_session() as session:
      self.assertEqual(
          round(flop_reg.get_cost().eval()), round(expected_flop_cost))
      self.assertNearRelatively(flop_reg.get_regularization_term().eval(),
                                expected_reg_term)

  def testFlopRegularizerDontConvertToVariable(self):
    tf.reset_default_graph()
    tf.set_random_seed(1234)

    x = tf.constant(1.0, shape=[2, 6], name='x', dtype=tf.float32)
    w = tf.Variable(tf.truncated_normal([6, 4], stddev=1.0), use_resource=True)
    net = tf.matmul(x, w)

    # Create FLOPs network regularizer.
    threshold = 0.9
    flop_reg = flop_regularizer.GroupLassoFlopsRegularizer([net.op], threshold,
                                                           0)

    with self.cached_session():
      tf.global_variables_initializer().run()
      flop_reg.get_regularization_term().eval()

  def test_group_lasso_conv3d(self):
    shape = [3, 3, 3]
    video = tf.zeros([2, 3, 3, 3, 1])
    net = slim.conv3d(
        video,
        5,
        shape,
        padding='VALID',
        weights_initializer=tf.glorot_normal_initializer(),
        scope='vconv1')
    conv3d_op = tf.get_default_graph().get_operation_by_name('vconv1/Conv3D')
    conv3d_weights = conv3d_op.inputs[1]

    threshold = 0.09
    flop_reg = flop_regularizer.GroupLassoFlopsRegularizer([net.op],
                                                           threshold=threshold)
    norm = tf.sqrt(tf.reduce_mean(tf.square(conv3d_weights), [0, 1, 2, 3]))
    alive = tf.reduce_sum(tf.cast(norm > threshold, tf.float32))
    with self.session():
      flop_coeff = 2 * shape[0] * shape[1] * shape[2]
      tf.compat.v1.global_variables_initializer().run()
      self.assertAllClose(flop_reg.get_cost(), flop_coeff * alive)
      self.assertAllClose(flop_reg.get_regularization_term(),
                          flop_coeff * tf.reduce_sum(norm))


def _get_op(name):  # pylint: disable=invalid-name
  return tf.get_default_graph().get_operation_by_name(name)


if __name__ == '__main__':
  tf.test.main()
