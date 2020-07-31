"""Tests for morph_net.tools.configurable_ops."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import functools
import json
import os

from absl import flags

from absl.testing import parameterized

from morph_net.tools import configurable_ops as ops
from morph_net.tools import test_module as tm

import tensorflow.compat.v1 as tf
from tensorflow.compat.v1.keras import layers as keras_layers
import tensorflow.contrib as tf_contrib

from tensorflow.contrib import layers
from tensorflow.contrib.framework import add_arg_scope
from tensorflow.contrib.framework import arg_scope
from tensorflow.contrib.slim.nets import resnet_utils
from tensorflow.contrib.slim.nets import resnet_v1
from tensorflow.contrib.slim.nets import resnet_v2


FLAGS = flags.FLAGS


@add_arg_scope
def mock_fully_connected(*args, **kwargs):
  return {'mock_name': 'myfully_connected', 'args': args, 'kwargs': kwargs}


@add_arg_scope
def mock_conv2d(*args, **kwargs):
  return {'mock_name': 'myconv2d', 'args': args, 'kwargs': kwargs}


@add_arg_scope
def mock_separable_conv2d(*args, **kwargs):
  return {'mock_name': 'myseparable_conv2d', 'args': args, 'kwargs': kwargs}


@add_arg_scope
def mock_concat(*args, **kwargs):
  return {'mock_name': 'myconcat', 'args': args, 'kwargs': kwargs}


@add_arg_scope
def mock_add_n(*args, **kwargs):
  return {'mock_name': 'myaddn', 'args': args, 'kwargs': kwargs}


class ConfigurableKerasLayersTest(parameterized.TestCase, tf.test.TestCase):

  def setUp(self):
    super(ConfigurableKerasLayersTest, self).setUp()
    tf.reset_default_graph()
    self.inputs_shape = [1, 10, 10, 3]
    self.inputs = tf.ones(self.inputs_shape, dtype=tf.float32)

  def testConfigurableConv2DFunctionality(self):
    out = ops.ConfigurableConv2D(filters=5, kernel_size=2)(self.inputs)
    expected = keras_layers.Conv2D(filters=5, kernel_size=2)(self.inputs)
    self.assertAllEqual(out.shape, expected.shape)
    self.assertIn('configurable_conv2d/ConfigurableConv2D',
                  [op.name for op in tf.get_default_graph().get_operations()])

  def testConfigurableSeparableConv2DFunctionality(self):
    out = ops.ConfigurableSeparableConv2D(filters=5, kernel_size=2)(self.inputs)
    expected = keras_layers.SeparableConv2D(filters=5, kernel_size=2)(
        self.inputs)
    self.assertAllEqual(out.shape, expected.shape)
    self.assertIn('configurable_separable_conv2d/separable_conv2d',
                  [op.name for op in tf.get_default_graph().get_operations()])

  def testConfigurableDenseFunctionality(self):
    out = ops.ConfigurableDense(units=5)(self.inputs)
    expected = keras_layers.Dense(units=5)(self.inputs)
    self.assertAllEqual(out.shape, expected.shape)
    self.assertIn('configurable_dense/Tensordot/MatMul',
                  [op.name for op in tf.get_default_graph().get_operations()])

  def testConfigurableConv2DParameterization(self):
    # with default name
    conv1 = ops.ConfigurableConv2D(
        parameterization={'configurable_conv2d/ConfigurableConv2D': 1},
        filters=10, kernel_size=3)
    out1 = conv1(self.inputs)
    self.assertEqual(1, out1.shape.as_list()[-1])

    # with custom name
    conv2 = ops.ConfigurableConv2D(
        parameterization={'conv2/ConfigurableConv2D': 2}, filters=10,
        kernel_size=3, name='conv2')
    out2 = conv2(self.inputs)
    self.assertEqual(2, out2.shape.as_list()[-1])

  def testConfigurableSeparableConv2DParameterization(self):
    # with default name
    conv1 = ops.ConfigurableSeparableConv2D(
        parameterization={'configurable_separable_conv2d/separable_conv2d': 1},
        filters=10, kernel_size=3)
    out1 = conv1(self.inputs)
    self.assertEqual(1, out1.shape.as_list()[-1])

    # with custom name
    conv2 = ops.ConfigurableSeparableConv2D(
        parameterization={'sep_conv/separable_conv2d': 2}, filters=10,
        kernel_size=3, name='sep_conv')
    out2 = conv2(self.inputs)
    self.assertEqual(2, out2.shape.as_list()[-1])

  def testConfigurableDenseParameterization(self):
    # with default name
    dense1 = ops.ConfigurableDense(
        parameterization={'configurable_dense/Tensordot/MatMul': 1}, units=10)
    out1 = dense1(self.inputs)
    self.assertEqual(1, out1.shape.as_list()[-1])

    # with custom name
    dense2 = ops.ConfigurableDense(
        parameterization={'fc/Tensordot/MatMul': 2}, units=10, name='fc')
    out2 = dense2(self.inputs)
    self.assertEqual(2, out2.shape.as_list()[-1])

  def testParameterizeDuplicateNames(self):
    parameterization = {
        'conv1/ConfigurableConv2D': 1,
        'conv1_1/ConfigurableConv2D': 2,
        'configurable_conv2d/ConfigurableConv2D': 3,
        'configurable_conv2d_1/ConfigurableConv2D': 4,
    }
    conv1 = ops.ConfigurableConv2D(
        parameterization=parameterization, filters=10, kernel_size=3,
        name='conv1')
    conv1_1 = ops.ConfigurableConv2D(
        parameterization=parameterization, filters=10, kernel_size=3,
        name='conv1')
    conv_default_name = ops.ConfigurableConv2D(
        parameterization=parameterization, filters=10, kernel_size=3)
    conv_default_name_1 = ops.ConfigurableConv2D(
        parameterization=parameterization, filters=10, kernel_size=3)

    out = conv1(self.inputs)
    self.assertEqual(1, out.shape.as_list()[-1])
    out = conv1_1(self.inputs)
    self.assertEqual(2, out.shape.as_list()[-1])
    out = conv_default_name(self.inputs)
    self.assertEqual(3, out.shape.as_list()[-1])
    out = conv_default_name_1(self.inputs)
    self.assertEqual(4, out.shape.as_list()[-1])

  def testParameterizeNamesWithSlashes(self):
    conv1 = ops.ConfigurableConv2D(
        parameterization={'name/with/slashes/ConfigurableConv2D': 1,},
        filters=10, kernel_size=3, name='name/with/slashes')
    out1 = conv1(self.inputs)
    self.assertEqual(out1.shape.as_list()[-1], 1)

    conv2 = ops.ConfigurableConv2D(
        parameterization={'name//with///multislash/ConfigurableConv2D': 2},
        filters=10, kernel_size=3, name='name//with///multislash')
    out2 = conv2(self.inputs)
    self.assertEqual(out2.shape.as_list()[-1], 2)

    # When Keras calls tf.variable_scope with N trailing slashes,
    # tf.variable_scope will create a scope with N-1 trailing slashes.
    conv3 = ops.ConfigurableConv2D(
        parameterization={'name/ends/with/slashes//ConfigurableConv2D': 3},
        filters=10, kernel_size=3, name='name/ends/with/slashes///')
    out3 = conv3(self.inputs)
    self.assertEqual(3, out3.shape.as_list()[-1])

  def testStrictness(self):
    parameterization = {
        'unused_conv/Conv2D': 2,
    }
    conv_not_strict = ops.ConfigurableConv2D(
        parameterization=parameterization, is_strict=False, filters=10,
        kernel_size=3)
    conv_strict = ops.ConfigurableConv2D(
        parameterization=parameterization, is_strict=True, filters=10,
        kernel_size=3)

    # extra ops in the parameterization are ok
    out = conv_not_strict(self.inputs)
    self.assertEqual(10, out.shape.as_list()[-1])

    # when strict=True, all ops in the parameterization must be used
    with self.assertRaises(KeyError):
      out = conv_strict(self.inputs)

  def testConfigurableConv2DAlternateOpSuffixes(self):
    # ConfigurableConv2D accepts both 'Conv2D' and 'ConfigurableConv2D' as op
    # suffixes in the parameterization to be compatible with structures learned
    # using keras.layers.Conv2D or ConfigurableConv2D.
    valid_parameterization_1 = {
        'conv1/Conv2D': 1,
    }
    out1 = ops.ConfigurableConv2D(
        parameterization=valid_parameterization_1, filters=10, kernel_size=3,
        name='conv1')(self.inputs)
    self.assertEqual(out1.shape.as_list()[-1], 1)

    valid_parameterization_2 = {
        'conv2/ConfigurableConv2D': 2,
    }
    out2 = ops.ConfigurableConv2D(
        parameterization=valid_parameterization_2, filters=10, kernel_size=3,
        name='conv2')(self.inputs)
    self.assertEqual(out2.shape.as_list()[-1], 2)

    # Only one op suffix variant should exist in the parameterization.
    bad_parameterization = {
        'conv3/Conv2D': 1,
        'conv3/ConfigurableConv2D': 2
    }
    with self.assertRaises(KeyError):
      _ = ops.ConfigurableConv2D(
          parameterization=bad_parameterization, filters=10, kernel_size=3,
          name='conv3')(self.inputs)

  def testHijackingImportedLayerLib(self):
    parameterization = {'conv/Conv2D': 1}
    module = tm.layers
    _, original_layers = ops.hijack_keras_module(parameterization, module)
    out = tm.build_simple_keras_model(self.inputs)
    ops.recover_module_functions(original_layers, module)
    self.assertEqual(out.shape.as_list()[-1], 1)

  def testHijackingImportedKerasLib(self):
    parameterization = {'conv/Conv2D': 1}
    module = tm.keras.layers
    _, original_layers = ops.hijack_keras_module(parameterization, module)
    out = tm.build_simple_keras_model_from_keras_lib(self.inputs)
    ops.recover_module_functions(original_layers, module)
    self.assertEqual(out.shape.as_list()[-1], 1)

  def testHijackingLocalAliases(self):
    parameterization = {'conv/Conv2D': 1}
    module = tm
    _, original_layers = ops.hijack_keras_module(parameterization, module)
    out = tm.build_simple_keras_model_from_local_aliases(self.inputs)
    ops.recover_module_functions(original_layers, module)
    self.assertEqual(out.shape.as_list()[-1], 1)

  def testConstructedOps(self):
    parameterization = {
        'conv/Conv2D': 1,
        'sep_conv/separable_conv2d': 2,
        'dense/Tensordot/MatMul': 3,
    }
    module = tm
    constructed_ops, original_layers = ops.hijack_keras_module(
        parameterization, module)
    out = tm.build_model_with_all_configurable_types(self.inputs)
    ops.recover_module_functions(original_layers, module)
    self.assertEqual(out.shape.as_list()[-1], 3)
    self.assertDictEqual(constructed_ops, parameterization)

  def _hijack_and_recover(self, parameterization, **kwargs):
    module = tm
    _, original_layers = ops.hijack_keras_module(
        parameterization, module, **kwargs)
    out = tm.build_model_with_all_configurable_types(self.inputs)
    ops.recover_module_functions(original_layers, module)
    return out

  def testRemoveCommonPrefix_SinglePrefix(self):
    parameterization = {
        'morphnet/conv/Conv2D': 1,
        'morphnet/sep_conv/separable_conv2d': 2,
        'morphnet/dense/Tensordot/MatMul': 3,
    }
    output = self._hijack_and_recover(
        parameterization, remove_common_prefix=True)

    # 'morphnet' prefix is removed and the network is correctly parameterized
    self.assertEqual(output.shape.as_list()[-1], 3)

  def testRemoveCommonPrefix_MultiPrefix(self):
    parameterization = {
        'multiple/common/prefixes/conv/Conv2D': 1,
        'multiple/common/prefixes/sep_conv/separable_conv2d': 2,
        'multiple/common/prefixes/dense/Tensordot/MatMul': 3,
    }
    output = self._hijack_and_recover(
        parameterization, remove_common_prefix=True)

    # only the first scope is removed and the network is not parameterized
    self.assertEqual(output.shape.as_list()[-1], 10)

  def testKeepFirstChannelAlive(self):
    parameterization = {
        'conv/Conv2D': 1,
        'sep_conv/separable_conv2d': 0,  # would disconnect the network
        'dense/Tensordot/MatMul': 3,
    }
    output = self._hijack_and_recover(
        parameterization, keep_first_channel_alive=True)
    self.assertEqual(output.shape.as_list()[-1], 3)

  @parameterized.named_parameters(
      ('BatchNormalization', keras_layers.BatchNormalization, {}),
      ('Activation', keras_layers.Activation, {'activation': tf.nn.relu}),
      ('UpSampling2D', keras_layers.UpSampling2D, {}))
  def testPassThroughSingleInput(self, keras_layer_class, kwargs):
    pass_through_layer = functools.partial(
        ops.PassThroughKerasLayerWrapper,
        keras_layer_class=keras_layer_class)
    output = pass_through_layer(**kwargs)(self.inputs)
    expected = keras_layer_class(**kwargs)(self.inputs)
    with self.cached_session():
      tf.global_variables_initializer().run()
      self.assertAllClose(output.eval(), expected.eval())

    output_vanished = pass_through_layer(**kwargs)(ops.VANISHED)
    self.assertEqual(output_vanished, ops.VANISHED)

  @parameterized.named_parameters(
      ('Add', keras_layers.Add),
      ('Concatenate', keras_layers.Concatenate),
      ('Multiply', keras_layers.Multiply))
  def testPassThroughMerge(self, keras_layer_class):
    pass_through_layer = functools.partial(
        ops.PassThroughKerasLayerWrapper,
        keras_layer_class=keras_layer_class)
    output = pass_through_layer()([self.inputs, 2 * self.inputs])
    expected = keras_layer_class()([self.inputs, 2 * self.inputs])
    self.assertAllEqual(output, expected)

    some_ops_vanished = pass_through_layer()(
        [self.inputs, ops.VANISHED, 2 * self.inputs])
    expected = keras_layer_class()([self.inputs, 2 * self.inputs])
    self.assertAllEqual(some_ops_vanished, expected)

    all_ops_vanished = pass_through_layer()([ops.VANISHED, ops.VANISHED])
    self.assertEqual(all_ops_vanished, ops.VANISHED)

  def testPassThroughHijacking(self):
    parameterization = {
        'conv1/Conv2D': 0,  # followed by BatchNorm, Activation, Add and Concat
        'conv2/Conv2D': 1,
    }
    module = tm.layers
    _, original_layers = ops.hijack_keras_module(
        parameterization, module, keep_first_channel_alive=False)

    # output = Concat([branch1, branch2, branch1 + branch2]).
    # if branch1 vanishes, output should have only 2 channels.
    output, branch1, branch2 = tm.build_two_branch_model(self.inputs)

    ops.recover_module_functions(original_layers, module)

    self.assertEqual(branch1, ops.VANISHED)
    self.assertEqual(branch2.shape.as_list(), [1, 8, 8, 1])
    self.assertEqual(output.shape.as_list(), [1, 8, 8, 2])


class ConfigurableOpsTest(parameterized.TestCase, tf.test.TestCase):

  def setUp(self):
    super(ConfigurableOpsTest, self).setUp()
    tf.reset_default_graph()
    self.inputs_shape = [2, 4, 4, 3]
    self.inputs = tf.ones(self.inputs_shape, dtype=tf.float32)
    self.fc_inputs = tf.ones([3, 12])

  def testMapBinding(self):
    # TODO(e1): Clean up this file/test. Split to different tests
    function_dict = {
        'fully_connected': mock_fully_connected,
        'conv2d': mock_conv2d,
        'separable_conv2d': mock_separable_conv2d,
        'concat': mock_concat,
        'add_n': mock_add_n,
    }
    parameterization = {
        'fc/MatMul': 13,
        'conv/Conv2D': 15,
        'sep/separable_conv2d': 17
    }
    num_outputs = lambda res: res['args'][1]
    decorator = ops.ConfigurableOps(
        parameterization=parameterization, function_dict=function_dict)
    fc = decorator.fully_connected(self.fc_inputs, num_outputs=88, scope='fc')
    self.assertEqual('myfully_connected', fc['mock_name'])
    self.assertEqual(parameterization['fc/MatMul'], num_outputs(fc))

    conv2d = decorator.conv2d(
        self.inputs, num_outputs=11, kernel_size=3, scope='conv')
    self.assertEqual('myconv2d', conv2d['mock_name'])
    self.assertEqual(parameterization['conv/Conv2D'], num_outputs(conv2d))

    separable_conv2d = decorator.separable_conv2d(
        self.inputs, num_outputs=88, kernel_size=3, scope='sep')
    self.assertEqual('myseparable_conv2d', separable_conv2d['mock_name'])
    self.assertEqual(parameterization['sep/separable_conv2d'],
                     num_outputs(separable_conv2d))

    concat = decorator.concat(axis=1, values=[1, None, 2])
    self.assertEqual(concat['args'][0], [1, 2])
    self.assertEqual(concat['kwargs']['axis'], 1)
    with self.assertRaises(ValueError):
      _ = decorator.concat(inputs=[1, None, 2])

    add_n = decorator.add_n(name='add_n', inputs=[1, None, 2])
    self.assertEqual(add_n['args'][0], [1, 2])

  def testScopeAndNameKwargs(self):
    function_dict = {
        'fully_connected': mock_fully_connected,
        'conv2d': mock_conv2d,
        'separable_conv2d': mock_separable_conv2d,
        'concat': mock_concat,
        'add_n': mock_add_n,
    }
    parameterization = {
        'fc/MatMul': 13,
        'conv/Conv2D': 15,
        'sep/separable_conv2d': 17
    }
    num_outputs = lambda res: res['args'][1]
    decorator = ops.ConfigurableOps(
        parameterization=parameterization, function_dict=function_dict)

    conv2d = decorator.conv2d(
        self.inputs, num_outputs=11, kernel_size=3, scope='conv')
    self.assertEqual('myconv2d', conv2d['mock_name'])
    self.assertEqual(parameterization['conv/Conv2D'], num_outputs(conv2d))

    conv2d = decorator.conv2d(
        self.inputs, num_outputs=11, kernel_size=3, name='conv')
    self.assertEqual('myconv2d', conv2d['mock_name'])
    self.assertEqual(parameterization['conv/Conv2D'], num_outputs(conv2d))

  def testFullyConnectedOpAllKwargs(self):
    decorator = ops.ConfigurableOps(parameterization={'test/MatMul': 13})
    output = decorator.fully_connected(
        inputs=self.fc_inputs, num_outputs=88, scope='test')
    self.assertEqual(13, output.shape.as_list()[-1])

  def testFullyConnectedOpInputArgs(self):
    decorator = ops.ConfigurableOps(parameterization={'test/MatMul': 14})
    output = decorator.fully_connected(
        self.fc_inputs, num_outputs=87, scope='test')
    self.assertEqual(14, output.shape.as_list()[-1])

  def testFullyConnectedOpAllArgs(self):
    decorator = ops.ConfigurableOps(parameterization={'test/MatMul': 15})
    output = decorator.fully_connected(self.fc_inputs, 86, scope='test')
    self.assertEqual(15, output.shape.as_list()[-1])

  def testSeparableConv2dOp(self):
    parameterization = {'test/separable_conv2d': 12}
    decorator = ops.ConfigurableOps(parameterization=parameterization)
    output = decorator.separable_conv2d(
        self.inputs,
        num_outputs=88,
        kernel_size=3,
        depth_multiplier=1,
        scope='test')
    self.assertEqual(12, output.shape.as_list()[-1])

  def testComplexNet(self):
    parameterization = {'Branch0/Conv_1x1/Conv2D': 13, 'Conv3_1x1/Conv2D': 77}
    decorator = ops.ConfigurableOps(parameterization=parameterization)

    def conv2d(inputs, num_outputs, kernel_size, scope):
      return decorator.conv2d(
          inputs, num_outputs=num_outputs, kernel_size=kernel_size, scope=scope)

    net = self.inputs

    with tf.variable_scope('Branch0'):
      branch_0 = conv2d(net, 1, 1, scope='Conv_1x1')
    with tf.variable_scope('Branch1'):
      branch_1 = conv2d(net, 2, 1, scope='Conv_1x1')
      out_2 = conv2d(branch_1, 3, 3, scope='Conv_3x3')
    net = conv2d(net, 1, 1, scope='Conv3_1x1')
    output = tf.concat([net, branch_0, branch_1, out_2], -1)
    expected_output_shape = self.inputs_shape
    expected_output_shape[-1] = 95
    self.assertEqual(expected_output_shape, output.shape.as_list())
    self.assertEqual(2, decorator.constructed_ops['Branch1/Conv_1x1/Conv2D'])
    self.assertEqual(13, decorator.constructed_ops['Branch0/Conv_1x1/Conv2D'])
    self.assertEqual(77, decorator.constructed_ops['Conv3_1x1/Conv2D'])
    self.assertEqual(3, decorator.constructed_ops['Branch1/Conv_3x3/Conv2D'])

  @parameterized.named_parameters(
      ('_first_only', {
          'first/Conv2D': 3
      }, (2, 2, 2, 3), (2, 4, 4, 7)),
      ('_second_only', {
          'second/Conv2D': 13
      }, (2, 2, 2, 7), (2, 4, 4, 13)),
      ('_both', {
          'first/Conv2D': 9,
          'second/Conv2D': 5
      }, (2, 2, 2, 9), (2, 4, 4, 5)),
  )
  def testDifferentParameterization(self, parameterization,
                                    expected_first_shape, expected_conv2_shape):
    alternate_num_outputs = 7
    decorator = ops.ConfigurableOps(parameterization=parameterization)
    with arg_scope([layers.conv2d], padding='VALID'):
      first_out = decorator.conv2d(
          self.inputs,
          num_outputs=alternate_num_outputs,
          kernel_size=3,
          scope='first')
      conv2_out = decorator.conv2d(
          self.inputs,
          num_outputs=alternate_num_outputs,
          kernel_size=1,
          scope='second')
      self.assertAllEqual(expected_first_shape, first_out.shape.as_list())
      self.assertAllEqual(expected_conv2_shape, conv2_out.shape.as_list())

  def testShareParams(self):
    # Tests reuse option.
    first_outputs = 2
    alternate_num_outputs = 12
    parameterization = {'first/Conv2D': first_outputs}
    decorator = ops.ConfigurableOps(parameterization=parameterization)
    explicit = layers.conv2d(
        self.inputs, first_outputs, 3, scope='first')
    with arg_scope([layers.conv2d], reuse=True):
      decorated = decorator.conv2d(
          self.inputs,
          num_outputs=alternate_num_outputs,
          kernel_size=3,
          scope='first')
    with self.cached_session():
      tf.global_variables_initializer().run()
      # verifies that parameters are shared.
      self.assertAllClose(explicit.eval(), decorated.eval())
    conv_ops = sorted([
        op.name
        for op in tf.get_default_graph().get_operations()
        if op.type == 'Conv2D'
    ])
    self.assertAllEqual(['first/Conv2D', 'first_1/Conv2D'], conv_ops)

  def testOneDConcat(self):
    # A 1-D tensor.
    tensor1 = tf.constant([1])
    # Another 1-D tensor.
    tensor2 = tf.constant([100])
    decorator = ops.ConfigurableOps()
    result = decorator.concat([tensor1, tensor2], axis=0)
    expected_tensor = tf.constant([1, 100])

    self.assertAllEqual(expected_tensor, result)
    self.assertAllEqual(
        expected_tensor,
        decorator.concat([None, ops.VANISHED, tensor1, tensor2], axis=0))

  def testFourDConcat(self):
    # A 4-D tensor.
    tensor1 = tf.constant([[[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]],
                           [[[13, 14, 15], [16, 17, 18]],
                            [[19, 20, 21], [22, 23, 24]]]])
    # Another 4-D tensor.
    tensor2 = tf.constant([[[[25, 26, 27], [28, 29, 30]],
                            [[31, 32, 33], [34, 35, 36]]],
                           [[[37, 38, 39], [40, 41, 42]],
                            [[43, 44, 45], [46, 47, 48]]]])
    decorator = ops.ConfigurableOps()

    result = decorator.concat([tensor1, tensor2], 3)
    expected_tensor = tf.constant([[[[1, 2, 3, 25, 26, 27],
                                     [4, 5, 6, 28, 29, 30]],
                                    [[7, 8, 9, 31, 32, 33],
                                     [10, 11, 12, 34, 35, 36]]],
                                   [[[13, 14, 15, 37, 38, 39],
                                     [16, 17, 18, 40, 41, 42]],
                                    [[19, 20, 21, 43, 44, 45],
                                     [22, 23, 24, 46, 47, 48]]]])

    self.assertAllEqual(expected_tensor, result)
    self.assertAllEqual(
        expected_tensor,
        decorator.concat([None, tensor1, None, tensor2, None], 3))

  def testNoneConcat(self):
    self.assertEqual(ops.ConfigurableOps().concat([None, None], 3),
                     ops.VANISHED)

  def testAddN(self):
    t1 = tf.constant([1, 2, 3])
    t2 = tf.constant([4, 2, 3])
    decorator = ops.ConfigurableOps()
    results = decorator.add_n([t1, None, t2])
    self.assertAllEqual(results, [5, 4, 6])

  def testNoneAddN(self):
    empty = ops.VANISHED
    self.assertEqual(ops.ConfigurableOps().add_n([None, empty]), empty)

  @parameterized.named_parameters(('_first_to_zero', {
      'first/Conv2D': 0
  }), ('_conv2_to_zero', {
      'second/Conv2D': 0
  }), ('_both_conv_to_zero', {
      'first/Conv2D': 0,
      'second/Conv2D': 0
  }))
  def testTowerVanishes(self, parameterization):
    depth = self.inputs.shape.as_list()[3]
    decorator = ops.ConfigurableOps(parameterization=parameterization)

    net = decorator.conv2d(
        self.inputs, num_outputs=12, kernel_size=3, scope='first')
    net = decorator.conv2d(
        net, num_outputs=depth, kernel_size=1, scope='second')
    self.assertTrue(ops.is_vanished(net))

  def testStrict_PartialParameterizationFails(self):
    partial_parameterization = {'first/Conv2D': 3}
    default_num_outputs = 7
    decorator = ops.ConfigurableOps(
        parameterization=partial_parameterization, fallback_rule='strict')
    decorator.conv2d(
        self.inputs,
        num_outputs=default_num_outputs,
        kernel_size=3,
        scope='first')
    with self.assertRaisesRegexp(
        KeyError, 'op_name \"second/Conv2D\" not found in parameterization'):
      decorator.conv2d(
          self.inputs,
          num_outputs=default_num_outputs,
          kernel_size=1,
          scope='second')

  def testDefaultToZero(self):
    parameterization = {'first/Conv2D': 3}
    decorator = ops.ConfigurableOps(
        parameterization=parameterization, fallback_rule='zero')
    first = decorator.conv2d(
        self.inputs, num_outputs=12, kernel_size=3, scope='first')
    second = decorator.conv2d(self.inputs, 13, kernel_size=1, scope='second')
    self.assertEqual(3, first.shape.as_list()[3])
    self.assertTrue(ops.is_vanished(second))
    self.assertEqual(0, decorator.constructed_ops['second/Conv2D'])

  @parameterized.named_parameters(
      ('_string_fallback_rule', 'strict'),
      ('enum_fallback_rule', ops.FallbackRule.strict))
  def testStrict_FullParameterizationPasses(self, fallback_rule):
    full_parameterization = {'first/Conv2D': 3, 'second/Conv2D': 13}
    default_num_outputs = 7
    decorator = ops.ConfigurableOps(
        parameterization=full_parameterization, fallback_rule=fallback_rule)
    first = decorator.conv2d(
        self.inputs,
        num_outputs=default_num_outputs,
        kernel_size=3,
        scope='first')
    second = decorator.conv2d(
        self.inputs,
        num_outputs=default_num_outputs,
        kernel_size=1,
        scope='second')

    self.assertAllEqual(3, first.shape.as_list()[3])
    self.assertAllEqual(13, second.shape.as_list()[3])

  def testBadFallbackRule(self):
    with self.assertRaises(KeyError):
      ops.ConfigurableOps(fallback_rule='bad bad rule')

  def testWrongTypeFallbackRule(self):
    with self.assertRaises(ValueError):
      ops.ConfigurableOps(fallback_rule=20180207)

  def testDecoratorFromParamFile(self):
    parameterization = {'first/Conv2D': 3, 'second/Conv2D': 13}
    filename = os.path.join(FLAGS.test_tmpdir, 'parameterization_file')
    with tf.gfile.Open(filename, 'w') as f:
      json.dump(parameterization, f)

    expected_decorator = ops.ConfigurableOps(
        parameterization=parameterization,
        fallback_rule=ops.FallbackRule.pass_through)

    default_num_outputs = 7
    _ = expected_decorator.conv2d(
        self.inputs,
        num_outputs=default_num_outputs,
        kernel_size=3,
        scope='first')
    _ = expected_decorator.conv2d(
        self.inputs,
        num_outputs=default_num_outputs,
        kernel_size=1,
        scope='second')
    decorator_from_file = ops.decorator_from_parameterization_file(
        filename, fallback_rule=ops.FallbackRule.pass_through)

    self.assertEqual(expected_decorator.constructed_ops,
                     decorator_from_file._parameterization)

  def testPool(self):
    decorator = ops.ConfigurableOps()
    empty = ops.VANISHED
    pool_kwargs = dict(kernel_size=2, stride=2, padding='same', scope='pool')

    for fn_name in ['max_pool2d', 'avg_pool2d']:
      decorator_pool_fn = getattr(decorator, fn_name)
      decorator_regular_output = decorator_pool_fn(self.inputs, **pool_kwargs)
      decorator_zero_output = decorator_pool_fn(empty, **pool_kwargs)

      tf_pool_fn = getattr(layers, fn_name)
      tf_output = tf_pool_fn(self.inputs, **pool_kwargs)

      self.assertAllEqual(decorator_regular_output, tf_output)
      self.assertTrue(ops.is_vanished(decorator_zero_output))

  def testBatchNorm(self):
    decorator = ops.ConfigurableOps()
    kwargs = dict(center=False, scale=False)
    decorator_regular_output = decorator.batch_norm(self.inputs, **kwargs)
    decorator_zero_output = decorator.batch_norm(ops.VANISHED, **kwargs)
    tf_output = layers.batch_norm(self.inputs, **kwargs)

    with self.cached_session():
      tf.global_variables_initializer().run()
      self.assertAllEqual(decorator_regular_output, tf_output)
    self.assertTrue(ops.is_vanished(decorator_zero_output))

  @parameterized.named_parameters(
      ('_SlimLayers', tf_contrib.slim.conv2d, 'num_outputs', 'Conv'),
      ('_ContribLayers', tf_contrib.layers.conv2d, 'num_outputs', 'Conv'),
      ('_TfLayer', tf.layers.conv2d, 'filters', 'conv2d'))
  def testDefaultScopes_Conv(
      self, conv_fn, num_outputs_kwarg, expected_op_scope):
    inputs = tf.ones([1, 3, 3, 2])
    parameterization = {
        '{}/Conv2D'.format(expected_op_scope): 5
    }
    decorator = ops.ConfigurableOps(
        parameterization=parameterization, function_dict={'conv2d': conv_fn})
    _ = decorator.conv2d(inputs, **{num_outputs_kwarg: 8, 'kernel_size': 2})
    self.assertDictEqual(parameterization, decorator.constructed_ops)

  @parameterized.named_parameters(
      ('_SlimLayers',
       tf_contrib.slim.fully_connected, 'num_outputs', 'fully_connected'),
      ('_ContribLayers',
       tf_contrib.layers.fully_connected, 'num_outputs', 'fully_connected'),
      ('_TfLayer',
       tf.layers.dense, 'units', 'dense'))
  def testDefaultScopes_Dense(
      self, dense_fn, num_outputs_kwarg, expected_op_scope):
    inputs = tf.ones([1, 2])
    parameterization = {
        '{}/MatMul'.format(expected_op_scope): 5
    }
    decorator = ops.ConfigurableOps(
        parameterization=parameterization,
        function_dict={'fully_connected': dense_fn})
    _ = decorator.fully_connected(inputs, **{num_outputs_kwarg: 8})
    self.assertDictEqual(parameterization, decorator.constructed_ops)

  def testDefaultScopesRepeated(self):
    inputs = tf.ones([1, 3, 3, 2])
    parameterization = {
        's1/SeparableConv2d/separable_conv2d': 1,
        's1/SeparableConv2d_1/separable_conv2d': 2,
        's1/s2/SeparableConv2d/separable_conv2d': 3,
        's1/s2/SeparableConv2d_1/separable_conv2d': 4,
    }
    decorator = ops.ConfigurableOps(
        parameterization=parameterization,
        function_dict={'separable_conv2d': tf_contrib.slim.separable_conv2d})

    with tf.variable_scope('s1'):
      # first call in s1: op scope should be `s1/SeparableConv2d`
      _ = decorator.separable_conv2d(inputs, num_outputs=8, kernel_size=2)

      with tf.variable_scope('s2'):
        # first call in s2: op scope should be `s1/s2/SeparableConv2d`
        _ = decorator.separable_conv2d(inputs, num_outputs=8, kernel_size=2)

        # second call in s2: op scope should be `s1/s2/SeparableConv2d_1`
        _ = decorator.separable_conv2d(inputs, num_outputs=8, kernel_size=2)

      # second call in s1: op scope should be `s1/SeparableConv2d_1`
      _ = decorator.separable_conv2d(inputs, num_outputs=8, kernel_size=2)

    conv_op_names = [op.name for op in tf.get_default_graph().get_operations()
                     if op.name.endswith('separable_conv2d')]
    self.assertCountEqual(parameterization, conv_op_names)
    self.assertDictEqual(parameterization, decorator.constructed_ops)


class Fake(object):
  # This Class is a cheap simulation of a module.
  # TODO(e1): Replace with an actual test module.

  def __init__(self):
    self.conv2d = layers.conv2d
    self.fully_connected = layers.fully_connected
    self.separable_conv2d = layers.separable_conv2d
    self.concat = tf.concat


class FakeConv2DMissing(object):
  # This class is a cheap simulation of a module, with 'second' missing.
  # TODO(e1): Replace with an actual test module.

  def __init__(self):
    self.fully_connected = layers.fully_connected
    self.separable_conv2d = layers.separable_conv2d


class HijackerTest(parameterized.TestCase, tf.test.TestCase):

  @parameterized.named_parameters(
      ('Normal', Fake(), True, True, True),
      ('MissingConv2d', FakeConv2DMissing(), False, True, True))
  def testHijack(self, fake_module, has_conv2d, has_separable_conv2d,
                 has_fully_connected):
    # This test verifies that hijacking works with arg scope.
    # TODO(e1): Test that all is correct when hijacking a real module.
    def name_and_output_fn(name):
      # By design there is no add arg_scope here.
      def fn(*args, **kwargs):
        return (name, args[1], kwargs['scope'])

      return fn

    function_dict = {
        'fully_connected': name_and_output_fn('testing_fully_connected'),
        'conv2d': name_and_output_fn('testing_conv2d'),
        'separable_conv2d': name_and_output_fn('testing_separable_conv2d')
    }

    decorator = ops.ConfigurableOps(function_dict=function_dict)
    originals = ops.hijack_module_functions(decorator, fake_module)

    self.assertEqual('conv2d' in originals, has_conv2d)
    self.assertEqual('separable_conv2d' in originals, has_separable_conv2d)
    self.assertEqual('fully_connected' in originals, has_fully_connected)

    if has_conv2d:
      with arg_scope([fake_module.conv2d], num_outputs=2):
        out = fake_module.conv2d(
            inputs=tf.zeros([10, 3, 3, 4]), scope='test_conv2d')
      self.assertAllEqual(['testing_conv2d', 2, 'test_conv2d'], out)

    if has_fully_connected:
      with arg_scope([fake_module.fully_connected], num_outputs=3):
        out = fake_module.fully_connected(
            inputs=tf.zeros([10, 4]), scope='test_fc')
      self.assertAllEqual(['testing_fully_connected', 3, 'test_fc'], out)

    if has_separable_conv2d:
      with arg_scope([fake_module.separable_conv2d], num_outputs=4):
        out = fake_module.separable_conv2d(
            inputs=tf.zeros([10, 3, 3, 4]), scope='test_sep')
      self.assertAllEqual(['testing_separable_conv2d', 4, 'test_sep'], out)

  def testConcatHijack(self):
    decorator = ops.ConfigurableOps()
    module = Fake()
    inputs = tf.ones([2, 3, 3, 5])
    empty = ops.VANISHED
    with self.assertRaises(ValueError):
      # empty will generate an error before the hijack.
      _ = module.concat([inputs, empty], 3).shape.as_list()

    # hijacking:
    ops.hijack_module_functions(decorator, module)
    # Verifying success of hijack.
    self.assertAllEqual(
        module.concat([inputs, empty], 3).shape.as_list(), [2, 3, 3, 5])
    self.assertTrue(ops.is_vanished(module.concat([empty, empty], 3)))
    self.assertAllEqual(
        module.concat([inputs, empty, inputs], 3).shape.as_list(),
        [2, 3, 3, 10])

  def testRecover(self):
    # If this test does not work well, then it might have some bizarre effect on
    # other tests as it changes the functions in layers
    decorator = ops.ConfigurableOps()
    true_separable_conv2d = layers.separable_conv2d
    original_dict = ops.hijack_module_functions(decorator, layers)

    self.assertEqual(true_separable_conv2d, original_dict['separable_conv2d'])
    # Makes sure hijacking worked.
    self.assertNotEqual(true_separable_conv2d,
                        layers.separable_conv2d)
    # Recovers original ops
    ops.recover_module_functions(original_dict, layers)
    self.assertEqual(true_separable_conv2d, layers.separable_conv2d)

  @parameterized.named_parameters(('_v1', 'resnet_v1'), ('_v2', 'resnet_v2'))
  def testResnet(self, resnet_version):
    resnets = {'resnet_v1': (resnet_v1, 'v1'), 'resnet_v2': (resnet_v2, 'v2')}
    resnet_module = resnets[resnet_version][0]

    decorator = ops.ConfigurableOps()
    hijacked_from_layers_lib = ops.hijack_module_functions(
        decorator, resnet_module.layers_lib)
    hijacked_from_utils = ops.hijack_module_functions(decorator,
                                                      resnet_utils.layers)
    hijacked_from_module = ops.hijack_module_functions(decorator,
                                                       resnet_module.layers)
    print('hijacked_from_layers_lib', hijacked_from_layers_lib)
    print('hijacked_from_utils', hijacked_from_utils)
    print('hijacked_from_module', hijacked_from_module)
    inputs = tf.ones([3, 16, 16, 5])
    _ = resnet_module.bottleneck(
        inputs, depth=64, depth_bottleneck=16, stride=1)

    self.assertLen(decorator.constructed_ops, 4)

    base_name = 'bottleneck_' + resnets[resnet_version][1]
    expected_decorated_ops = sorted([
        base_name + '/conv1/Conv2D',
        base_name + '/conv2/Conv2D',
        base_name + '/conv3/Conv2D',
        base_name + '/shortcut/Conv2D',
    ])

    self.assertAllEqual(expected_decorated_ops,
                        sorted(decorator.constructed_ops.keys()))


if __name__ == '__main__':
  tf.test.main()
