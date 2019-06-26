"""Tests for morph_net.tools.configurable_ops."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import json
import os

from absl import flags

from absl.testing import parameterized

from morph_net.tools import configurable_ops as ops

import tensorflow as tf

from tensorflow.contrib.slim.nets import resnet_utils
from tensorflow.contrib.slim.nets import resnet_v1
from tensorflow.contrib.slim.nets import resnet_v2

resnet_utils = tf.contrib.slim.nets.resnet_utils
resnet_v1 = tf.contrib.slim.nets.resnet_v1
resnet_v2 = tf.contrib.slim.nets.resnet_v2

arg_scope = tf.contrib.framework.arg_scope
add_arg_scope = tf.contrib.framework.add_arg_scope

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
        depth_multiplier=1.0,
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
    with arg_scope([tf.contrib.layers.conv2d], padding='VALID'):
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
    explicit = tf.contrib.layers.conv2d(
        self.inputs, first_outputs, 3, scope='first')
    with arg_scope([tf.contrib.layers.conv2d], reuse=True):
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

      tf_pool_fn = getattr(tf.contrib.layers, fn_name)
      tf_output = tf_pool_fn(self.inputs, **pool_kwargs)

      self.assertAllEqual(decorator_regular_output, tf_output)
      self.assertTrue(ops.is_vanished(decorator_zero_output))

  def testBatchNorm(self):
    decorator = ops.ConfigurableOps()
    kwargs = dict(center=False, scale=False)
    decorator_regular_output = decorator.batch_norm(self.inputs, **kwargs)
    decorator_zero_output = decorator.batch_norm(ops.VANISHED, **kwargs)
    tf_output = tf.contrib.layers.batch_norm(self.inputs, **kwargs)

    self.assertAllEqual(decorator_regular_output, tf_output)
    self.assertTrue(ops.is_vanished(decorator_zero_output))


class Fake(object):
  # This Class is a cheap simulation of a module.
  # TODO(e1): Replace with an actual test module.

  def __init__(self):
    self.conv2d = tf.contrib.layers.conv2d
    self.fully_connected = tf.contrib.layers.fully_connected
    self.separable_conv2d = tf.contrib.layers.separable_conv2d
    self.concat = tf.concat


class FakeConv2DMissing(object):
  # This class is a cheap simulation of a module, with 'second' missing.
  # TODO(e1): Replace with an actual test module.

  def __init__(self):
    self.fully_connected = tf.contrib.layers.fully_connected
    self.separable_conv2d = tf.contrib.layers.separable_conv2d


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
    # other tests as it changes the functions in tf.contrib.layers
    decorator = ops.ConfigurableOps()
    true_separable_conv2d = tf.contrib.layers.separable_conv2d
    original_dict = ops.hijack_module_functions(decorator, tf.contrib.layers)

    self.assertEqual(true_separable_conv2d, original_dict['separable_conv2d'])
    # Makes sure hijacking worked.
    self.assertNotEqual(true_separable_conv2d,
                        tf.contrib.layers.separable_conv2d)
    # Recovers original ops
    ops.recover_module_functions(original_dict, tf.contrib.layers)
    self.assertEqual(true_separable_conv2d, tf.contrib.layers.separable_conv2d)

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
