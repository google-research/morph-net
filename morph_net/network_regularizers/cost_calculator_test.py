"""Tests for network_regularizers.cost_calculator."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
from absl.testing import parameterized
from morph_net.framework import batch_norm_source_op_handler
from morph_net.framework import concat_op_handler
from morph_net.framework import grouping_op_handler
from morph_net.framework import op_regularizer_manager as orm
from morph_net.framework import output_non_passthrough_op_handler
from morph_net.network_regularizers import cost_calculator as cc
from morph_net.network_regularizers import resource_function
from morph_net.testing import add_concat_model_stub
import tensorflow as tf

arg_scope = tf.contrib.framework.arg_scope
layers = tf.contrib.layers


class CostCalculatorTest(parameterized.TestCase, tf.test.TestCase):

  def _batch_norm_scope(self):
    params = {
        'trainable': True,
        'normalizer_fn': layers.batch_norm,
        'normalizer_params': {
            'scale': True
        }
    }

    with arg_scope([layers.conv2d], **params) as sc:
      return sc

  def testImageIsNotZerothOutputOfOp(self):
    # Throughout the framework, we assume that the 0th output of each op is the
    # only one of interest. One exception that often happens is when the input
    # image comes from a queue or from a staging op. Then the image is one of
    # the outputs of the dequeue (or staging) op, not necessarily the 0th one.
    # Here we test that the BilinearNetworkRegularizer deals correctly with this
    # case.

    # Create an input op where the image is output number 1, not 0.
    # TODO(g1) Move this mechanism to add_concat_model_stub, possibly using
    # tf.split to produce an op where the image is not the 0th output image
    # (instead of FIFOQueue).
    image = add_concat_model_stub.image_stub()
    non_image_tensor = tf.zeros(shape=(41,))
    queue = tf.FIFOQueue(
        capacity=1,
        dtypes=(tf.float32,) * 2,
        shapes=(non_image_tensor.shape, image.shape))

    # Pass the image (output[1]) to the network.
    with arg_scope(self._batch_norm_scope()):
      output_op = add_concat_model_stub.build_model(queue.dequeue()[1])

    # Create OpHandler dict for test.
    op_handler_dict = collections.defaultdict(
        grouping_op_handler.GroupingOpHandler)
    op_handler_dict.update({
        'FusedBatchNormV3':
            batch_norm_source_op_handler.BatchNormSourceOpHandler(0.1),
        'Conv2D':
            output_non_passthrough_op_handler.OutputNonPassthroughOpHandler(),
        'ConcatV2':
            concat_op_handler.ConcatOpHandler(),
    })

    # Create OpRegularizerManager and NetworkRegularizer for test.
    manager = orm.OpRegularizerManager([output_op], op_handler_dict)
    calculator = cc.CostCalculator(manager, resource_function.flop_function)

    # Calculate expected FLOP cost.
    expected_alive_conv1 = sum(add_concat_model_stub.expected_alive()['conv1'])
    conv1_op = tf.get_default_graph().get_operation_by_name('conv1/Conv2D')
    conv1_coeff = resource_function.flop_coeff(conv1_op)
    num_channels = 3
    expected_cost = conv1_coeff * num_channels * expected_alive_conv1

    with self.session():
      tf.global_variables_initializer().run()
      # Set gamma values to replicate aliveness in add_concat_model_stub.
      name_to_var = {v.op.name: v for v in tf.global_variables()}
      gamma1 = name_to_var['conv1/BatchNorm/gamma']
      gamma1.assign([0, 1, 1, 0, 1, 0, 1]).eval()
      gamma4 = name_to_var['conv4/BatchNorm/gamma']
      gamma4.assign([0, 1, 0, 1, 1, 0, 1, 0, 0, 1, 0, 0]).eval()

      queue.enqueue((non_image_tensor, image)).run()
      self.assertEqual(expected_cost,
                       calculator.get_cost([conv1_op]).eval())
      # for 0/1 assigments cost and reg_term are equal:
      self.assertEqual(expected_cost,
                       calculator.get_regularization_term([conv1_op]).eval())

  @parameterized.named_parameters(
      ('_conv2d', 4, lambda x: layers.conv2d(x, 16, 3), 'Conv2D'),
      ('_convt', 4, lambda x: layers.conv2d_transpose(x, 16, 3),
       'conv2d_transpose'),
      ('_conv2s', 4, lambda x: layers.separable_conv2d(x, None, 3),
       'depthwise'),
      ('_conv3d', 5, lambda x: layers.conv3d(x, 16, 3), 'Conv3D'))
  def test_get_input_activation2(self, rank, fn, op_name):
    g = tf.get_default_graph()
    inputs = tf.zeros([6] * rank)
    with arg_scope([
        layers.conv2d, layers.conv2d_transpose, layers.separable_conv2d,
        layers.conv3d
    ],
                   scope='test_layer'):
      _ = fn(inputs)
    for op in g.get_operations():
      print(op.name)
    self.assertEqual(
        inputs,
        cc.get_input_activation(
            g.get_operation_by_name('test_layer/' + op_name)))


if __name__ == '__main__':
  tf.test.main()
