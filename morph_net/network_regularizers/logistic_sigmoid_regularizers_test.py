"""Tests for logistic_sigmoid_flop_regularizer."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl.testing import parameterized

from morph_net.network_regularizers import flop_regularizer as fr
from morph_net.network_regularizers import model_size_regularizer as msr
from morph_net.network_regularizers import resource_function
from morph_net.probabilistic_gating import activation_gating

import tensorflow.compat.v1 as tf


LOGIT_ON_VAL = 10.0
LOGIT_OFF_VAL = -10.0


class LogisticSigmoidFlopRegularizerTest(
    parameterized.TestCase, tf.test.TestCase):

  def BuildConvWithGating(
      self, inputs, num_outputs, kernel_size, stride=1, scope=None):
    with tf.variable_scope(scope, 'conv_gating', [inputs]):
      c = tf.layers.conv2d(
          inputs, num_outputs,
          kernel_size, padding='SAME', strides=stride, name='conv')
      gated_c = activation_gating.logistic_sigmoid_gating(
          c, axis=3, is_training=True, scope='gate')
    return gated_c

  def BuildAndInitModel(self,
                        regularizer_cls=fr.LogisticSigmoidFlopsRegularizer):
    self.regularizer_cls = regularizer_cls
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
    # We add gating OPs after every conv.

    # op.name: 'Const'
    image = tf.constant(0.0, shape=[1, 17, 19, 3])
    # op.name: 'conv1/Conv2D'
    self.conv1 = self.BuildConvWithGating(
        image, 13, [7, 5], scope='conv1')
    self.conv2 = self.BuildConvWithGating(
        image, 23, [1, 1], scope='conv2')
    self.concat = tf.concat([self.conv1, self.conv2], 3)
    self.conv3 = self.BuildConvWithGating(
        self.concat, 29, [3, 3], stride=2, scope='conv3')
    self.conv4 = self.BuildConvWithGating(
        self.concat, 31, [1, 1], stride=1, scope='conv4')
    self.name_to_var = {v.op.name: v for v in tf.global_variables()}

  def AddRegularizer(self, input_boundary=None):
    self.ls_reg = self.regularizer_cls(
        [self.conv3.op, self.conv4.op],
        input_boundary=input_boundary)

  def GetConv(self, name):
    return tf.get_default_graph().get_operation_by_name(name + '/Conv2D')

  def GetCost(self, conv):
    with self.cached_session():
      return self.ls_reg.get_cost(conv).eval()

  def GetSourceOps(self):
    op_regularizer_manager = self.ls_reg.op_regularizer_manager
    return [
        op.name
        for op in op_regularizer_manager.ops
        if op_regularizer_manager.is_source_op(op)
    ]

  def Init(self):
    tf.global_variables_initializer().run()
    mask_logits1 = self.name_to_var['conv1/gate/mask_logits']
    mask_logits1.assign([LOGIT_ON_VAL] * 7 + [LOGIT_OFF_VAL] * 6).eval()
    mask_logits2 = self.name_to_var['conv2/gate/mask_logits']
    mask_logits2.assign([LOGIT_ON_VAL] * 11 + [LOGIT_OFF_VAL] * 12).eval()
    mask_logits3 = self.name_to_var['conv3/gate/mask_logits']
    mask_logits3.assign([LOGIT_ON_VAL] * 10 + [LOGIT_OFF_VAL] * 19).eval()
    mask_logits4 = self.name_to_var['conv4/gate/mask_logits']
    mask_logits4.assign([LOGIT_ON_VAL] * 17 + [LOGIT_OFF_VAL] * 14).eval()

  @parameterized.named_parameters(
      ('_LogisticSigmoidFLOPs', fr.LogisticSigmoidFlopsRegularizer,
       resource_function.flop_coeff),
      ('_LogisticSigmoidModelSize', msr.LogisticSigmoidModelSizeRegularizer,
       resource_function.num_weights_coeff))
  def testCost(self, regularizer_cls, cost_coeff):
    self.BuildAndInitModel(regularizer_cls)
    self.AddRegularizer(input_boundary=None)

    # Conv1 has 7 on channels, and 3 inputs (from the image).
    conv = self.GetConv('conv1/conv')
    self.assertEqual(cost_coeff(conv) * 7 * 3, self.GetCost([conv]))

    # Conv2 has 11 on channels, and 3 inputs (from the image).
    conv = self.GetConv('conv2/conv')
    self.assertEqual(cost_coeff(conv) * 11 * 3, self.GetCost([conv]))

    # Conv3 has 10 on channels, and 7 + 11 inputs from conv1 and conv2.
    conv = self.GetConv('conv3/conv')
    self.assertEqual(cost_coeff(conv) * 10 * 18, self.GetCost([conv]))

    # Conv4 has 17 on channels, and 7 + 11 inputs from conv1 and conv2.
    conv = self.GetConv('conv4/conv')
    self.assertEqual(cost_coeff(conv) * 17 * 18, self.GetCost([conv]))

    # Test that passing a list of convs sums their contributions:
    convs = [self.GetConv('conv3/conv'), self.GetConv('conv4/conv')]
    self.assertEqual(
        self.GetCost(convs[:1]) + self.GetCost(convs[1:]), self.GetCost(convs))

  def testInputBoundaryNone(self):
    self.BuildAndInitModel()
    self.AddRegularizer(input_boundary=None)
    self.assertCountEqual(self.GetSourceOps(), [
        'conv1/gate/LogisticSigmoidGating', 'conv2/gate/LogisticSigmoidGating',
        'conv3/gate/LogisticSigmoidGating', 'conv4/gate/LogisticSigmoidGating'
    ])

  def testInputBoundaryConv3(self):
    # Only block one path, can still reach all other convolutions.
    self.BuildAndInitModel()
    self.AddRegularizer(input_boundary=[self.conv3.op])
    self.assertCountEqual(self.GetSourceOps(), [
        'conv1/gate/LogisticSigmoidGating', 'conv2/gate/LogisticSigmoidGating',
        'conv4/gate/LogisticSigmoidGating'
    ])

  def testInputBoundaryConv3And4(self):
    # Block both paths, can no longer reach Concat and earlier convolutions.
    self.BuildAndInitModel()
    self.AddRegularizer(input_boundary=[self.conv3.op, self.conv4.op])
    self.assertCountEqual(self.GetSourceOps(), [])


if __name__ == '__main__':
  tf.test.main()
