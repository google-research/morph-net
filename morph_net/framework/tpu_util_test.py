# Lint as: python3
"""Tests for morph_net.framework.tpu_util."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl.testing import parameterized
from morph_net.framework import tpu_util
import tensorflow.compat.v1 as tf
from tensorflow.contrib import slim


class TpuUtilTest(parameterized.TestCase, tf.test.TestCase):

  def build_model(self):
    return slim.conv2d(
        tf.zeros([64, 10, 10, 3]),
        32, [5, 5],
        scope='conv1',
        trainable=True,
        normalizer_fn=slim.batch_norm,
        normalizer_params={
            'scale': True,
            'fused': True
        })

  def get_gamma(self, activation_tensor):
    # The input to the activation tensor is a FusedBatchNorm tensor; its name
    # should be conv1/BatchNorm/FusedBatchNormV3:0, but the version may change.
    batch_norm_tensor, = activation_tensor.op.inputs
    assert 'FusedBatchNorm' in batch_norm_tensor.name

    # gamma tensor is used by MorphNet regularizer. It is:
    # conv1/BatchNorm/ReadVariableOp:0 for ResourceVariable,
    # conv1/BatchNorm/gamma/read:0 for VariableV2.
    (unused_input_tensor, gamma_tensor, unused_beta_tensor,
     unused_population_mean_tensor,
     unused_population_variance_tensor) = batch_norm_tensor.op.inputs

    # gamma_source is the op that drives the value of the gamma:
    # 'conv1/BatchNorm/gamma' of type VarHandleOp for ResourceVariable,
    # 'conv1/BatchNorm/gamma' of type VariableV2 for VariableV2.
    gamma_source_op = gamma_tensor.op.inputs[0].op

    return gamma_tensor, gamma_source_op

  def test_variable_v2(self):
    with tf.variable_scope('', use_resource=False):
      relu = self.build_model()
    gamma_tensor, _ = self.get_gamma(relu)
    # Check that maybe_convert_to_variable ignores VariableV2 (i.e., is no op).
    self.assertEqual(
        tpu_util.maybe_convert_to_variable(gamma_tensor), gamma_tensor)

  def test_resource_variable(self):
    with tf.variable_scope('', use_resource=True):
      relu = self.build_model()
    gamma_tensor, gamma_source_op = self.get_gamma(relu)
    variable = tpu_util.maybe_convert_to_variable(gamma_tensor)

    # First assert that we didn't return the original tensor
    self.assertNotEqual(variable, gamma_tensor)

    # Now check that the variable created by maybe_convert_to_variable is
    # driven by the same op as the tensor passed as input.
    self.assertEqual(variable.op, gamma_source_op)

    # If input tensor is separated from a variable by an extra hop of Identity,
    # maybe_read_variable pretends the Identity op isn't there.
    identity_tensor = tf.identity(gamma_tensor)
    self.assertEqual(
        tpu_util.maybe_convert_to_variable(identity_tensor), variable)

  def test_noop(self):
    with tf.variable_scope('', use_resource=True):
      relu = self.build_model()
    # Check tensors that are not variable reads are ignored.
    self.assertEqual(tpu_util.maybe_convert_to_variable(relu), relu)

  def test_write_to_variable(self):
    foo = tf.constant(0., name='foo')
    tpu_util.write_to_variable(foo)

    with self.assertRaises(ValueError):
      tpu_util.write_to_variable(foo, fail_if_exists=True)

    # Variable sharing behavior should be dictated by `fail_if_exists` which
    # overrides the effect of outer scopes.
    with tf.variable_scope('', reuse=True):
      # should fail to return existing variable even though reuse=True
      with self.assertRaises(ValueError):
        tpu_util.write_to_variable(foo, fail_if_exists=True)

    with tf.variable_scope('', reuse=False):
      # should return existing variable even though reuse=False
      foo_copy = tpu_util.write_to_variable(foo, fail_if_exists=False)
      self.assertEqual(tpu_util.var_store[foo], tpu_util.var_store[foo_copy])
      self.assertLen(set(tpu_util.var_store.values()), 1)

    with tf.variable_scope('', reuse=True):
      # should create new variable even though reuse=True
      bar = tf.constant(0., name='bar')
      tpu_util.write_to_variable(bar)
      self.assertLen(set(tpu_util.var_store.values()), 2)

  def test_write_to_variable_check_inputs(self):
    variable = tf.get_variable('x', shape=[1], dtype=tf.float32)
    with self.assertRaises(ValueError):
      tpu_util.write_to_variable(variable)


if __name__ == '__main__':
  tf.test.main()
