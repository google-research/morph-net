"""Tests for tpu_util."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from morph_net.framework import tpu_util
import numpy as np
import tensorflow as tf


class TpuUtilTest(tf.test.TestCase):

  def test_variable_store(self):
    stored_value = np.array([1, 1])
    g = tf.Graph()
    with g.as_default():
      with tf.variable_scope('some_scope'):
        c = tf.constant(stored_value)
        result = tpu_util.variable_store.replace_with_variable(c)
      with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        self.assertAllEqual(sess.run(result), stored_value)

        # Now look up the tensor in a different scope.
        with tf.variable_scope('a_different_scope'):
          var_value = tpu_util.variable_store.lookup_tensor(c)
          self.assertAllEqual(sess.run(var_value), stored_value)

    # Check that multiple graphs don't get clobbered.
    g1 = tf.Graph()
    with g1.as_default():
      with tf.variable_scope('some_scope'):
        c = tf.constant(stored_value * 2)
        result = tpu_util.variable_store.replace_with_variable(c)
      with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        self.assertAllEqual(sess.run(result), stored_value * 2)

        # Now look up the tensor in a different scope.
        with tf.variable_scope('a_different_scope'):
          var_value = tpu_util.variable_store.lookup_tensor(c)
          self.assertAllEqual(sess.run(var_value), stored_value * 2)

  def test_variable_store_tf_while(self):
    n_iterations = 10
    original_value = np.array([2, 2])
    expected_value = original_value**(n_iterations + 1)
    endpoints = {}
    g = tf.Graph()
    with g.as_default():
      with tf.variable_scope('some_scope'):

        def condition(i, unused_output):
          return tf.less(i, n_iterations)

        def body(i, out):
          endpoints['out'] = out + out
          out = tpu_util.variable_store.replace_with_variable(endpoints['out'])
          return (i + 1, out)

        result = tf.while_loop(condition, body, [0, original_value])

      with tf.Session(graph=g) as sess:
        sess.run(tf.global_variables_initializer())

        actual_iterations, actual_value = sess.run(result)
        self.assertEqual(actual_iterations, n_iterations)
        self.assertAllEqual(actual_value, expected_value)

        # Now look up the tensor in a different scope, outside the loop.
        with tf.variable_scope('a_different_scope'):
          var_value = tpu_util.variable_store.lookup_tensor(endpoints['out'])
          self.assertAllEqual(sess.run(var_value), expected_value)


if __name__ == '__main__':
  tf.test.main()
