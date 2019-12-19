"""Tests for morph_net.framework.matmul_source_op_handler."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from absl.testing import parameterized
from morph_net.framework import matmul_source_op_handler
from morph_net.framework import op_regularizer_manager as orm
import tensorflow.compat.v1 as tf


class MatmulSourceOpHandlerTest(parameterized.TestCase, tf.test.TestCase):

  @parameterized.named_parameters(('_slice_all', 3), ('_part_slice', 2))
  def testMatMul2D(self, size):
    inputs = tf.zeros((13, 2))
    handler = matmul_source_op_handler.MatMulSourceOpHandler(0.1)

    kernel = tf.constant([[1, 2, 3], [4, 5, 6]], dtype=tf.float32)
    x = tf.matmul(inputs, kernel, transpose_b=False, name='MatMul')
    op_slice = orm.OpSlice(x.op, orm.Slice(0, size))

    transpose_kernel = tf.constant([[1, 4], [2, 5], [3, 6]], dtype=tf.float32)
    x_other = tf.matmul(
        inputs,
        transpose_kernel,
        transpose_b=True,
        name='MatMulTransposedKernel')
    op_slice_other = orm.OpSlice(x_other.op, orm.Slice(0, size))

    self.assertAllClose(
        handler.create_regularizer(op_slice).regularization_vector,
        handler.create_regularizer(op_slice_other).regularization_vector)


if __name__ == '__main__':
  tf.test.main()
