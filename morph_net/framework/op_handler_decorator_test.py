"""Tests for morph_net.framework.op_regularizer_decorator."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from morph_net.framework import conv_source_op_handler
from morph_net.framework import generic_regularizers
from morph_net.framework import op_handler_decorator
from morph_net.framework import op_regularizer_manager as orm
import numpy as np
import tensorflow.compat.v1 as tf


class DummyDecorator(generic_regularizers.OpRegularizer):
  """A dummy decorator that multiply the regularization vector by 0.5.

  """

  def __init__(self, regularizer_object):
    """Creates an instance.

    Accept an OpRegularizer that is decorated by this class.

    Args:
      regularizer_object: OpRegularizer to decorate.
    """

    self._regularization_vector = regularizer_object.regularization_vector * 0.5
    self._alive_vector = regularizer_object.alive_vector

  @property
  def regularization_vector(self):
    return self._regularization_vector

  @property
  def alive_vector(self):
    return self._alive_vector


class OpHandlerDecoratorTest(tf.test.TestCase):
  """Test class for OpHandlerDecorator."""

  def testOpHandlerDecorator(self):
    image = tf.constant(0.0, shape=[1, 17, 19, 3])
    kernel = tf.ones([5, 5, 3, 3])

    output = tf.nn.conv2d(image, kernel, strides=[1, 1, 1, 1], padding='SAME')

    decorated_op_handler = op_handler_decorator.OpHandlerDecorator(
        conv_source_op_handler.ConvSourceOpHandler(1e-3, 0), DummyDecorator)
    op_slice = orm.OpSlice(output.op, orm.Slice(0, 3))
    regularizer = decorated_op_handler.create_regularizer(op_slice)

    self.assertAllClose(0.5 * np.ones(3), regularizer.regularization_vector)
    self.assertAllClose(np.ones(3), regularizer.alive_vector)


if __name__ == '__main__':
  tf.test.main()
