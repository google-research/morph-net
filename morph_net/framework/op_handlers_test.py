"""Tests for framework.op_handlers."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from morph_net.framework import op_handlers

import tensorflow.compat.v1 as tf


class OpHandlersTest(tf.test.TestCase):

  def test_dict_logic(self):
    gamma_dict = op_handlers.get_gamma_op_handler_dict()
    self.assertIn('Conv2D', gamma_dict)
    self.assertIn('MatMul', gamma_dict)
    group_lasso_dict = op_handlers.get_group_lasso_op_handler_dict()
    self.assertNotIn('Conv2D', group_lasso_dict)
    self.assertNotIn('MatMul', group_lasso_dict)
    for op in group_lasso_dict:
      self.assertIn(op, gamma_dict)


if __name__ == '__main__':
  tf.test.main()
