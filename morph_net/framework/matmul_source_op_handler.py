"""OpHandler implementation for MatMul ops that are regularizer sources.

OpHandler for MatMul ops source ops that use group lasso regularization.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from morph_net.framework import group_lasso_base_op_handler
import tensorflow.compat.v1 as tf


class MatMulSourceOpHandler(
    group_lasso_base_op_handler.GroupLassoBaseSourceOpHandler):
  """OpHandler for MatMul source operations."""
  # TODO(a1): Sometimes this should not be skipped (b/123778611)

  def _reduce_dims(self, op):
    tf.logging.info('MatMulSourceOpHandler: found kernel = %s for op %s',
                    op.inputs[0], op.type)
    # Reduction dimensions for Group Lasso.
    try:
      if op.get_attr('transpose_b'):
        return (1,)
    except ValueError:
      tf.logging.warning(
          'MatMulSourceOpHandler: used on op.type %s with no transpose_b attr',
          op.type)
    return (0,)
