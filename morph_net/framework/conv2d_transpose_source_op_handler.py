"""OpHandler implementation for regularizer source conv2d_transpose ops.

OpHandler for Conv2DBackpropInput (conv2d_transpose) source ops that use
group lasso regularization.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from morph_net.framework import group_lasso_base_op_handler


class Conv2DTransposeSourceOpHandler(
    group_lasso_base_op_handler.GroupLassoBaseSourceOpHandler):
  """OpHandler for Conv2DBackpropInput (conv2d_transpose) source operations."""

  def _reduce_dims(self, op):
    del op  # Unused.
    # Reduction dimensions for Group Lasso.
    return (0, 1, 3)
