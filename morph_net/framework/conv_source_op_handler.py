"""OpHandler implementation for conv2d ops that are regularizer sources.

This OpHandler is used when a conv2d op does not have an associated batch norm
gamma. When a path in the model graph has a group lasso based source op which is
followed downstream by a batch norm source, the batch norm takes precedence.
For example: conv2d -> relu -> {bn, foo}.
Relu is passthrough so conv2d is grouped with the last op.
If the last op is batch norm, then the source op for the group is the batch
norm gamma. If not, the source op for the group is the conv2d group lasso.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from morph_net.framework import group_lasso_base_op_handler


class ConvSourceOpHandler(
    group_lasso_base_op_handler.GroupLassoBaseSourceOpHandler):
  """OpHandler implementation for Conv2D, Conv3D source operations."""

  def _reduce_dims(self, op):
    # Reduction dimensions for Group Lasso.
    if op.type == 'Conv2D':
      return (0, 1, 2)
    elif op.type == 'Conv3D':
      return (0, 1, 2, 3)
    else:
      raise ValueError('Unsupported op type %s' % op.type)

Conv2DSourceOpHandler = ConvSourceOpHandler
