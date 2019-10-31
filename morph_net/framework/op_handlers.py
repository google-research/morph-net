"""Op Handlers for use with different NetworkRegularizers."""

from __future__ import absolute_import
from __future__ import division
# [internal] enable type annotations
from __future__ import print_function

import collections

from morph_net.framework import concat_op_handler
from morph_net.framework import depthwise_convolution_op_handler
from morph_net.framework import grouping_op_handler
from morph_net.framework import leaf_op_handler
from morph_net.framework import output_non_passthrough_op_handler

RESIZE_OP_NAMES = [
    'ResizeArea', 'ResizeBicubic', 'ResizeBilinear', 'ResizeNearestNeighbor'
]


def _get_base_op_hander_dicts():
  """Returns the base op_hander_dict for all regularizers."""
  base_dict = collections.defaultdict(
      grouping_op_handler.GroupingOpHandler, {
          'ConcatV2':
              concat_op_handler.ConcatOpHandler(),
          'DepthToSpace':
              output_non_passthrough_op_handler.OutputNonPassthroughOpHandler(),
          'DepthwiseConv2dNative':
              depthwise_convolution_op_handler.DepthwiseConvolutionOpHandler(),
          'ExpandDims':
              output_non_passthrough_op_handler.OutputNonPassthroughOpHandler(),
          'RandomUniform':
              leaf_op_handler.LeafOpHandler(),
          'Reshape':
              leaf_op_handler.LeafOpHandler(),
          'Shape':
              leaf_op_handler.LeafOpHandler(),
          'SpaceToDepth':
              leaf_op_handler.LeafOpHandler(),
          'StridedSlice':
              leaf_op_handler.LeafOpHandler(),
          'TensorArrayGatherV3':
              leaf_op_handler.LeafOpHandler(),
          'Transpose':
              output_non_passthrough_op_handler.OutputNonPassthroughOpHandler(),
      })
  for resize_method in RESIZE_OP_NAMES:
    # Resize* ops, second input might be a tensor which will result in an error.
    base_dict[resize_method] = grouping_op_handler.GroupingOpHandler([0])
  return base_dict


def get_gamma_op_handler_dict():
  """Returns the base op_hander_dict for gamma based regularizers."""
  op_handler_dict = _get_base_op_hander_dicts()
  op_handler_dict.update({
      'Conv3D':
          output_non_passthrough_op_handler.OutputNonPassthroughOpHandler(),
      'Conv2D':
          output_non_passthrough_op_handler.OutputNonPassthroughOpHandler(),
      'MatMul':
          output_non_passthrough_op_handler.OutputNonPassthroughOpHandler(),
      'Conv2DBackpropInput':
          output_non_passthrough_op_handler.OutputNonPassthroughOpHandler(),
  })
  return op_handler_dict


def get_group_lasso_op_handler_dict():
  """Returns the base op_hander_dict for group-lasso based regularizers."""
  return _get_base_op_hander_dicts()
