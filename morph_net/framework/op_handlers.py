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


def _get_base_op_hander_dicts():
  return collections.defaultdict(
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


def get_gamma_op_handler_dict():
  op_handler_dict = _get_base_op_hander_dicts()
  op_handler_dict.update({
      'Conv2D':
          output_non_passthrough_op_handler.OutputNonPassthroughOpHandler(),
      'MatMul':
          output_non_passthrough_op_handler.OutputNonPassthroughOpHandler(),
  })
  return op_handler_dict


def get_group_lasso_op_handler_dict():
  return _get_base_op_hander_dicts()
