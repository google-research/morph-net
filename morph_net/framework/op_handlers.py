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


def get_gamma_op_handler_dict():
  return collections.defaultdict(
      grouping_op_handler.GroupingOpHandler, {
          'Conv2D':
              output_non_passthrough_op_handler.OutputNonPassthroughOpHandler(),
          'ConcatV2':
              concat_op_handler.ConcatOpHandler(),
          'DepthToSpace':
              output_non_passthrough_op_handler.OutputNonPassthroughOpHandler(),
          'DepthwiseConv2dNative':
              depthwise_convolution_op_handler.DepthwiseConvolutionOpHandler(),
          'MatMul':
              output_non_passthrough_op_handler.OutputNonPassthroughOpHandler(),
          'TensorArrayGatherV3':
              leaf_op_handler.LeafOpHandler(),
          'RandomUniform':
              leaf_op_handler.LeafOpHandler(),
          'Reshape':
              leaf_op_handler.LeafOpHandler(),
          'Transpose':
              output_non_passthrough_op_handler.OutputNonPassthroughOpHandler(),
          'ExpandDims':
              output_non_passthrough_op_handler.OutputNonPassthroughOpHandler(),
      })


def get_group_lasso_op_handler_dict():
  return collections.defaultdict(
      grouping_op_handler.GroupingOpHandler, {
          'ConcatV2':
              concat_op_handler.ConcatOpHandler(),
          'DepthToSpace':
              output_non_passthrough_op_handler.OutputNonPassthroughOpHandler(),
          'DepthwiseConv2dNative':
              depthwise_convolution_op_handler.DepthwiseConvolutionOpHandler(),
          'RandomUniform':
              leaf_op_handler.LeafOpHandler(),
          'Reshape':
              leaf_op_handler.LeafOpHandler(),
          'Shape':
              leaf_op_handler.LeafOpHandler(),
          'TensorArrayGatherV3':
              leaf_op_handler.LeafOpHandler(),
          'Transpose':
              output_non_passthrough_op_handler.OutputNonPassthroughOpHandler(),
          'StridedSlice':
              leaf_op_handler.LeafOpHandler(),
      })
