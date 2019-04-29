"""A L1 loss on BatchNorm gamma that targets the number of activations."""

from __future__ import absolute_import
from __future__ import division
# [internal] enable type annotations
from __future__ import print_function

import collections
from morph_net.framework import batch_norm_source_op_handler
from morph_net.framework import concat_op_handler
from morph_net.framework import conv2d_source_op_handler
from morph_net.framework import conv2d_transpose_source_op_handler
from morph_net.framework import depthwise_convolution_op_handler
from morph_net.framework import generic_regularizers
from morph_net.framework import grouping_op_handler
from morph_net.framework import leaf_op_handler
from morph_net.framework import matmul_source_op_handler
from morph_net.framework import op_handler_decorator
from morph_net.framework import op_regularizer_manager as orm
from morph_net.framework import output_non_passthrough_op_handler
from morph_net.network_regularizers import cost_calculator
from morph_net.network_regularizers import resource_function
from typing import Type


class GammaActivationRegularizer(generic_regularizers.NetworkRegularizer):
  """A NetworkRegularizer that targets activation count using Gamma L1."""

  def __init__(
      self,
      ops,
      gamma_threshold,
      regularizer_decorator: Type[generic_regularizers.OpRegularizer] = None,
      decorator_parameters=None,
      force_group=None,
      regularizer_blacklist=None):
    """Creates a GammaActivationRegularizer object.

    Args:
      ops: A list of tf.Operation. An OpRegularizer will be created for all the
        ops in `ops`, and recursively for all ops they depend on via data
        dependency. Typically `ops` would contain a single tf.Operation, which
        is the output of the network.
      gamma_threshold: A float scalar, will be used as a 'gamma_threshold' for
        all instances GammaL1Regularizer created by this class.
      regularizer_decorator: A class of OpRegularizer decorator to use.
      decorator_parameters: A dictionary of parameters to pass to the decorator
        factory. To be used only with decorators that requires parameters,
        otherwise use None.
      force_group: List of regex for ops that should be force-grouped.  Each
        regex corresponds to a separate group.  Use '|' operator to specify
        multiple patterns in a single regex. See op_regularizer_manager for more
        detail.
      regularizer_blacklist: List of regex for ops that should not be
        regularized. See op_regularizer_manager for more detail.
    """
    source_op_handler = batch_norm_source_op_handler.BatchNormSourceOpHandler(
        gamma_threshold)
    if regularizer_decorator:
      source_op_handler = op_handler_decorator.OpHandlerDecorator(
          source_op_handler, regularizer_decorator, decorator_parameters)
    op_handler_dict = collections.defaultdict(
        grouping_op_handler.GroupingOpHandler)
    op_handler_dict.update({
        'FusedBatchNorm':
            source_op_handler,
        'FusedBatchNormV2':
            source_op_handler,
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

    self._manager = orm.OpRegularizerManager(
        ops,
        op_handler_dict,
        force_group=force_group,
        regularizer_blacklist=regularizer_blacklist)
    self._calculator = cost_calculator.CostCalculator(
        self._manager, resource_function.activation_count_function)

  def get_regularization_term(self, ops=None):
    return self._calculator.get_regularization_term(ops)

  def get_cost(self, ops=None):
    return self._calculator.get_cost(ops)

  @property
  def op_regularizer_manager(self):
    return self._manager

  @property
  def name(self):
    return 'GammaActivationCount'

  @property
  def cost_name(self):
    return 'Activations'


class GroupLassoActivationRegularizer(generic_regularizers.NetworkRegularizer):
  """A NetworkRegularizer that targets activation count using L1 group lasso."""

  def __init__(
      self,
      ops,
      threshold,
      l1_fraction=0,
      regularizer_decorator: Type[generic_regularizers.OpRegularizer] = None,
      decorator_parameters=None,
      force_group=None,
      regularizer_blacklist=None):
    """Creates a GroupLassoActivationRegularizer object.

    Args:
      ops: A list of tf.Operation. An OpRegularizer will be created for all the
        ops in `ops`, and recursively for all ops they depend on via data
        dependency. Typically `ops` would contain a single tf.Operation, which
        is the output of the network.
      threshold: A float scalar, will be used as a 'threshold' for all
        regularizer instances created by this class.
      l1_fraction: Relative weight of L1 in L1 + L2 regularization.
      regularizer_decorator: A class of OpRegularizer decorator to use.
      decorator_parameters: A dictionary of parameters to pass to the decorator
        factory. To be used only with decorators that requires parameters,
        otherwise use None.
      force_group: List of regex for ops that should be force-grouped.  Each
        regex corresponds to a separate group.  Use '|' operator to specify
        multiple patterns in a single regex. See op_regularizer_manager for more
        detail.
      regularizer_blacklist: List of regex for ops that should not be
        regularized. See op_regularizer_manager for more detail.
    """
    conv2d_handler = conv2d_source_op_handler.Conv2DSourceOpHandler(
        threshold, l1_fraction)
    conv2d_transpose_handler = (
        conv2d_transpose_source_op_handler.Conv2DTransposeSourceOpHandler(
            threshold, l1_fraction))
    matmul_handler = matmul_source_op_handler.MatMulSourceOpHandler(
        threshold, l1_fraction)
    if regularizer_decorator:
      conv2d_handler = op_handler_decorator.OpHandlerDecorator(
          conv2d_handler, regularizer_decorator, decorator_parameters)
      conv2d_transpose_handler = op_handler_decorator.OpHandlerDecorator(
          conv2d_transpose_handler, regularizer_decorator, decorator_parameters)
      matmul_handler = op_handler_decorator.OpHandlerDecorator(
          matmul_handler, regularizer_decorator, decorator_parameters)

    op_handler_dict = collections.defaultdict(
        grouping_op_handler.GroupingOpHandler)
    op_handler_dict.update({
        'Conv2D':
            conv2d_handler,
        'Conv2DBackpropInput':
            conv2d_transpose_handler,
        'ConcatV2':
            concat_op_handler.ConcatOpHandler(),
        'DepthToSpace':
            output_non_passthrough_op_handler.OutputNonPassthroughOpHandler(),
        'DepthwiseConv2dNative':
            depthwise_convolution_op_handler.DepthwiseConvolutionOpHandler(),
        'MatMul':
            matmul_handler,
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
    })

    self._manager = orm.OpRegularizerManager(
        ops,
        op_handler_dict,
        force_group=force_group,
        regularizer_blacklist=regularizer_blacklist)
    self._calculator = cost_calculator.CostCalculator(
        self._manager, resource_function.activation_count_function)

  def get_regularization_term(self, ops=None):
    return self._calculator.get_regularization_term(ops)

  def get_cost(self, ops=None):
    return self._calculator.get_cost(ops)

  @property
  def op_regularizer_manager(self):
    return self._manager

  @property
  def name(self):
    return 'GroupLassoActivationCount'

  @property
  def cost_name(self):
    return 'Activations'
