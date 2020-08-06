"""A NetworkRegularizer that targets the number of FLOPs."""

from __future__ import absolute_import
from __future__ import division
# [internal] enable type annotations
from __future__ import print_function
from typing import Type, List

from morph_net.framework import batch_norm_source_op_handler
from morph_net.framework import conv2d_transpose_source_op_handler as conv2d_transpose_handler
from morph_net.framework import conv_source_op_handler as conv_handler
from morph_net.framework import generic_regularizers
from morph_net.framework import matmul_source_op_handler as matmul_handler
from morph_net.framework import op_handler_decorator
from morph_net.framework import op_handlers
from morph_net.framework import op_regularizer_manager as orm
from morph_net.network_regularizers import cost_calculator
from morph_net.network_regularizers import logistic_sigmoid_regularizer
from morph_net.network_regularizers import resource_function
import tensorflow.compat.v1 as tf


class LogisticSigmoidFlopsRegularizer(
    logistic_sigmoid_regularizer.LogisticSigmoidRegularizer):
  """A LogisticSigmoidRegularizer that targets FLOPs."""

  def get_calculator(self):
    return cost_calculator.CostCalculator(
        self._manager, resource_function.flop_function)

  @property
  def name(self):
    return 'LogisticSigmoidFlops'

  @property
  def cost_name(self):
    return 'FLOPs'


class GammaFlopsRegularizer(generic_regularizers.NetworkRegularizer):
  """A NetworkRegularizer that targets FLOPs using Gamma L1 as OpRegularizer."""

  def __init__(
      self,
      output_boundary: List[tf.Operation],
      gamma_threshold,
      regularizer_decorator: Type[generic_regularizers.OpRegularizer] = None,
      decorator_parameters=None,
      input_boundary: List[tf.Operation] = None,
      force_group=None,
      regularizer_blacklist=None):
    """Creates a GammaFlopsRegularizer object.

    Args:
      output_boundary: An OpRegularizer will be created for all these
        operations, and recursively for all ops they depend on via data
        dependency that does not involve ops from input_boundary.
      gamma_threshold: A float scalar, will be used as a 'gamma_threshold' for
        all instances GammaL1Regularizer created by this class.
      regularizer_decorator: A class of OpRegularizer decorator to use.
      decorator_parameters: A dictionary of parameters to pass to the decorator
        factory. To be used only with decorators that requires parameters,
        otherwise use None.
      input_boundary: A list of ops that represent the input boundary of the
        subgraph being regularized (input boundary is not regularized).
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
    op_handler_dict = op_handlers.get_gamma_op_handler_dict()
    op_handler_dict.update({
        'FusedBatchNorm': source_op_handler,
        'FusedBatchNormV2': source_op_handler,
        'FusedBatchNormV3': source_op_handler,
    })

    self._manager = orm.OpRegularizerManager(
        output_boundary,
        op_handler_dict,
        input_boundary=input_boundary,
        force_group=force_group,
        regularizer_blacklist=regularizer_blacklist)
    self._calculator = cost_calculator.CostCalculator(
        self._manager, resource_function.flop_function)

  def get_regularization_term(self, ops=None):
    return self._calculator.get_regularization_term(ops)

  def get_cost(self, ops=None):
    return self._calculator.get_cost(ops)

  @property
  def op_regularizer_manager(self):
    return self._manager

  @property
  def name(self):
    return 'GammaFlops'

  @property
  def cost_name(self):
    return 'FLOPs'


class GroupLassoFlopsRegularizer(generic_regularizers.NetworkRegularizer):
  """A NetworkRegularizer that targets FLOPs using L1 group lasso."""

  def __init__(
      self,
      output_boundary: List[tf.Operation],
      threshold,
      l1_fraction=0,
      regularizer_decorator: Type[generic_regularizers.OpRegularizer] = None,
      decorator_parameters=None,
      input_boundary: List[tf.Operation] = None,
      force_group=None,
      regularizer_blacklist=None):
    """Creates a GroupLassoFlopsRegularizer object.

    Args:
      output_boundary: An OpRegularizer will be created for all these
        operations, and recursively for all ops they depend on via data
        dependency that does not involve ops from input_boundary.
      threshold: A float scalar, will be used as a 'threshold' for all
        regularizer instances created by this class.
      l1_fraction: Relative weight of L1 in L1 + L2 regularization.
      regularizer_decorator: A class of OpRegularizer decorator to use.
      decorator_parameters: A dictionary of parameters to pass to the decorator
        factory. To be used only with decorators that requires parameters,
        otherwise use None.
      input_boundary: A list of ops that represent the input boundary of the
        subgraph being regularized (input boundary is not regularized).
      force_group: List of regex for ops that should be force-grouped.  Each
        regex corresponds to a separate group.  Use '|' operator to specify
        multiple patterns in a single regex. See op_regularizer_manager for more
        detail.
      regularizer_blacklist: List of regex for ops that should not be
        regularized. See op_regularizer_manager for more detail.
    """
    custom_handlers = {
        'Conv2D':
            conv_handler.ConvSourceOpHandler(threshold, l1_fraction),
        'Conv3D':
            conv_handler.ConvSourceOpHandler(threshold, l1_fraction),
        'Conv2DBackpropInput':
            conv2d_transpose_handler.Conv2DTransposeSourceOpHandler(
                threshold, l1_fraction),
        'MatMul':
            matmul_handler.MatMulSourceOpHandler(threshold, l1_fraction)
    }
    if regularizer_decorator:
      for key in custom_handlers:
        custom_handlers[key] = op_handler_decorator.OpHandlerDecorator(
            custom_handlers[key], regularizer_decorator, decorator_parameters)

    op_handler_dict = op_handlers.get_group_lasso_op_handler_dict()
    op_handler_dict.update(custom_handlers)

    self._manager = orm.OpRegularizerManager(
        output_boundary,
        op_handler_dict,
        input_boundary=input_boundary,
        force_group=force_group,
        regularizer_blacklist=regularizer_blacklist)
    self._calculator = cost_calculator.CostCalculator(
        self._manager, resource_function.flop_function)

  def get_regularization_term(self, ops=None):
    return self._calculator.get_regularization_term(ops)

  def get_cost(self, ops=None):
    return self._calculator.get_cost(ops)

  @property
  def op_regularizer_manager(self):
    return self._manager

  @property
  def name(self):
    return 'GroupLassoFlops'

  @property
  def cost_name(self):
    return 'FLOPs'
