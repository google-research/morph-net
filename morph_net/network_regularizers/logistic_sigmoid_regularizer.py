"""NetworkRegularizers applied on top of probabilistic sampling."""

from __future__ import absolute_import
from __future__ import division
# [internal] enable type annotations
from __future__ import print_function

import abc
from typing import Type, List

from morph_net.framework import generic_regularizers
from morph_net.framework import logistic_sigmoid_source_op_handler as ls_handler
from morph_net.framework import op_handler_decorator
from morph_net.framework import op_handlers
from morph_net.framework import op_regularizer_manager as orm
from morph_net.framework import probabilistic_grouping_regularizer as pgr

import six
import tensorflow.compat.v1 as tf


@six.add_metaclass(abc.ABCMeta)
class LogisticSigmoidRegularizer(generic_regularizers.NetworkRegularizer):
  """Base class for NetworkRegularizers that use probabilistic sampling."""

  def __init__(
      self,
      output_boundary: List[tf.Operation],
      regularize_on_mask=True,
      alive_threshold=0.1,
      mask_as_alive_vector=True,
      regularizer_decorator: Type[generic_regularizers.OpRegularizer] = None,
      decorator_parameters=None,
      input_boundary: List[tf.Operation] = None,
      force_group=None,
      regularizer_blacklist=None):
    """Creates a LogisticSigmoidFlopsRegularizer object.

    Args:
      output_boundary: An OpRegularizer will be created for all these
        operations, and recursively for all ops they depend on via data
        dependency that does not involve ops from input_boundary.
      regularize_on_mask: Bool. If True uses the binary mask as the
        regularization vector. Else uses the probability vector.
      alive_threshold: Float. Threshold below which values are considered dead.
        This can be used both when mask_as_alive_vector is True and then the
        threshold is used to binarize the sampled values and
        when mask_as_alive_vector is False, and then the threshold is on the
        channel probability.
      mask_as_alive_vector: Bool. If True use the thresholded sampled mask
        as the alive vector. Else, use thresholded probabilities from the
        logits.
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
    source_op_handler = ls_handler.LogisticSigmoidSourceOpHandler(
        regularize_on_mask, alive_threshold, mask_as_alive_vector)
    if regularizer_decorator:
      source_op_handler = op_handler_decorator.OpHandlerDecorator(
          source_op_handler, regularizer_decorator, decorator_parameters)
    op_handler_dict = op_handlers.get_gamma_op_handler_dict()
    op_handler_dict.update({
        'LogisticSigmoidGating': source_op_handler,
    })

    self._manager = orm.OpRegularizerManager(
        output_boundary,
        op_handler_dict,
        create_grouping_regularizer=pgr.ProbabilisticGroupingRegularizer,
        input_boundary=input_boundary,
        force_group=force_group,
        regularizer_blacklist=regularizer_blacklist)
    self._calculator = self.get_calculator()

  @abc.abstractmethod
  def get_calculator(self):
    pass

  def get_regularization_term(self, ops=None):
    return self._calculator.get_regularization_term(ops)

  def get_cost(self, ops=None):
    return self._calculator.get_cost(ops)

  @property
  def op_regularizer_manager(self):
    return self._manager
