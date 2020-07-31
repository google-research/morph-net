"""A module that facilitates creation of configurable networks.

The goal of this module is to allow centralized parameterization of a (possibly)
complex deep network.

An important detail in this implementation is about the behaviour of function
given a trivial parameterization. By trivial we mean the case where
NUM_OUTPUTS is 0. We define the output of a trivial parameterization to be the
special value VANISHED, which is recognized by supported ops. We use 0.0 for its
value, so that it's treated as a regular 0.0 by supported Tensorflow ops.
This choice implies that:
  * For a vanished input, functions such as 'conv2d', or 'fully_connected' will
    also produce vanished output.
  * The 'concat' function will ignore VANISHED inputs. If all inputs are
    VANISHED, then the output is also VANISHED.

This edge-case behaviour achieves two goals:
  * It minimizes creation of ops in the graph which are not used.
  * It allows seamless use of the class in networks where some elements
    might be "turned off" by the parameterization.

For instance the following code will work for any parameterization of
the first and second convolutions, including 0.
```
# input.shape[-1] == 128
ops = ConfigurableOps(parameterization)
net_1 = ops.conv2d(input, num_outputs=256, kernel_size=1, scope='conv1')
net_2 = ops.conv2d(net_1, num_outputs=64, kernel_size=3, scope='conv2')
net_3 = ops.conv2d(net_2, num_outputs=128, kernel_size=1, scope='conv3')

output = net_3 + input
```
For `parameterization = '{'conv1': 0}'`
the values of `net_1`, `net_2`, and `net_3` will be all vanished sentinels, and
the bypass of this block will essentially vanish.

Note that the VANISHED functionality will save downsteam ops from being created
but not upstream ops.  For example, with `parameterization = '{'conv2': 0}'`,
then `net_1` will still be created.

"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import copy
import enum
import functools
import json

from morph_net.tools import structure_exporter as se
import tensorflow.compat.v1 as tf
from tensorflow.compat.v1 import layers as tf_layers
from tensorflow.compat.v1.keras import layers as keras_layers
from tensorflow.contrib import framework
from tensorflow.contrib import layers as contrib_layers
from tensorflow.contrib import slim as slim_layers
# gfile = tf.gfile  # Aliase needed for mock.

VANISHED = 0.0
_DEFAULT_NUM_OUTPUTS_KWARG = 'num_outputs'

DEFAULT_FUNCTION_DICT = {
    'fully_connected': contrib_layers.fully_connected,
    'conv2d': contrib_layers.conv2d,
    'separable_conv2d': contrib_layers.separable_conv2d,
    'concat': tf.concat,
    'add_n': tf.add_n,
    'avg_pool2d': contrib_layers.avg_pool2d,
    'max_pool2d': contrib_layers.max_pool2d,
    'batch_norm': contrib_layers.batch_norm,
}

_OP_SCOPE_DEFAULTS = {
    tf_layers.conv2d: 'conv2d',
    slim_layers.conv2d: 'Conv',
    contrib_layers.conv2d: 'Conv',

    tf_layers.separable_conv2d: 'separable_conv2d',
    slim_layers.separable_conv2d: 'SeparableConv2d',
    contrib_layers.separable_conv2d: 'SeparableConv2d',

    tf_layers.dense: 'dense',
    slim_layers.fully_connected: 'fully_connected',
    contrib_layers.fully_connected: 'fully_connected',
}

# Maps function names to the suffix of the name of the regularized ops.
_SUFFIX_DICT = {
    'fully_connected': 'MatMul',
    'conv2d': 'Conv2D',
    'separable_conv2d': 'separable_conv2d',
}


def get_function_dict(overrides=None):
  """Get mapping from function name to function for ConfigurableOps.

  Args:
    overrides: Dict: str -> function. Optionally replace entries in
      `DEFAULT_FUNCTION_DICT`.

  Returns:
    Dict: function name (str) to function.
  """
  overrides = overrides or {}
  function_dict = copy.deepcopy(DEFAULT_FUNCTION_DICT)
  function_dict.update(overrides)
  return function_dict


def is_vanished(maybe_tensor):
  """Checks if the argument represents a real tensor or None/vanished sentinel.

  For example:
    `is_vanished(ConfigurableOps({'conv1': 0}).conv2d(...,scope='conv1'))`
  returns True, since 0 channels in a conv2d produces vanished output.

  Args:
    maybe_tensor: A tensor or None or the vanished sentinel.

  Returns:
    A boolean, whether maybe_tensor is a tensor.
  """
  return (isinstance(maybe_tensor, float) and
          maybe_tensor == VANISHED) or maybe_tensor is None


class FallbackRule(enum.Enum):
  """Fallback rules for the ConfigurableOps class."""
  pass_through = 'pass_through'
  strict = 'strict'
  zero = 'zero'


def _get_op_name(configurable_keras_layer_instance):
  """Get full op name including scopes."""
  parameterization = configurable_keras_layer_instance.parameterization
  layer_name = configurable_keras_layer_instance.name
  op_suffixes = configurable_keras_layer_instance.op_suffixes
  is_strict = configurable_keras_layer_instance.is_strict

  full_scope = framework.get_name_scope()
  sep = '' if layer_name.endswith('/') else '/'
  possible_op_names = [full_scope + sep + suffix for suffix in op_suffixes]
  matches_in_parameterization = [
      name for name in possible_op_names if name in parameterization]

  if (is_strict and not matches_in_parameterization):
    raise KeyError(
        'None of the following op names were found in the '
        'parameterization: {}'.format(possible_op_names))

  if len(matches_in_parameterization) > 1:
    raise KeyError(
        'Found multiple matching op names in the parameterization: {}'.format(
            matches_in_parameterization))

  if matches_in_parameterization:
    return matches_in_parameterization[0]

  # If no op names were found in the parameterization, then return the default
  # op name generated by Keras
  default_op_suffix = configurable_keras_layer_instance.default_op_suffix
  return full_scope + '/' + default_op_suffix


def _insert_to_constructed_ops(
    configurable_keras_layer_instance, op_name, num_outputs):
  """Logs the NUM_OUTPUTS into constructed_ops dict."""
  constructed_ops = configurable_keras_layer_instance.constructed_ops
  if op_name in constructed_ops:
    tf.logging.warning('Function called more than once with scope %s.', op_name)
  constructed_ops[op_name] = num_outputs


def _configurable_keras_layer_build(configurable_keras_layer_instance):
  """Encapsulates the `build` logic of all Configurable Keras layers."""
  parameterization = configurable_keras_layer_instance.parameterization
  num_outputs_attr = configurable_keras_layer_instance.num_outputs_attr

  op_name = _get_op_name(configurable_keras_layer_instance)
  original_num_outputs = getattr(
      configurable_keras_layer_instance, num_outputs_attr)
  num_outputs = parameterization.get(op_name, original_num_outputs)
  _insert_to_constructed_ops(
      configurable_keras_layer_instance, op_name, num_outputs)
  configurable_keras_layer_instance.is_null_op = num_outputs == 0
  setattr(configurable_keras_layer_instance, num_outputs_attr, num_outputs)


class ConfigurableConv2D(keras_layers.Conv2D):
  """Keras Conv2D Layer that sets NUM_OUTPUTS according to parameterization."""

  def __init__(self,
               sentinel=None,
               parameterization=None,
               constructed_ops=None,
               is_strict=False,
               **kwargs):
    """Initialize configurable Keras layer.

    Args:
      sentinel: Sentinel argument to prevent positional arguments.
      parameterization: Dict: MorphNet-generated parameterization dict mapping
        op names to number of filters/neurons.
      constructed_ops: A dict to keep track of parameterized ops. NOTE: This
        dict will be modified during `build`.
      is_strict: If True, all constructed ops must exist in the
        parameterization.
      **kwargs: Kwargs for Keras layer.
    """
    if sentinel is not None:
      raise ValueError(
          'Found positional argument. __init__ only accepts kwargs.')
    self.parameterization = parameterization or {}
    self.constructed_ops = (constructed_ops if constructed_ops is not None else
                            collections.OrderedDict())
    self.is_strict = is_strict

    # may be modified by `_configurable_keras_layer_build` (during `build`)
    self.is_null_op = False

    super(ConfigurableConv2D, self).__init__(**kwargs)

  @property
  def op_suffixes(self):
    return ['Conv2D', 'ConfigurableConv2D']

  @property
  def default_op_suffix(self):
    return 'ConfigurableConv2D'

  @property
  def num_outputs_attr(self):
    return 'filters'

  def __call__(self, inputs, *args, **kwargs):
    if is_vanished(inputs):
      return VANISHED

    outputs = super(ConfigurableConv2D, self).__call__(inputs, *args, **kwargs)

    # Note: If num_outputs=0, we want to return VANISHED. However, we can't
    # return VANISHED in `call` since Keras requires the outputs of `call` be
    # Tensors. Instead, we let Keras build a no-op convolution (with output
    # channels = 0) and ignore its output.
    return VANISHED if self.is_null_op else outputs

  def build(self, input_shape):
    _configurable_keras_layer_build(self)
    return super(ConfigurableConv2D, self).build(input_shape)


class ConfigurableSeparableConv2D(keras_layers.SeparableConv2D):
  """Keras SeparableConv2D that sets NUM_OUTPUTS from parameterization."""

  def __init__(self,
               sentinel=None,
               parameterization=None,
               constructed_ops=None,
               is_strict=False,
               **kwargs):
    """Initialize configurable Keras layer.

    Args:
      sentinel: Sentinel argument to prevent positional arguments.
      parameterization: Dict: MorphNet-generated parameterization dict mapping
        op names to number of filters/neurons.
      constructed_ops: A dict to keep track of parameterized ops. NOTE: This
        dict will be modified during `build`.
      is_strict: If True, all constructed ops must exist in the
        parameterization.
      **kwargs: Kwargs for Keras layer.
    """
    if sentinel is not None:
      raise ValueError(
          'Found positional argument. __init__ only accepts kwargs.')
    self.parameterization = parameterization or {}
    self.constructed_ops = (constructed_ops if constructed_ops is not None else
                            collections.OrderedDict())
    self.is_strict = is_strict

    # may be modified by `_configurable_keras_layer_build` (during `build`)
    self.is_null_op = False

    super(ConfigurableSeparableConv2D, self).__init__(**kwargs)

  @property
  def op_suffixes(self):
    return ['separable_conv2d']

  @property
  def default_op_suffix(self):
    return 'separable_conv2d'

  @property
  def num_outputs_attr(self):
    return 'filters'

  def __call__(self, inputs, *args, **kwargs):
    if is_vanished(inputs):
      return VANISHED
    outputs = super(ConfigurableSeparableConv2D, self).__call__(
        inputs, *args, **kwargs)

    # Note: If num_outputs=0, we want to return VANISHED. However, we can't
    # return VANISHED in `call` since Keras requires the outputs of `call` be
    # Tensors. Instead, we let Keras build a no-op convolution (with output
    # channels = 0) and ignore its output.
    return VANISHED if self.is_null_op else outputs

  def build(self, input_shape):
    _configurable_keras_layer_build(self)
    return super(ConfigurableSeparableConv2D, self).build(input_shape)


class ConfigurableDense(keras_layers.Dense):
  """Keras Dense Layer that sets NUM_OUTPUTS according to parameterization."""

  def __init__(self,
               sentinel=None,
               parameterization=None,
               constructed_ops=None,
               is_strict=False,
               **kwargs):
    """Initialize configurable Keras layer.

    Args:
      sentinel: Sentinel argument to prevent positional arguments.
      parameterization: Dict: MorphNet-generated parameterization dict mapping
        op names to number of filters/neurons.
      constructed_ops: A dict to keep track of parameterized ops. NOTE: This
        dict will be modified during `build`.
      is_strict: If True, all constructed ops must exist in the
        parameterization.
      **kwargs: Kwargs for Keras layer.
    """
    if sentinel is not None:
      raise ValueError(
          'Found positional argument. __init__ only accepts kwargs.')
    self.parameterization = parameterization or {}
    self.constructed_ops = (constructed_ops if constructed_ops is not None else
                            collections.OrderedDict())
    self.is_strict = is_strict

    # may be modified by `_configurable_keras_layer_build` (during `build`)
    self.is_null_op = False

    super(ConfigurableDense, self).__init__(**kwargs)

  @property
  def op_suffixes(self):
    return ['Tensordot/MatMul']

  @property
  def default_op_suffix(self):
    return 'Tensordot/MatMul'

  @property
  def num_outputs_attr(self):
    return 'units'

  def __call__(self, inputs, *args, **kwargs):
    if is_vanished(inputs):
      return VANISHED
    outputs = super(ConfigurableDense, self).__call__(inputs, *args, **kwargs)

    # Note: If num_outputs=0, we want to return VANISHED. However, we can't
    # return VANISHED in `call` since Keras requires the outputs of `call` be
    # Tensors. Instead, we let Keras build a no-op convolution (with output
    # channels = 0) and ignore its output.
    return VANISHED if self.is_null_op else outputs

  def build(self, input_shape):
    _configurable_keras_layer_build(self)
    return super(ConfigurableDense, self).build(input_shape)


class PassThroughKerasLayerWrapper(object):
  """Wraps Keras Layer to handle VANISHED inputs.

  Wraps `keras_layer_class` (Keras Layer) to return VANISHED output on VANISHED
  input. This is useful when MorphNet removes convolutions from the network (by
  setting num_filters=0) in which case downstream ops should not be constructed.
  """

  def __init__(self,
               keras_layer_class,
               *args_for_keras_layer,
               **kwargs_for_keras_layer):
    """Initialize configurable Keras Layer wrapper.

    Args:
      keras_layer_class: Keras Layer class to wrap.
      *args_for_keras_layer: Args to initialize keras_layer_class (see
        `__call__`).
      **kwargs_for_keras_layer: Kwargs to initialize keras_layer_class (see
        `__call__`).
    """
    self.keras_layer_class = keras_layer_class
    self.args_for_keras_layer = args_for_keras_layer
    self.kwargs_for_keras_layer = kwargs_for_keras_layer

  def __call__(self, inputs, *args, **kwargs):
    # Handle list of tensors (Merge layers).
    if isinstance(inputs, (list, tuple)):
      inputs = [t for t in inputs if not is_vanished(t)]

      # If `inputs` is a list, we assume it is correct behavior to return
      # `inputs[0]` if len(inputs) == 1 (as is true with Add, Multiply,
      # Concatenate). We preempt __call__ since Merge layers require
      # len(inputs) > 1.
      if not inputs:
        return VANISHED
      elif len(inputs) == 1:
        return inputs[0]

    # Handle single tensor inputs.
    elif is_vanished(inputs):
      return VANISHED

    self.keras_layer_instance = self.keras_layer_class(
        *self.args_for_keras_layer,
        **self.kwargs_for_keras_layer)
    return self.keras_layer_instance(inputs, *args, **kwargs)


def hijack_keras_module(
    parameterization_or_file,
    module,
    fallback_rule=FallbackRule.pass_through,
    remove_common_prefix=False,
    keep_first_channel_alive=True):
  """Replaces Keras module with a fake "module" containing configurable Layers.

  If a module imports Keras layers:
  ```
  from tensorflow.compat.v1.keras.layers import Conv2D
  from tensorflow.compat.v1.keras.layers import SeparableConv2D
  # ...
  ```

  or defines global aliases:
  ```
  import tensorflow.compat.v1.keras.layers as keras_layers
  Conv2D = keras_layers.Conv2D
  ```

  then this function can be used to easily replace these classes with a
  configurable variant where the number of filters/neurons in each layer are
  set by a parameterization dict (or file) produced by MorphNet.

  After calling `hijack_keras_module(parameterization, module)`, any call to
  `module.Conv2D` or `module.Dense` will be replaced with a call to a unique
  Keras Layer which wraps the original base_class (see
  `_configurable_keras_layer_factory` for more details).

  Args:
    parameterization_or_file: Either:
      * A dict mapping op names to integer NUM_OUTPUTs
      * A path to a JSON file containing the above dict.
    module: A module name to override its Keras.
    fallback_rule: A `FallbackRule` enum which controls fallback behavior
      (see ConfigurableOps.__init__ for more details.)
    remove_common_prefix: If True, ignores outer level scope in all op names
      in the parameterization.
    keep_first_channel_alive: If True, keeps at least 1 neuron in each layer.

  Returns:
    (1) An OrderedDict of constructed ops.
    (2) A dict of function pointers before the hijacking.

  Raises:
    ValueError if trying to hijack the Keras layers module
    (`tensorflow.compat.v1.keras`) itself.
  """
  parameterization_or_file = parameterization_or_file or {}
  if not isinstance(parameterization_or_file, (str, dict)):
    raise ValueError(
        'Expected dict or string (filename) for `parameterization_or_file`. '
        'Instead got: {}'.format(type(parameterization_or_file)))
  if isinstance(parameterization_or_file, str):
    with tf.gfile.Open(parameterization_or_file, 'r') as f:
      parameterization = json.loads(f.read())
  else:
    parameterization = parameterization_or_file

  if remove_common_prefix:
    rename_op = se.get_remove_common_prefix_fn(parameterization)
    parameterization = {rename_op(k): v for k, v in parameterization.items()}

  if keep_first_channel_alive:
    parameterization = {k: max(v, 1) for k, v in parameterization.items()}

  fallback_rule = _get_fallback_rule_as_enum(fallback_rule)
  is_strict = fallback_rule == FallbackRule.strict
  original_layer_classes = {}
  constructed_ops = collections.OrderedDict()

  def _maybe_replace_layer_class(class_name, configurable_layer_class):
    if hasattr(module, class_name):
      original_layer_classes[class_name] = getattr(module, class_name)
      setattr(module, class_name, configurable_layer_class)

  for configurable_layer_class, class_name in [
      (ConfigurableConv2D, 'Conv2D'),
      (ConfigurableSeparableConv2D, 'SeparableConv2D'),
      (ConfigurableDense, 'Dense')]:
    configurable_layer_class = functools.partial(
        configurable_layer_class,
        parameterization=parameterization,
        constructed_ops=constructed_ops,
        is_strict=is_strict)
    _maybe_replace_layer_class(class_name, configurable_layer_class)

  for keras_layer_class, class_name in [
      (keras_layers.BatchNormalization, 'BatchNormalization'),
      (keras_layers.Activation, 'Activation'),
      (keras_layers.UpSampling2D, 'UpSampling2D'),
      (keras_layers.Add, 'Add'),
      (keras_layers.Concatenate, 'Concatenate'),
      (keras_layers.Multiply, 'Multiply')]:
    pass_through_layer_class = functools.partial(
        PassThroughKerasLayerWrapper,
        keras_layer_class)
    _maybe_replace_layer_class(class_name, pass_through_layer_class)

  return constructed_ops, original_layer_classes


def _get_fallback_rule_as_enum(fallback_rule):
  if not (isinstance(fallback_rule, FallbackRule) or
          isinstance(fallback_rule, str)):
    raise ValueError('fallback_rule must be a string or FallbackRule Enum')
  if isinstance(fallback_rule, str):
    return FallbackRule[fallback_rule]  # Converts from string.
  return fallback_rule


class ConfigurableOps(object):
  """A class that facilitates structure modification of a Tensorflow graph.

  The ConfigurableOps allows modifications of the NUM_OUTPUTS argument ops.
  The functionality is determined by a 'parameterization' and by modifiers.
  The 'parameterization' is a map between scope names and new NUM_OUTPUTS
  values. If the scope of an op matches a key from the parameterization, the
  decorator will override the NUM_OUTPUTS argument.

  Another feature of the ConfigurableOps is support for vanishing input sizes
  that arise when an the NUM_OUTPUTS argument of a downstream op is set to
  zero. Specifically, the functions decorated by the class adhere to the
  following input/output logic:
    * If NUM_OUTPUTS is set to zero, then the output of the op will be the
      vanished sentinel, and will return False when checked with is_vanished().
    * If the input is vanished, the output will be the same.
    * The concatenation (configurable_ops.concat) of an vanished element with
      other tensors ignores the vanished elements. The result of concatenating
      only vanished elements is also vanished.

  In addition the object collects and report the actual NUM_OUTPUTS argument
  that was used in every context.
  """

  def __init__(self,
               parameterization=None,
               function_dict=None,
               fallback_rule=FallbackRule.pass_through):
    """Constructs a ConfigurableOps.

    Args:
      parameterization: A dictionary between scope name to be overridden and a
        integer which is the target NUM_OUTPUTS.
      function_dict: A dict between names of ops (strings) and functions
        which accept num_outputs as the second argument. If None defaults to
        DEFAULT_FUNCTION_DICT.
      fallback_rule: A `FallbackRule` enum which controls fallback behavior:
          * 'pass_through' provided NUM_OUTPUTS is passed to decorated
            function (default).
          * 'strict' requires the scope name appear in parameterization or else
            throws an error.
          * 'zero' uses `num_outputs` equal to zero if scope name is not in the
            parameterization.
    Raises:
      ValueError: If fallback_rule is not a string or a FallbackRule enum.
    """

    fallback_rule = _get_fallback_rule_as_enum(fallback_rule)
    self._parameterization = parameterization or {}
    self._function_dict = function_dict or DEFAULT_FUNCTION_DICT
    self._suffix_dict = _SUFFIX_DICT
    self._constructed_ops = collections.OrderedDict()
    self._default_to_zero = fallback_rule == FallbackRule.zero
    self._strict = fallback_rule == FallbackRule.strict
    self.default_scope_to_counts_map = {}

    # To keep track of the number of identical scopes encountered
    self._scope_counts = {}

  @property
  def parameterization(self):
    """Returns the parameterization dict mapping op names to num_outputs."""
    return self._parameterization

  @framework.add_arg_scope
  def conv2d(self, *args, **kwargs):
    """Masks num_outputs from the function pointed to by 'conv2d'.

    The object's parameterization has precedence over the given NUM_OUTPUTS
    argument. The resolution of the op names uses
    tf.contrib.framework.get_name_scope() and kwargs['scope'].

    Args:
      *args: Arguments for the operation.
      **kwargs: Key arguments for the operation.

    Returns:
      The result of the application of the function_dict['conv2d'] to the given
      'inputs', '*args' and '**kwargs' while possibly overriding NUM_OUTPUTS
      according the parameterization.

    Raises:
      ValueError: If kwargs does not contain a key named 'scope'.
    """
    fn, suffix = self._get_function_and_suffix('conv2d')
    return self._mask(fn, suffix, *args, **kwargs)

  @framework.add_arg_scope
  def fully_connected(self, *args, **kwargs):
    """Masks NUM_OUTPUTS from the function pointed to by 'fully_connected'.

    The object's parameterization has precedence over the given NUM_OUTPUTS
    argument. The resolution of the op names uses
    tf.contrib.framework.get_name_scope() and kwargs['scope'].

    Args:
      *args: Arguments for the operation.
      **kwargs: Key arguments for the operation.

    Returns:
      The result of the application of the function_map['fully_connected'] to
      the given 'inputs', '*args' and '**kwargs' while possibly overriding
      NUM_OUTPUTS according the parameterization.

    Raises:
      ValueError: If kwargs does not contain a key named 'scope'.
    """
    inputs = _get_from_args_or_kwargs('inputs', 0, args, kwargs)
    if inputs.shape.ndims != 2:
      raise ValueError(
          'ConfigurableOps does not suport fully_connected with rank != 2')
    fn, suffix = self._get_function_and_suffix('fully_connected')
    return self._mask(fn, suffix, *args, **kwargs)

  @framework.add_arg_scope
  def separable_conv2d(self, *args, **kwargs):
    """Masks NUM_OUTPUTS from the function pointed to by 'separable_conv2d'.

    The object's parameterization has precedence over the given NUM_OUTPUTS
    argument. The resolution of the op names uses
    tf.contrib.framework.get_name_scope() and kwargs['scope'].

    Args:
      *args: Arguments for the operation.
      **kwargs: Key arguments for the operation.

    Returns:
      The result of the application of the function_map['separable_conv2d'] to
      the given 'inputs', '*args', and '**kwargs' while possibly overriding
      NUM_OUTPUTS according the parameterization.

    Raises:
      ValueError: If kwargs does not contain a key named 'scope'.
    """
    # This function actually only decorates the num_outputs of the Conv2D after
    # the depthwise convolution, as the former does not have any free params.
    fn, suffix = self._get_function_and_suffix('separable_conv2d')
    num_outputs_kwarg_name = self._get_num_outputs_kwarg_name(fn)
    num_outputs = _get_from_args_or_kwargs(
        num_outputs_kwarg_name, 1, args, kwargs, False)
    if num_outputs is None:
      tf.logging.warning(
          'Trying to decorate separable_conv2d with num_outputs = None')
      kwargs[num_outputs_kwarg_name] = None

    return self._mask(fn, suffix, *args, **kwargs)

  def _mask(self, function, suffix, *args, **kwargs):
    """Masks num_outputs from the given function.

    The object's parameterization has precedence over the given NUM_OUTPUTS
    argument. The resolution of the op names uses
      `tf.contrib.framework.get_name_scope()` and `kwargs['scope']`.

    The NUM_OUTPUTS argument is assumed to be either in **kwargs or held in
    *args[1].

    In case the `inputs` argument is VANISHED or that `num_outputs` is 0,
    returns VANISHED without adding ops to the graph.

    Args:
      function: A callable function to mask the NUM_OUTPUTS parameter from.
        Examples for functions are in DEFAULT_FUNCTION_DICT.
        The callable function must accept a NUM_OUTPUTS parameter as the
        second argument.
      suffix: A string with the suffix of the op name.
      *args: Arguments for the operation.
      **kwargs: Key arguments for the operation.

    Returns:
      The result of the application of the function to the given 'inputs',
      '*args', and '**kwargs' while possibly overriding NUM_OUTPUTS according
      to the parameterization.

    Raises:
      ValueError: If kwargs does not contain a key named 'scope'.
    """
    inputs = args[0] if args else kwargs.pop('inputs')
    if is_vanished(inputs):
      return VANISHED

    # Support for tf.contrib.layers and tf.layers API.
    op_scope = kwargs.get('scope')
    current_scope = framework.get_name_scope() or ''
    if current_scope and not current_scope.endswith('/'):
      current_scope += '/'

    op_scope = kwargs.get('scope') or kwargs.get('name')
    if op_scope:
      if op_scope.endswith('/'):
        raise ValueError(
            'Scope `{}` ends with `/` which leads to unexpected '
            'behavior.'.format(op_scope))
      full_scope = current_scope + op_scope
    else:
      # Use default scope, optionally appending a unique ID if scope exists
      if function not in _OP_SCOPE_DEFAULTS:
        raise ValueError(
            'No `scope` or `name` found in kwargs, and no default scope '
            'defined for {}'.format(_get_function_name(function)))
      op_scope = _OP_SCOPE_DEFAULTS[function]
      full_scope = current_scope + op_scope
      if full_scope in self._scope_counts:
        new_scope = full_scope + '_' + str(self._scope_counts[full_scope])
        self._scope_counts[full_scope] += 1
        full_scope = new_scope
      else:
        self._scope_counts[full_scope] = 1

    op_name = full_scope + '/' + suffix

    # Assumes `inputs` is the first argument and `num_outputs` is the second
    # argument.
    num_outputs = self._parse_num_outputs(
        op_name, self._get_num_outputs_kwarg_name(function), args, kwargs)
    args = args[2:]  # Possibly and empty list of < 3 positional args are used.

    self._insert_to_constructed_ops(op_name, num_outputs)
    if num_outputs == 0:
      return VANISHED

    return function(inputs, num_outputs, *args, **kwargs)

  @property
  def constructed_ops(self):
    """Returns a dictionary between op names built to their NUM_OUTPUTS.

       The dictionary will contain an op.name: NUM_OUTPUTS pair for each op
       constructed by the decorator. The dictionary is ordered according to the
       order items were added.
       The parameterization is accumulated during all the calls to the object's
       members, such as `conv2d`, `fully_connected` and `separable_conv2d`.
       The values used are either the values from the parameterization set for
       the object, or the values that where passed to the members.
    """
    return self._constructed_ops

  def concat(self, *args, **kwargs):
    return self._pass_through_mask_list('concat', 'values', *args, **kwargs)

  def add_n(self, *args, **kwargs):
    return self._pass_through_mask_list('add_n', 'inputs', *args, **kwargs)

  @framework.add_arg_scope
  def avg_pool2d(self, *args, **kwargs):
    return self._pass_through_mask(
        self._function_dict['avg_pool2d'], *args, **kwargs)

  @framework.add_arg_scope
  def max_pool2d(self, *args, **kwargs):
    return self._pass_through_mask(
        self._function_dict['max_pool2d'], *args, **kwargs)

  @framework.add_arg_scope
  def batch_norm(self, *args, **kwargs):
    return self._pass_through_mask(
        self._function_dict['batch_norm'], *args, **kwargs)

  def _get_num_outputs_kwarg_name(self, function):
    """Gets the `num_outputs`-equivalent kwarg for a supported function."""
    alt_num_outputs_kwarg = {
        tf_layers.conv2d: 'filters',
        tf_layers.separable_conv2d: 'filters',
        tf_layers.dense: 'units',
    }
    return alt_num_outputs_kwarg.get(function, _DEFAULT_NUM_OUTPUTS_KWARG)

  def _parse_num_outputs(self, op_name, num_outputs_kwarg_name, args, kwargs):
    """Computes the target NUM_OUTPUTS and adjusts kwargs in place.

    Will try to extract the number of outputs from the op_name's
    parameterization. If that's not possible, it will default to 0 when
    _default_to_zero is set, otherwise defaulting to the NUM_OUTPUTS argument
    that is either in kwargs or args[1].

    Args:
      op_name: A string, the name of the op to get NUM_OUTPUTS for.
      num_outputs_kwarg_name: A string, the name of the `num_outputs`-equivalent
        kwarg.
      args: Position arguments for the callable. Assumes that NUM_OUTPUTS
      position is 1.
      kwargs: key word arguments for the callable.

    Returns:
      The target value.

    Raises:
      KeyError: If strict and op_name not found in parameterization.
    """
    if self._strict and op_name not in self._parameterization:
      # If strict and op_name not found in parameterization, throw an error.
      raise KeyError('op_name \"%s\" not found in parameterization' % op_name)

    # Assumes that the position of num_outputs is 1.
    base_num_outputs = _get_from_args_or_kwargs(
        num_outputs_kwarg_name, 1, args, kwargs)
    kwargs.pop(num_outputs_kwarg_name, None)  # Removes num_outputs from kwargs.

    default_num_outputs = 0 if self._default_to_zero else base_num_outputs
    return self._parameterization.get(op_name, default_num_outputs)

  def _get_function_and_suffix(self, key):
    """Returns the function and suffix associated with key."""
    if key not in self._function_dict:
      raise KeyError('Function "%s" not supported by function_dict' % key)
    return self._function_dict[key], self._suffix_dict[key]

  def _insert_to_constructed_ops(self, name, num_outputs):
    """Logs the NUM_OUTPUTS for scope 'name' into _constructed_ops."""
    if name in self._constructed_ops:
      tf.logging.warning('Function called more than once with scope %s.', name)
    self._constructed_ops[name] = num_outputs

  def _pass_through_mask_list(self, fn_name, inputs_name, *args, **kwargs):
    """Drops any tensors that are None or vanished and applies `fn` to result.

    Assumes the first argument to `fn` is the list of tensors.

    Args:
      fn_name: Function name to apply on filtered inputs, must be a key of
        'function_dict'.
      inputs_name: Name of the input argument (in case it's passed as a kwarg).
      *args: Args for the function defined by `fn_name`.
      **kwargs: Kwargs for he function defined by `fn_name`.

    Returns:
      Output of function on filtered inputs, or vanished if all inputs are
        vanished.
    """
    if fn_name not in self._function_dict:
      raise ValueError('Unrecognized function name %s' % fn_name)
    fn = self._function_dict[fn_name]
    if args:
      inputs = args[0]
      args = args[1:]
    else:
      if inputs_name not in kwargs:
        raise ValueError('Missing `{}` argument.'.format(inputs_name))
      inputs = kwargs.pop(inputs_name)

    inputs = [t for t in inputs if not is_vanished(t)]
    return fn(inputs, *args, **kwargs) if inputs else VANISHED

  def _pass_through_mask(self, layer_fn, *args, **kwargs):
    inputs = args[0] if args else kwargs['inputs']
    return VANISHED if is_vanished(inputs) else layer_fn(*args, **kwargs)


def _get_from_args_or_kwargs(name, index, args, kwargs, is_required=True):
  try:
    return kwargs[name] if name in kwargs else args[index]
  except IndexError:
    if is_required:
      raise ValueError('Argument `{}` is required.'.format(name))
    return None


def _get_function_name(function):
  """Get a descriptive identifier for `function`."""
  return '{}.{}'.format(function.__module__, function.__name__)


def hijack_module_functions(configurable_ops, module):
  """Hijacks the functions from module using configurable_ops.

  Overrides globally declared function reference in module with configurable_ops
  member functions.

  If a module defines global aliases in the form:

  example_module.py
    ```
    conv2d = tr.contrib.layers.conv2d
    fully_connected = tr.contrib.layers.fully_connected

    def build_layer(inputs):
      return conv2d(inputs, 64, 3, scope='demo')
    ```

  Then this function provides the possibility to replace these aliases with
  the members of the given `configurable_ops` object.

  So after a call to `hijack_module_functions(configurable_ops, example_module)`
  the call `example_module.build_layer(net)` will under the hood use
  `configurable_ops.conv2d` rather than `tf.contrib.layers.conv2d`.

  Note: This function could be unsafe as it depends on aliases defined in a
  possibly external module. In addition, a function in that module that calls
  directly, will not be affected by the hijacking, for instance:

  ```
    def build_layer_not_affected(inputs):
      return tf.contrib.layers.conv2d(inputs, 64, 3, scope='bad')
  ```

  Args:
    configurable_ops: An ConfigurableOps object, to use functions as defined in
    'DEFAULT_FUNCTION_DICT'.
    module: A module name to override its functions.

  Returns:
    A dict of the function pointers before the hijacking.
  """
  originals = {}

  def maybe_setattr(attr):
    """Sets module.attr = configurable_ops.attr if module has attr.

    Overrides module.'attr' with configurable_ops.'attr', if module already has
    an attribute name 'attr'.

    Args:

      attr: Name of the attribute to override.
    """
    if hasattr(module, attr):
      originals[attr] = getattr(module, attr)
      setattr(module, attr, getattr(configurable_ops, attr))

  for fn in DEFAULT_FUNCTION_DICT:
    maybe_setattr(fn)
  return originals


def recover_module_functions(originals, module):
  """Recovers the functions hijacked to from module.

  Args:
    originals: Dict of functions to recover. Assumes keys are a contained in
    'DEFAULT_FUNCTION_DICT'.
    module: A module name to recover its functions.

  """
  for attr, original in originals.items():
    setattr(module, attr, original)


def decorator_from_parameterization_file(
    filename, fallback_rule=FallbackRule.pass_through, **kwargs):
  """Create a ConfigurableOps from a parameterization file.

    Loads a json parameterization file from disk
    (as saved by tools.structure_exporter) and creates an ConfigurableOps from
    it.

  Args:
    filename: Path to a parameterization file in json format.
    fallback_rule: A `FallbackRule` enum which controls fallback behavior
      (see __init__ for more detail.)
    **kwargs: Miscellaneous args for ConfigurableOps.

  Returns:
    An ConfigurableOps instance with the parameterization from `filename`.
  """
  with tf.gfile.Open(filename, 'r') as f:
    parameterization = json.loads(f.read())
    return ConfigurableOps(
        parameterization=parameterization, fallback_rule=fallback_rule,
        **kwargs)
