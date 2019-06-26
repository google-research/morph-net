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
import json
from enum import Enum

import tensorflow as tf

gfile = tf.gfile  # Aliase needed for mock.

VANISHED = 0.0
NUM_OUTPUTS = 'num_outputs'


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
  return maybe_tensor == VANISHED or maybe_tensor is None


class FallbackRule(Enum):
  """Fallback rules for the ConfigurableOps class."""
  pass_through = 'pass_through'
  strict = 'strict'
  zero = 'zero'


DEFAULT_FUNCTION_DICT = {
    'fully_connected': tf.contrib.layers.fully_connected,
    'conv2d': tf.contrib.layers.conv2d,
    'separable_conv2d': tf.contrib.layers.separable_conv2d,
    'concat': tf.concat,
    'add_n': tf.add_n,
    'avg_pool2d': tf.contrib.layers.avg_pool2d,
    'max_pool2d': tf.contrib.layers.max_pool2d,
    'batch_norm': tf.contrib.layers.batch_norm,
}

# Maps function names to the suffix of the name of the regularized ops.
SUFFIX_DICT = {
    'fully_connected': 'MatMul',
    'conv2d': 'Conv2D',
    'separable_conv2d': 'separable_conv2d',
}


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

    self._parameterization = parameterization or {}

    if not (isinstance(fallback_rule, FallbackRule) or
            isinstance(fallback_rule, str)):
      raise ValueError('fallback_rule must be a string or FallbackRule Enum')

    self._function_dict = function_dict or DEFAULT_FUNCTION_DICT
    self._suffix_dict = SUFFIX_DICT
    self._constructed_ops = collections.OrderedDict()
    if isinstance(fallback_rule, str):
      fallback_rule = FallbackRule[fallback_rule]  # Converts from string.
    self._default_to_zero = fallback_rule == FallbackRule.zero
    self._strict = fallback_rule == FallbackRule.strict

  @tf.contrib.framework.add_arg_scope
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

  @tf.contrib.framework.add_arg_scope
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

  @tf.contrib.framework.add_arg_scope
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
    num_outputs = _get_from_args_or_kwargs(NUM_OUTPUTS, 1, args, kwargs,
                                           False)
    if num_outputs is None:
      tf.logging.warning(
          'Trying to decorate separable_conv2d with num_outputs = None')
      kwargs[NUM_OUTPUTS] = None
    # This function actually only decorates the num_outputs of the Conv2D after
    # the depthwise convolution, as the former does not have any free params.

    fn, suffix = self._get_function_and_suffix('separable_conv2d')
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
    if 'scope' not in kwargs:  # TODO(e1): Can this be fixed.
      raise ValueError('kwargs must contain key \'scope\'')
    inputs = args[0] if args else kwargs.pop('inputs')
    if is_vanished(inputs):
      return VANISHED
    op_scope = kwargs['scope']
    current_scope = tf.contrib.framework.get_name_scope() or ''
    if current_scope and not current_scope.endswith('/'):
      current_scope += '/'
    op_name = ''.join([current_scope, op_scope, '/', suffix])

    # Assumes `inputs` is the first argument and `num_outputs` is the second
    # argument.
    num_outputs = self._parse_num_outputs(op_name, args, kwargs)
    args = args[2:]  # Possibly and empty list of < 3 positional args are used.

    self._insert_to_parameterization_log(op_name, num_outputs)
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

  @tf.contrib.framework.add_arg_scope
  def avg_pool2d(self, *args, **kwargs):
    return self._pass_through_mask(
        self._function_dict['avg_pool2d'], *args, **kwargs)

  @tf.contrib.framework.add_arg_scope
  def max_pool2d(self, *args, **kwargs):
    return self._pass_through_mask(
        self._function_dict['max_pool2d'], *args, **kwargs)

  @tf.contrib.framework.add_arg_scope
  def batch_norm(self, *args, **kwargs):
    return self._pass_through_mask(
        self._function_dict['batch_norm'], *args, **kwargs)

  def _parse_num_outputs(self, op_name, args, kwargs):
    """Computes the target NUM_OUTPUTS and adjusts kwargs in place.

    Will try to extract the number of outputs from the op_name's
    parameterization. If that's not possible, it will default to 0 when
    _default_to_zero is set, otherwise defaulting to the NUM_OUTPUTS argument
    that is either in kwargs or args[1].

    Args:
      op_name: A string, the name of the op to get NUM_OUTPUTS for.
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
    base_num_outputs = _get_from_args_or_kwargs(NUM_OUTPUTS, 1, args, kwargs)
    kwargs.pop(NUM_OUTPUTS, None)  # Removes num_outputs from kwargs if there.

    default_num_outputs = 0 if self._default_to_zero else base_num_outputs
    return self._parameterization.get(op_name, default_num_outputs)

  def _get_function_and_suffix(self, key):
    """Returns the function and suffix associated with key."""
    if key not in self._function_dict:
      raise KeyError('Function "%s" not supported by function_dict' % key)
    return self._function_dict[key], self._suffix_dict[key]

  def _insert_to_parameterization_log(self, name, num_outputs):
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
    filename, fallback_rule=FallbackRule.pass_through):
  """Create a ConfigurableOps from a parameterization file.

    Loads a json parameterization file from disk
    (as saved by tools.structure_exporter) and creates an ConfigurableOps from
    it.

  Args:
    filename: Path to a parameterization file in json format.
    fallback_rule: A `FallbackRule` enum which controls fallback behavior
      (see __init__ for more detail.)

  Returns:
    An ConfigurableOps instance with the parameterization from `filename`.
  """
  with gfile.Open(filename, 'r') as f:
    parameterization = json.loads(f.read())
    return ConfigurableOps(
        parameterization=parameterization, fallback_rule=fallback_rule)

