"""Tensorflow OPs that stochastically gate (on/off) activations using sampling.

A set of Tensorflow OPs that provide stochastic gating (on/off) functionality.
This is done using the Gumbel-Softmax and Logistic-Sigmoid reparameterization
tricks. The gating probability is trainable.

References:
[1] Categorical Reparameterization with Gumbel-Softmax:
    https://arxiv.org/abs/1611.01144
[2] Fine-Grained Stochastic Architecture Search:
    https://arxiv.org/abs/2006.09581
"""

from __future__ import absolute_import
from __future__ import division
# [internal] enable type annotations
from __future__ import print_function

from morph_net.framework import tpu_util
import tensorflow.compat.v1 as tf
import tensorflow_probability as tfp
# pylint: disable=g-direct-tensorflow-import
from tensorflow.contrib import layers as contrib_layers
from tensorflow.python.framework import function
from tensorflow.python.ops import init_ops
# pylint: enable=g-direct-tensorflow-import

BN_FN = contrib_layers.batch_norm


class LogisticSigmoidGating(tf.keras.layers.Layer):
  """Keras layer for stochastic gating using the logistic-sigmoid trick.

    This layer provides a gating functionality based on the Logistic-Sigmoid
    relaxtion of a Bernoulli RV. A single gating sample is created for all
    alements across the axis of operation.
    For example, when applied on a tensor of shape [BS, W, H, C], and axis=3
    C independent Logistic-Sigmoid distributions are created. At every step, C
    values are sampled, and applied across all values in the dimension. I.e.
    all values across batch, width and height share one sample.
    For stability the layer works in the log odds space: log(p / 1-p).
  """

  def __init__(self,
               axis=3,
               temperature=0.001,
               straight_through=False,
               log_odds_init=2.5,
               soft_mask_for_inference=True,
               keep_first_channel_alive=True,
               annealing_rate=None,
               global_step=None,
               name=None):
    """Initialize gating layer (instantiate logit variables and define mask).

    Args:
      axis: The axis on which to gate. E.g. if activation is a [b, w, h, C]
        tensor and axis=3, the mask tensor will be of shape [1, 1, 1, C].
      temperature: Float scalar. The temperature variable that controls
        sampling. See [1] and [2] above for more details.
      straight_through: Bool. If True, use Straight Through sampling, in which
        the forward pass is discrete, and the backward pass uses the gradients
        of a differentiable approximation to the gate. This arg should not
        change when calling logistic_sigmoid_gating multiple times for the same
        graph, because there can be only one LogisticSigmoidGating op.
      log_odds_init: Number or TF Initializer function.
        Value to initialize the log odds ratio (logits).
        logits = log(p/1-p). Default: 2.5 => 92% of being 'on'.
      soft_mask_for_inference: multiple the mask by logits for inference.
      keep_first_channel_alive: If True, the first channel is always alive
        (unmasked). If False, it is possible to sample a mask of all zeros.
      annealing_rate: (Optional) Integer. If provided, the temperature is
        exponentially decayed by a factor of 0.1 every `annealing_rate`.
      global_step: (Optional) Required if decaying temperature (see above).
      name: Keras Layer name.
    """
    super(LogisticSigmoidGating, self).__init__(name=name)

    self.axis = axis
    self.temperature = temperature
    self.straight_through = straight_through
    self.log_odds_init = log_odds_init
    self.soft_mask_for_inference = soft_mask_for_inference
    self.keep_first_channel_alive = keep_first_channel_alive
    self.annealing_rate = annealing_rate
    self.global_step = global_step

    if isinstance(self.log_odds_init, (int, float)):
      self.log_odds_init = tf.constant_initializer(self.log_odds_init)
    elif not isinstance(self.log_odds_init, init_ops.Initializer):
      raise ValueError(
          'log_odds_init has unsupported value. '
          'Should be a number or an initializer. Instead got: {}'.format(
              type(self.log_odds_init)))

    if self.annealing_rate:
      if self.global_step is None:
        raise ValueError('Must provide global_step if decaying temperature.')
      self.temperature = self.temperature * (
          0.1 ** (tf.to_float(self.global_step) / float(self.annealing_rate)))

  def build(self, activation_shape):
    """Instantiate the mask variables (logits)."""
    self.mask_len = activation_shape[self.axis]
    self.logits = self.add_weight(
        name='mask_logits',
        shape=[self.mask_len],
        initializer=self.log_odds_init,
        trainable=True)

  def call(self, activation, is_training):
    """Build and apply stochastic mask on `activation`.

    Args:
      activation: 4D Float tensor on which to apply the mask.
      is_training: If False, no sampling is done. Gating is deterministically
        performed if the learned log_odds > 0 (probability > 50%).

    Returns:
      4D Float tensor: the masked activations.
    """
    if is_training:
      mask = tpu_util.write_to_variable(
          _logistic_sigmoid_sample(
              self.logits, self.temperature, self.straight_through),
          fail_if_exists=False)
    else:
      mask = tf.cast(self.logits > 0.0, self.logits.dtype)
      if self.soft_mask_for_inference:
        # Like dropout we multiply the mask by the prob.
        mask *= tf.sigmoid(self.logits)

    if self.keep_first_channel_alive:
      # TODO(y1) This is a hack. Find a better way to avoid breaking
      # the network. Currently this can confuse the FLOPs estimation of
      # Morphnet. Consider setting m to 1 for highest logit instead of first.
      m = tf.concat([tf.ones(1), tf.zeros(self.mask_len - 1)],
                    axis=0)
      mask = tf.maximum(mask, m)

    self.mask = tf.cast(mask, activation.dtype)
    return self.mask * activation


def logistic_sigmoid_gating(activation,
                            axis,
                            is_training,
                            temperature=0.001,
                            straight_through=False,
                            log_odds_init=2.5,
                            soft_mask_for_inference=True,
                            keep_first_channel_alive=True,
                            annealing_rate=None,
                            global_step=None,
                            scope=None):
  """Apply logistic-sigmoid gating (wrapper for LogisticSigmoidGating Layer)."""
  layer_fn = LogisticSigmoidGating(
      axis=axis,
      temperature=temperature,
      straight_through=straight_through,
      log_odds_init=log_odds_init,
      soft_mask_for_inference=soft_mask_for_inference,
      keep_first_channel_alive=keep_first_channel_alive,
      annealing_rate=annealing_rate,
      global_step=global_step,
      name=scope)
  return layer_fn(activation, is_training=is_training)


def _logistic_sigmoid_sample(logits, tau, straight_through):
  """Defines the Logistic Sigmoid OP.

     Defines the Logistic-Sigmoid OP with an optional "Straight-Through"
     implementation: A discrete forward pass, and soft backward pass.

  Args:
    logits: Float tensor. Values in this tensor represent the
      log odds ratio of the Bernoulli probability. i.e. log(p/1-p).
    tau: Float scalar. The temperature variable that controls sampling.
      See [1] and [2] above for more details.
    straight_through: Bool. If True, use Straight Through sampling, in which
      the forward pass is discrete, and the backward pass uses the gradients
      of a differentiable approximation to the gate.

  Returns:
    The sampled mask (gating) tensor.
  """
  logistic_dist = tfp.distributions.Logistic(loc=0.0, scale=1.0)
  logistic_sample = logistic_dist.sample(logits.shape.as_list())

  if not isinstance(tau, (float, tf.Tensor)):
    raise ValueError(
        '`tau` should be a float or tf.Tensor. Got: {}'.format(type(tau)))

  if isinstance(tau, float) and tau < 0.0:
    raise ValueError('`tau` should be positive.')

  if isinstance(tau, tf.Tensor) or tau > 0.0:

    @function.Defun(tf.float32, tf.float32, tf.float32, tf.float32)
    def _logistic_sigmoid_grad(log_odds, logistic_sample, tau, dy):
      mask = tf.nn.sigmoid((log_odds + logistic_sample) / tau)

      return tf.gradients([mask], [log_odds, logistic_sample, tau],
                          grad_ys=[dy])

    @function.Defun(
        tf.float32,
        tf.float32,
        tf.float32,
        grad_func=_logistic_sigmoid_grad,
        func_name='LogisticSigmoidGating')
    def _logistic_sigmoid(log_odds, logistic_sample, tau):
      """The LogisticSigmoidGating Op."""
      if straight_through:
        mask = (log_odds + logistic_sample) > 0.0
      else:
        mask = tf.nn.sigmoid((log_odds + logistic_sample) / tau)
      return tf.cast(mask, log_odds.dtype, name='gating_mask')

    mask = _logistic_sigmoid(logits, logistic_sample, tau)
  else:
    # In the tau=0 limit, straight_through is equivalent to not
    # straight_through. The gradient with respect to logits becomes infinite, so
    # we don't define it. Moreover, we will not register the Op
    # LogisticSigmoidGating and there is no associated fluidnet regularizer.
    mask = (logits + logistic_sample) > 0.0
    mask = tf.cast(mask, logits.dtype, name='gating_mask')

  mask.set_shape(logits.get_shape())
  return mask


def gated_batch_norm(gating_fn=logistic_sigmoid_gating,
                     axis=3,
                     is_training=True,
                     **kwargs_for_gating_fn):
  """Adds probabilistic gating to batch_norm.

  Example:
  gated_bn = activation_gating.gated_batch_norm()
  activation = tf.layers.conv2d(inputs, kernel=[3,3], num_outputs=6)
  gated_activation = gated_bn(activation, is_training=True)

  Args:
    gating_fn: Gating function to use. Default: logistic_sigmoid_gating.
    axis: The axis on which to gate. E.g. if activation is a [b, w, h, C] tensor
      and axis=3, the mask tensor will be of shape [1, 1, 1, C].
    is_training: If False, no sampling is done. Gating is deterministically
      performed if the learned log_odds_ratio > 0 (probability > 50%).
    **kwargs_for_gating_fn: Keyword args to pass into gating function.

  Returns:
    A callable that computes y = gating_fn(batch_norm(x))
  """
  return add_gating_to_fn(
      BN_FN, gating_fn=gating_fn, axis=axis, is_training=is_training,
      **kwargs_for_gating_fn)


def gated_relu_activation(gating_fn=logistic_sigmoid_gating,
                          axis=3,
                          is_training=True,
                          **kwargs_for_gating_fn):
  """Adds probabilistic gating to ReLU activation.

  Example:
  gated_relu = activation_gating.gated_relu_activation()
    activation = tf.layers.conv2d(inputs, kernel=[3,3], num_outputs=6)
    gated_activation = gated_relu(activation, is_training=True)

  Args:
    gating_fn: Gating function to use. Default: logistic_sigmoid_gating.
    axis: The axis on which to gate. E.g. if activation is a [b, w, h, C] tensor
      and axis=3, the mask tensor will be of shape [1, 1, 1, C].
    is_training: If False, no sampling is done. Gating is deterministically
      performed if the learned log_odds_ratio > 0 (probability > 50%).
    **kwargs_for_gating_fn: Keyword args to pass into gating function.

  Returns:
    A callable that computes y = gating_fn(tf.nn.relu(x))
  """
  return add_gating_to_fn(
      tf.nn.relu, gating_fn=gating_fn, axis=axis, is_training=is_training,
      **kwargs_for_gating_fn)


def add_gating_to_fn(
    fn, gating_fn=logistic_sigmoid_gating, axis=3,
    is_training=True, **kwargs_for_gating_fn):
  """Adds probabilistic gating to a provided function.

  Args:
    fn: A function pointer. The function is expected to output a TF tensor.
    gating_fn: Gating function to use. Default: logistic_sigmoid_gating.
    axis: The axis on which to gate. E.g. if activation is a [b, w, h, C] tensor
      and axis=3, the mask tensor will be of shape [1, 1, 1, C].
    is_training: If False, no sampling is done. Gating is deterministically
      performed if the learned log_odds_ratio > 0 (probability > 50%).
    **kwargs_for_gating_fn: Keyword args to pass into gating function.

  Returns:
    A callable that computes y = logistic_sigmoid_gating(fn(x))
  """
  def _gated_fn(*args, **kwargs):
    fn_out = fn(*args, **kwargs)
    return gating_fn(fn_out, axis, is_training, **kwargs_for_gating_fn)

  return _gated_fn


