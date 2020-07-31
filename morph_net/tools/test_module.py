"""Model-building functions for testing."""

import tensorflow.compat.v1 as tf
from tensorflow.compat.v1 import keras
from tensorflow.compat.v1.keras import layers  # type: ignore


Conv2D = layers.Conv2D
SeparableConv2D = layers.Conv2D
Dense = layers.Dense
BatchNormalization = layers.BatchNormalization
Activation = layers.Activation


def build_simple_keras_model(inputs):
  """Builds lightweight Keras model."""
  x = inputs
  x = layers.Conv2D(filters=10, kernel_size=3, name='conv')(x)
  x = layers.BatchNormalization(name='bn')(x)
  x = layers.Activation(tf.nn.relu, name='activation')(x)
  return x


def build_simple_keras_model_from_keras_lib(inputs):
  """Builds lightweight Keras model."""
  x = inputs
  x = keras.layers.Conv2D(filters=10, kernel_size=3, name='conv')(x)
  x = keras.layers.BatchNormalization(name='bn')(x)
  x = keras.layers.Activation(tf.nn.relu, name='activation')(x)
  return x


def build_simple_keras_model_from_local_aliases(inputs):
  """Builds lightweight Keras model."""
  x = inputs
  x = Conv2D(filters=10, kernel_size=3, name='conv')(x)
  x = BatchNormalization(name='bn')(x)
  x = Activation(tf.nn.relu, name='activation')(x)
  return x


def build_model_with_all_configurable_types(inputs):
  """Builds model that uses all of {Conv2D, SeparableConv2D, Dense}."""
  x = inputs
  x = Conv2D(filters=10, kernel_size=3, name='conv')(x)
  x = BatchNormalization(name='bn')(x)
  x = Activation(tf.nn.relu, name='activation')(x)
  x = SeparableConv2D(filters=10, kernel_size=3, name='sep_conv')(x)
  x = tf.reduce_sum(x, axis=[1, 2])
  x = Dense(units=10, name='dense')(x)
  return x


def build_two_branch_model(inputs):
  """Two parallel convolutions followed by Add and Concat."""
  branch1 = layers.Conv2D(filters=10, kernel_size=3, name='conv1')(inputs)
  branch1 = layers.BatchNormalization(name='bn1')(branch1)
  branch1 = layers.Activation(tf.nn.relu, name='activation1')(branch1)

  branch2 = layers.Conv2D(filters=10, kernel_size=3, name='conv2')(inputs)
  branch2 = layers.BatchNormalization(name='bn2')(branch2)
  branch2 = layers.Activation(tf.nn.relu, name='activation2')(branch2)

  merge = layers.Add()([branch1, branch2])
  concat = layers.Concatenate(axis=-1)([merge, branch1, branch2])
  return concat, branch1, branch2
