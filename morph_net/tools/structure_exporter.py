"""Module for calculating and saving learned structures.

When training with a network regularizer, the emerging structure of the
network is encoded in the `alive_vector`s and `regularization_vector`s of the
`OpRegularizerManager`.
"""

from __future__ import absolute_import
from __future__ import division
# [internal] enable type annotations
from __future__ import print_function

import json
import os
from morph_net.framework import op_regularizer_manager as orm
import numpy as np
import tensorflow.compat.v1 as tf
from typing import Text, Sequence, Dict, Optional, IO, Iterable, Callable

SUPPORTED_OPS = ['Conv2D', 'Conv2DBackpropInput', 'Conv3D']
ALIVE_FILENAME = 'alive'


class StructureExporter(object):
  """Reports statistics about the current state of regularization.

  Obtains live activation counts for supported ops: a map from each op name
  to its count of alive activations (filters).

  Usage:
    1. Build model.
      `logits = build_model(parmas)`
    2. Create network regularizer.
      `network_regularizer = flop_regularizer.GammaFlopsRegularizer([logits.op])
    3. Create StructureExporter:
      `exporter = StructureExporter(net_reg.op_regularizer_manager)`
    4. Gather tensors to eval:
      `tensor_to_eval_dict = exporter.tensors`
    5. Within a `tf.Session()` eval and populate tensors:
      `exporter.populate_tensor_values(tensor_to_eval_dict.eval())`
    6. Export structure:
      `exporter.save_alive_counts(tf.gfile.Open(...))`
  """

  def __init__(self,
               op_regularizer_manager: orm.OpRegularizerManager,
               remove_common_prefix: bool = False) -> None:
    """Build a StructureExporter object.

    Args:
      op_regularizer_manager: An OpRegularizerManager, an object that contains
        info about every op we care about and its corresponding regularizer.
      remove_common_prefix: A bool. If True, determine if all op names start
        with the same prefix (up to and including the first '/'), and if so,
        skip that prefix in exported data.
    """
    self._alive_vectors_as_tensors = {}  # type: Dict[Text, tf.Tensor]
    self._alive_vectors_as_values = None  # type: Optional[Dict[Text, Sequence[bool]]]

    for op in op_regularizer_manager.ops:
      if op.type not in SUPPORTED_OPS:
        continue
      op_regularizer = op_regularizer_manager.get_regularizer(op)
      if not op_regularizer:
        tf.logging.warning('No regularizer found for: %s', op.name)
        continue
      self._alive_vectors_as_tensors[op.name] = op_regularizer.alive_vector

    if remove_common_prefix:
      rename_op = get_remove_common_prefix_fn(self._alive_vectors_as_tensors)
      self._alive_vectors_as_tensors = {
          rename_op(k): v for k, v in self._alive_vectors_as_tensors.items()
      }

  @property
  def tensors(self):
    """A dictionary between op names and alive vectors.

    Alive vectors are `tf.Tensor`s of type tf.int32.

    Returns:
      Dict: op name -> alive vector tensor
    """
    return self._alive_vectors_as_tensors

  def populate_tensor_values(self, values: Dict[Text, Sequence[bool]]) -> None:
    """Records alive values for ops regularized by op_regularizer_manager.

    The given mapping must match op names from `self.tensor`.

    Args:
      values: A dict mapping op names to a boolean alive status.

    Raises:
      ValueError: If keys of input do not match keys of `self.tensor`.
    """
    # TODO(p1): Rename values to something better. values is a dict!
    if sorted(values) != sorted(self.tensors):
      raise ValueError(
          '`values` and `self.tensors` must have the same keys but are %s and %s'
          % (sorted(values), sorted(self.tensors)))
    self._alive_vectors_as_values = values

  def get_alive_counts(self) -> Dict[Text, int]:
    """Computes alive counts.

    populate_tensor_values() must have been called earlier.

    Returns:
      A dict {op_name: alive_count}, alive_count is a scalar integer tf.Tensor.

    Raises:
      RuntimeError: tensor values not populated.
    """

    if self._alive_vectors_as_values is None:
      raise RuntimeError('Tensor values not populated.')
    return _compute_alive_counts(self._alive_vectors_as_values)

  def save_alive_counts(self, f: IO[bytes]) -> None:
    """Saves live counts to a file.

    Args:
      f: a file object where alive counts are saved.
    """
    f.write(format_structure(self.get_alive_counts()))  # pytype: disable=wrong-arg-types

  def create_file_and_save_alive_counts(self, base_dir: Text,
                                        global_step: int) -> None:
    """Creates and updates files with alive counts.

    Creates the directory `{base_dir}/learned_structure/` and saves the current
    alive counts to:
      `{base_dir}/learned_structure/{ALIVE_FILENAME}_{global_step}`.

    Args:
      base_dir: where to export the alive counts.
      global_step: current value of global step, used as a suffix in filename.
    """
    current_filename = '%s_%s' % (ALIVE_FILENAME, global_step)
    directory = os.path.join(base_dir, 'learned_structure')
    try:
      tf.gfile.MakeDirs(directory)
    except tf.errors.OpError:
      # Probably already exists. If not, we'll see the error in the next line.
      pass
    with tf.gfile.Open(os.path.join(directory, current_filename), 'w') as f:
      self.save_alive_counts(f)  # pytype: disable=wrong-arg-types


# TODO(p1): maybe check that we still end up with unique names after prefix
# removal, and do nothing if that's not the case?
def get_remove_common_prefix_fn(iterable: Iterable[Text]
                               ) -> Callable[[Text], Text]:
  """Obtains a function that removes common prefix.

  Determines if all items in iterable start with the same substring (up to and
  including the first '/'). If so, returns a function str->str that removes
  the prefix of matching length. Otherwise returns identity function.

  Args:
    iterable: strings to process.

  Returns:
    A function that removes the common prefix from a string.
  """
  try:
    first = next(iter(iterable))
  except StopIteration:
    return lambda x: x
  separator_index = first.find('/')
  if separator_index == -1:
    return lambda x: x
  prefix = first[:separator_index + 1]
  if not all(k.startswith(prefix) for k in iterable):
    return lambda x: x
  return lambda item: item[len(prefix):]


def _compute_alive_counts(
    alive_vectors: Dict[Text, Sequence[bool]]) -> Dict[Text, int]:
  """Computes alive counts.

  Args:
    alive_vectors: A mapping from op_name to a vector where each element
      indicates whether the corresponding output activation is alive.

  Returns:
    Mapping from op_name to the number of its alive output activations.
  """
  return {
      op_name: int(np.sum(alive_vector))
      for op_name, alive_vector in alive_vectors.items()
  }


def format_structure(structure: Dict[Text, int]) -> Text:
  return json.dumps(structure, indent=2, sort_keys=True, default=str)
