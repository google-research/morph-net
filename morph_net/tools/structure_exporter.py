"""Helper module for calculating the live activation counts."""

from __future__ import absolute_import
from __future__ import division
# [internal] enable type annotations
from __future__ import print_function

import json
import os
from morph_net.framework import op_regularizer_manager as orm
import numpy as np
import tensorflow as tf
from typing import Text, Sequence, Dict, Optional, IO, Iterable, Callable

_SUPPORTED_OPS = ['Conv2D']
_ALIVE_FILENAME = 'alive'


def compute_alive_counts(
    alive_vectors: Dict[Text, Sequence[bool]]) -> Dict[Text, int]:
  """Computes alive counts.

  Args:
    alive_vectors: A mapping from op_name to a vector where each element says
      whether the corresponding output activation is alive.

  Returns:
    Mapping from op_name to the number of its alive output activations.
  """
  return {
      op_name: int(np.sum(alive_vector))
      for op_name, alive_vector in alive_vectors.items()
  }


class StructureExporter(object):
  """Reports statistics about the current state of regularization.

  Obtains live activation counts for supported ops: a map from each op name
  to its count of alive activations (filters). Optionally, thresholds the counts
  so that very low counts are reported as 0. Currently, only supports Conv2D.
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
    self._op_regularizer_manager = op_regularizer_manager
    self._alive_tensors = {}  # type: Dict[Text, tf.Tensor]
    self._alive_vectors = None  # type: Optional[Dict[Text, Sequence[bool]]]

    for op in self._op_regularizer_manager.ops:
      if op.type not in _SUPPORTED_OPS:
        continue
      opreg = self._op_regularizer_manager.get_regularizer(op)
      if opreg:
        # TODO(p1): use bool here (no cast), and then convert later?
        self._alive_tensors[op.name] = tf.cast(opreg.alive_vector, tf.int32)
      else:
        tf.logging.warning('No regularizer found for: %s', op.name)

    if remove_common_prefix:
      rename_op = get_remove_common_prefix_op(self._alive_tensors)
      self._alive_tensors = {
          rename_op(k): v for k, v in self._alive_tensors.items()
      }

  @property
  def tensors(self):
    """The list of tensors required to compute statistics.

    Returns:
      Dict: op name -> alive vector tensor
    """
    return self._alive_tensors

  def populate_tensor_values(self, values: Dict[Text, Sequence[bool]]) -> None:
    # TODO(p1): make this a hierarchy with 'alive_vectors' key at the top
    assert sorted(values) == sorted(self.tensors)
    self._alive_vectors = values

  def get_alive_counts(self) -> Dict[Text, int]:
    """Computes alive counts.

    populate_tensor_values() must have been called earlier.

    Returns:
      A dict {op_name: alive_count}, alive_count is a scalar integer tf.Tensor.

    Raises:
      RuntimeError: tensor values not populated.
    """

    if self._alive_vectors is None:
      raise RuntimeError('Tensor values not populated.')
    # TODO(p1): consider warning if same values are used twice?
    return compute_alive_counts(self._alive_vectors)

  def save_alive_counts(self, f: IO[bytes]) -> None:
    """Saves live counts to a file.

    Args:
      f: a file object where alive counts are saved.
    """
    f.write(
        json.dumps(
            self.get_alive_counts(), indent=2, sort_keys=True, default=str))

  def create_file_and_save_alive_counts(self, train_dir: Text,
                                        global_step: tf.Tensor) -> None:
    """Creates a file and saves live counts to it.

    Creates the directory {train_dir}/learned_structure/ and saves the current
    alive counts to {path}/{_ALIVE_FILENAME}_{global_step} and overwrites
    {path}/{_ALIVE_FILENAME}.

    Args:
      train_dir: where to export the alive counts.
      global_step: current value of global step, used as a suffix in filename.
    """
    current_filename = '%s_%s' % (_ALIVE_FILENAME, global_step)
    directory = os.path.join(train_dir, 'learned_structure')
    try:
      tf.gfile.MkDir(directory)
    except tf.errors.OpError:
      # Probably already exists. If not, we'll see the error in the next line.
      pass
    with tf.gfile.Open(os.path.join(directory, current_filename), 'w') as f:
      self.save_alive_counts(f)
    with tf.gfile.Open(os.path.join(directory, _ALIVE_FILENAME), 'w') as f:
      self.save_alive_counts(f)


# TODO(p1): maybe check that we still end up with unique names after prefix
# removal, and do nothing if that's not the case?
def get_remove_common_prefix_op(
    iterable: Iterable[Text]) -> Callable[[Text], Text]:
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
