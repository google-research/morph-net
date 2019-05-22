"""Helper module for calculating and saving learned structures.

TODO(e1)
Ops for exporting OpRegularizer values to json module.

When training with a network regularizer, the emerging structure of the
network is encoded in the `alive_vector`s and `regularization_vector`s of the
`OpRegularizers` of the ops in the graph. This module offers a way to create ops
that save those vectors as json files during the training.
"""

from __future__ import absolute_import
from __future__ import division
# [internal] enable type annotations
from __future__ import print_function

import json
import os
from enum import Enum
from morph_net.framework import op_regularizer_manager as orm
from morph_net.tools.ops import gen_json_tensor_exporter_op_py
import numpy as np
import tensorflow as tf
from typing import Text, Sequence, Dict, Optional, IO, Iterable, Callable


_SUPPORTED_OPS = ['Conv2D', 'Conv2DBackpropInput']
_ALIVE_FILENAME = 'alive'
_REG_FILENAME = 'regularizer'


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
    # TODO(p1): Consider deleting unused `remove_common_prefix` b/133261798.
    self._tensors = {}  # type: Dict[Text, tf.Tensor]
    self._alive_vectors = None  # type: Optional[Dict[Text, Sequence[bool]]]
    rename_fn = get_remove_common_prefix_fn(
        self._tensors) if remove_common_prefix else lambda x: x

    for op in op_regularizer_manager.ops:
      if op.type not in _SUPPORTED_OPS:
        continue

      opreg = op_regularizer_manager.get_regularizer(op)
      if not opreg:
        tf.logging.warning('No regularizer found for: %s', op.name)
        continue

      self._tensors[rename_fn(op.name)] = tf.cast(opreg.alive_vector, tf.int32)

  @property
  def tensors(self):
    """A dictionary between op names and alive vectors.

    Alive vectors are `tf.Tensor`s of type tf.int32.

    Returns:
      Dict: op name -> alive vector tensor
    """
    # TODO(p1): Rename tensors to something better. tensors is a dict!
    return self._tensors

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
    return _compute_alive_counts(self._alive_vectors)

  def save_alive_counts(self, f: IO[bytes]) -> None:
    """Saves live counts to a file.

    Args:
      f: a file object where alive counts are saved.
    """
    f.write(format_structure(self.get_alive_counts()))

  def create_file_and_save_alive_counts(self, base_dir: Text,
                                        global_step: int) -> None:
    """Creates and updates files with alive counts.

    Creates the directory `{base_dir}/learned_structure/` and saves the current
    alive counts to:
      `{base_dir}/learned_structure/{_ALIVE_FILENAME}_{global_step}`
    and overwrites:
      `{base_dir}/learned_structure/{_ALIVE_FILENAME}`.

    Args:
      base_dir: where to export the alive counts.
      global_step: current value of global step, used as a suffix in filename.
    """
    current_filename = '%s_%s' % (_ALIVE_FILENAME, global_step)
    directory = os.path.join(base_dir, 'learned_structure')
    try:
      tf.gfile.MakeDirs(directory)
    except tf.errors.OpError:
      # Probably already exists. If not, we'll see the error in the next line.
      pass
    with tf.gfile.Open(os.path.join(directory, current_filename), 'w') as f:
      self.save_alive_counts(f)  # pytype: disable=wrong-arg-types
    with tf.gfile.Open(os.path.join(directory, _ALIVE_FILENAME), 'w') as f:
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


class ExportInfo(Enum):
  """ExportInfo for selecting file to be exported."""
  # Export alive count of op.
  alive = 'alive'
  # Export regularization vector of op.
  regularization = 'regularization'
  # Export both alive count and regularization vector.
  both = 'both'


class StructureExporterOp(object):
  """Manages the export of the alive and regularization json files."""

  def __init__(self,
               directory,
               save,
               opreg_manager,
               alive_file=_ALIVE_FILENAME,
               regularization_file=_REG_FILENAME):
    """Init an object with all the exporter vars.

    Args:
      directory: A string, directory to write the json files to.
      save: A scalar `tf.Tensor` of type boolean. If `False`, the exporting is a
        no-op.
      opreg_manager: An OpRegularizerManager that manages the OpRegularizers.
      alive_file: A string with a file name that will contain the alive counts
        (in json format).
      regularization_file: A string with a file name that will contain the
        regularization vectors (in json format).
    """
    self._save = save
    self._opreg_manager = opreg_manager
    self._alive_file = os.path.join(directory, alive_file)
    self._regularization_file = os.path.join(directory, regularization_file)

  def export(self, info=ExportInfo.both):
    """Returns an `tf.Operation` that saves the ExportInfo in json files.

    Args:
      info: An 'ExportInfo' enum that defines the data to be exported.
    Returns:
      A `tf.Operation` that executes the exporting.
    Raises:
      ValueError: If info is not 'ExportInfo' enum.
    """
    if not isinstance(info, ExportInfo):
      raise ValueError('`info` must be an ExportInfo enum value.')

    op = None
    if info == ExportInfo.regularization or info == ExportInfo.both:
      op = self._export_helper(
          self._regularization_file, lambda x: x.regularization_vector,
          'ExportRegularization')
    if info == ExportInfo.alive or info == ExportInfo.both:
      alive_op = self._export_helper(self._alive_file, _alive_count,
                                     'ExportAlive')
      op = alive_op if op is None else tf.group(op, alive_op)
    return op

  def export_state_every_n(self, n, info=ExportInfo.both):
    """Returns an `tf.Operation` that saves the ExportInfo every `n` steps.

    Args:
      n: An integer. Actual export will happen once in every `n` calls, all
        other calls will be no-ops.
      info: an 'ExportInfo' enum that defined what data to export.

    Returns:
      A `tf.Operation` that executes the export.
    """
    with tf.name_scope('ExportEveryN'):
      counter = tf.get_variable(
          'counter', dtype=tf.int32, initializer=-1, trainable=False)
      counter = counter.assign_add(1)

      return tf.cond(
          tf.equal(counter % n, 0),
          lambda: self.export(info),
          lambda: tf.no_op(name='DontSave')).op

  def _export_helper(self, filename, getter, name):
    """Helper function for OpRegularizer state vectors as JSON.

    Args:
      filename: A string with the filename to save.
      getter: A single-argument function, which receives an OpRegularizer object
        and returns the value that needs to be exported (alive count or
        regularization vector).
      name: Name for the StructureExporterOp op.

    Returns:
      An op that exports the state if `save` evaluates to `True`.
    """
    op_to_reg = {
        op: getter(self._opreg_manager.get_regularizer(op))
        for op in self._opreg_manager.ops
        if op.type in _SUPPORTED_OPS and self._opreg_manager.get_regularizer(op)
    }

    # Sort by name, to make the exported file more easily human-readable.
    sorted_ops = sorted(op_to_reg.keys(), key=lambda x: x.name)
    keys = [op.name for op in sorted_ops]
    values = [op_to_reg[op] for op in sorted_ops]
    return gen_json_tensor_exporter_op_py.json_tensor_exporter(
        filename=filename, keys=keys, values=values, save=self._save, name=name)


def format_structure(structure: Dict[Text, int]) -> Text:
  return json.dumps(structure, indent=2, sort_keys=True, default=str)


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


def _alive_count(op_regularizer):
  return tf.reduce_sum(tf.cast(op_regularizer.alive_vector, tf.int32))
