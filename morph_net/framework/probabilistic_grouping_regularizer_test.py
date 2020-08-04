"""Tests for framework.probabilistic_grouping_regularizer."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from morph_net.framework import probabilistic_grouping_regularizer as pgr
from morph_net.testing import op_regularizer_stub
import tensorflow.compat.v1 as tf


class ProbabilisticGroupingRegularizerTest(tf.test.TestCase):

  def testProbGroupingRegularizer(self):
    reg_vec1 = [0.1, 0.3, 0.6, 0.2]
    alive_vec1 = [False, True, True, False]
    reg_vec2 = [0.2, 0.4, 0.5, 0.1]
    alive_vec2 = [False, True, False, True]
    reg_vec3 = [0.3, 0.2, 0.0, 0.25]
    alive_vec3 = [False, True, False, True]

    reg1 = op_regularizer_stub.OpRegularizerStub(reg_vec1,
                                                 alive_vec1)
    reg2 = op_regularizer_stub.OpRegularizerStub(reg_vec2,
                                                 alive_vec2)
    reg3 = op_regularizer_stub.OpRegularizerStub(reg_vec3,
                                                 alive_vec3)

    for reg in [reg1, reg2, reg3]:
      reg.is_probabilistic = True

    expected_grouped_reg = [0.496, 0.664, 0.8, 0.46]

    group_reg = pgr.ProbabilisticGroupingRegularizer(
        [reg1, reg2, reg3])
    with self.cached_session():
      self.assertAllEqual(
          [x or y or z for x, y, z in zip(
              alive_vec1, alive_vec2, alive_vec3)],
          group_reg.alive_vector.eval())
      self.assertAllClose(
          expected_grouped_reg,
          group_reg.regularization_vector.eval(), 1e-5)


if __name__ == '__main__':
  tf.test.main()
