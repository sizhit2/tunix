# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""generation/completions/clip_ratio comes from the collect engine's status."""

import tempfile
from unittest import mock

from absl.testing import absltest
import numpy as np
from tunix.experimental.common import datatypes
from tunix.experimental.orchestrator import distributed_rl_engine
from tunix.experimental.orchestrator import rl_program
from tunix.sft import metrics_logger as metrics_logger_lib

MAX_RESP = 4
Status = datatypes.TrajectoryStatus


def _item(tokens, status):
  return datatypes.TrajectoryItem(
      prompt_id="p",
      group_index=0,
      start_step=0,
      traj=datatypes.Trajectory(reward=1.0, status=status),
      prompt_tokens=np.array([1, 2], dtype=np.int32),
      completion_tokens=np.array(tokens, dtype=np.int32),
  )


class ClipRatioTest(absltest.TestCase):

  def _program(self):
    algo = mock.MagicMock()
    algo.group_size = 2
    algo.mini_batch_size = 1
    algo.max_turns = 1
    algo.max_packed_len = 16
    algo.train_micro_batch_size = 1
    algo.max_response_length = MAX_RESP
    algo.requires_reference_kl = False
    prog = rl_program.StandardRLProgram(
        algo=algo,
        generation_args=datatypes.GenerationArgs(max_response_length=MAX_RESP),
        metrics_logging_options=metrics_logger_lib.MetricsLoggerOptions(
            log_dir=tempfile.mkdtemp(), flush_every_n_steps=1
        ),
    )
    prog.engine = mock.MagicMock()
    return prog

  def _metric(self, prog, items, name):
    prog._collect_and_log_step_metrics(
        all_step_items=items,
        step_rewards=[1.0] * len(items),
        num_rollouts=len(items),
        num_microbatches=1,
        step_time_sec=1.0,
        consumed_policy_version=0,
        log_step=0,
    )
    return prog.metrics_logger.get_metric("", name, "train")

  def test_clip_ratio_is_the_context_limit_status_rate(self):
    prog = self._program()
    items = [
        # Budget exhausted: the collect engine flags it, whatever the tokens.
        _item([3, 4, 5, 6], Status.MAX_CONTEXT_LIMIT_REACHED),
        # Full budget but the engine says it finished cleanly -> not clipped.
        _item([3, 4, 5, 6], Status.SUCCEEDED),
        _item([3, 4], Status.SUCCEEDED),
        _item([3, 4, 5, 6], Status.MAX_CONTEXT_LIMIT_REACHED),
    ]
    self.assertAlmostEqual(
        self._metric(prog, items, "generation/completions/clip_ratio"), 0.5
    )

  def test_other_terminal_statuses_are_not_clipped(self):
    prog = self._program()
    items = [_item([3], Status.TIMEOUT), _item([3], Status.FAILED)]
    self.assertAlmostEqual(
        self._metric(prog, items, "generation/completions/clip_ratio"), 0.0
    )
    self.assertAlmostEqual(
        self._metric(prog, items, "rollout/success_rate"), 0.0
    )

  def test_engine_keeps_the_collect_engine_status_name(self):
    f = distributed_rl_engine._trajectory_status_from_name
    self.assertEqual(f("MAX_CONTEXT_LIMIT_REACHED"), Status.MAX_CONTEXT_LIMIT_REACHED)
    self.assertEqual(f("COMPLETED"), Status.SUCCEEDED)
    self.assertEqual(f("SUCCEEDED"), Status.SUCCEEDED)
    self.assertEqual(f("ERROR"), Status.FAILED)
    self.assertEqual(f(None), Status.FAILED)


if __name__ == "__main__":
  absltest.main()
