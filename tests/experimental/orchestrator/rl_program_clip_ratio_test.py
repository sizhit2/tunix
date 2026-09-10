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

"""generation/completions/clip_ratio in StandardRLProgram's step metrics."""

import tempfile
from unittest import mock

from absl.testing import absltest
import numpy as np
from tunix.experimental.common import datatypes
from tunix.experimental.orchestrator import rl_program
from tunix.sft import metrics_logger as metrics_logger_lib

EOS = 7
MAX_RESP = 4


def _item(tokens):
  return datatypes.TrajectoryItem(
      prompt_id="p",
      group_index=0,
      start_step=0,
      traj=datatypes.Trajectory(reward=1.0),
      prompt_tokens=np.array([1, 2], dtype=np.int32),
      completion_tokens=np.array(tokens, dtype=np.int32),
  )


class ClipRatioTest(absltest.TestCase):

  def _program(self, eos_ids):
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
        eos_ids=eos_ids,
        metrics_logging_options=metrics_logger_lib.MetricsLoggerOptions(
            log_dir=tempfile.mkdtemp(), flush_every_n_steps=1
        ),
    )
    prog.engine = mock.MagicMock()
    return prog

  def _clip_ratio(self, prog, items):
    prog._collect_and_log_step_metrics(
        all_step_items=items,
        step_rewards=[1.0] * len(items),
        num_rollouts=len(items),
        num_microbatches=1,
        step_time_sec=1.0,
        consumed_policy_version=0,
        log_step=0,
    )
    return prog.metrics_logger.get_metric(
        "", "generation/completions/clip_ratio", "train"
    )

  def test_budget_exhausted_without_eos_counts_as_clipped(self):
    prog = self._program(eos_ids=[EOS])
    items = [
        _item([3, 4, 5, 6]),  # full budget, no stop token -> clipped
        _item([3, 4, 5, EOS]),  # full budget but ended on eos -> not clipped
        _item([3, 4]),  # short -> not clipped
        _item([3, 4, 5, 6]),  # clipped
    ]
    self.assertAlmostEqual(self._clip_ratio(prog, items), 0.5)

  def test_without_eos_ids_length_alone_decides(self):
    prog = self._program(eos_ids=None)
    items = [_item([3, 4, 5, EOS]), _item([3, 4])]
    self.assertAlmostEqual(self._clip_ratio(prog, items), 0.5)


if __name__ == "__main__":
  absltest.main()
