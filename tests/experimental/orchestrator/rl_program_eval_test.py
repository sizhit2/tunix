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

"""Held-out evaluation in StandardRLProgram."""

import asyncio
import tempfile
from unittest import mock

from absl.testing import absltest
import numpy as np
from tunix.experimental.common import datatypes
from tunix.experimental.orchestrator import rl_program
from tunix.sft import metrics_logger as metrics_logger_lib

Status = datatypes.TrajectoryStatus


def _item(reward_text, status=Status.SUCCEEDED, n_tokens=3, env_reward=0.0):
  return datatypes.TrajectoryItem(
      prompt_id="p",
      group_index=0,
      start_step=0,
      traj=datatypes.Trajectory(reward=env_reward, status=status),
      prompt_tokens=np.array([1, 2], dtype=np.int32),
      completion_tokens=np.arange(n_tokens, dtype=np.int32),
      metadata={"answer": reward_text},
  )


class HoldoutEvalTest(absltest.TestCase):

  def _program(self, **kwargs):
    algo = mock.MagicMock()
    algo.group_size = 2
    algo.mini_batch_size = 1
    algo.max_turns = 1
    algo.max_packed_len = 16
    algo.train_micro_batch_size = 1
    algo.max_response_length = 8
    algo.requires_reference_kl = False
    prog = rl_program.StandardRLProgram(
        algo=algo,
        reward_fns=[lambda item: float(item.metadata["answer"])],
        generation_args=datatypes.GenerationArgs(max_response_length=8),
        metrics_logging_options=metrics_logger_lib.MetricsLoggerOptions(
            log_dir=tempfile.mkdtemp(), flush_every_n_steps=1
        ),
        **kwargs,
    )
    prog.engine = mock.MagicMock()
    return prog

  def test_eval_is_due_only_on_the_interval(self):
    prog = self._program(eval_dataset=["q"], eval_every_n_steps=5)
    # Step indices are 0-based and the check runs after the step finishes, so
    # the first eval lands after step 4, i.e. after five completed steps.
    due = [s for s in range(12) if prog._eval_is_due(s)]
    self.assertEqual(due, [4, 9])

  def test_eval_is_never_due_without_a_dataset_or_an_interval(self):
    self.assertFalse(
        self._program(eval_every_n_steps=1)._eval_is_due(0)
    )
    self.assertFalse(
        self._program(eval_dataset=["q"], eval_every_n_steps=0)._eval_is_due(0)
    )

  def test_eval_stage_scores_and_logs(self):
    prog = self._program(
        eval_dataset=["q1", "q2", "q3"], eval_every_n_steps=1
    )
    prog.engine.generate = mock.AsyncMock(
        return_value=[
            _item("1.0", n_tokens=4),
            _item("0.5", n_tokens=2),
            _item("0.0", Status.MAX_CONTEXT_LIMIT_REACHED, n_tokens=6),
        ]
    )
    metrics = asyncio.run(prog.eval_stage(log_step=7))

    self.assertEqual(metrics["eval/num_prompts"], 3.0)
    self.assertAlmostEqual(metrics["eval/rewards/mean"], 0.5)
    self.assertAlmostEqual(metrics["eval/rewards/max"], 1.0)
    # Only the 1.0 clears the format-only tier; 0.5 is the boundary, not above.
    self.assertAlmostEqual(metrics["eval/solve_ratio"], 1 / 3)
    self.assertAlmostEqual(metrics["eval/completions/mean_length"], 4.0)
    self.assertAlmostEqual(metrics["eval/completions/clip_ratio"], 1 / 3)
    self.assertEqual(
        prog.metrics_logger.get_metric("", "eval/rewards/mean", "train"), 0.5
    )

  def test_eval_falls_back_to_the_environment_reward(self):
    """`--reward_mode=env` leaves reward_fns empty and scores in the env."""
    prog = self._program(eval_dataset=["q1", "q2"], eval_every_n_steps=1)
    prog.reward_fns = []
    prog.engine.generate = mock.AsyncMock(
        return_value=[
            _item("ignored", env_reward=1.0),
            _item("ignored", env_reward=0.1),
        ]
    )
    metrics = asyncio.run(prog.eval_stage(log_step=0))

    self.assertAlmostEqual(metrics["eval/rewards/mean"], 0.55)
    self.assertAlmostEqual(metrics["eval/solve_ratio"], 0.5)

  def test_eval_stage_stamps_prompt_ids_and_honors_the_batch_cap(self):
    prog = self._program(
        eval_dataset=["q1", "q2", "q3"],
        eval_every_n_steps=1,
        eval_batch_size=2,
    )
    prog.engine.generate = mock.AsyncMock(return_value=[])
    asyncio.run(prog.eval_stage(log_step=0))

    sent = prog.engine.generate.await_args.args[0]
    self.assertEqual(
        sent,
        [
            {"prompt": "q1", "prompt_id": "eval_0"},
            {"prompt": "q2", "prompt_id": "eval_1"},
        ],
    )

  def test_eval_stage_passes_its_own_generation_args(self):
    greedy = datatypes.GenerationArgs(temperature=0.0, max_response_length=8)
    prog = self._program(
        eval_dataset=["q"],
        eval_every_n_steps=1,
        eval_generation_args=greedy,
    )
    prog.engine.generate = mock.AsyncMock(return_value=[])
    asyncio.run(prog.eval_stage(log_step=0))

    self.assertIs(
        prog.engine.generate.await_args.kwargs["generation_args"], greedy
    )

  def test_eval_stage_without_a_dataset_does_nothing(self):
    prog = self._program()
    prog.engine.generate = mock.AsyncMock(return_value=[])
    self.assertEqual(asyncio.run(prog.eval_stage(log_step=0)), {})
    prog.engine.generate.assert_not_awaited()


if __name__ == "__main__":
  absltest.main()
