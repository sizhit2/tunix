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

"""Tests that run_rollout_node threads the store config into its worker.

Exercises the vanilla sampler path, which needs no vLLM — only the model,
mesh and sampler adapter are mocked out. Everything else (the parser,
RolloutConfig, RolloutWorker, and the store they build) is the real code.
"""

import tempfile
from unittest import mock

from absl.testing import absltest
from etils import epath
from tunix.experimental.examples.math_gsm8k_dist import run_rollout_node
from tunix.experimental.trajectory import file_store


class VanillaRolloutNodeTrajectoryStoreTest(absltest.TestCase):

  def _build_worker(self, argv: list[str]):
    args = run_rollout_node._parse_args(argv)  # pylint: disable=protected-access
    with mock.patch.object(run_rollout_node, "_create_rollout_mesh"), (
        mock.patch.object(run_rollout_node, "models")
    ), mock.patch.object(
        run_rollout_node, "vanilla_sampler_adapter"
    ), mock.patch.object(
        run_rollout_node, "tokenizer_adapter_lib"
    ), mock.patch.object(
        run_rollout_node, "chat_parser_lib"
    ):
      return run_rollout_node._create_vanilla_worker(args, mock.MagicMock())  # pylint: disable=protected-access

  def test_vanilla_worker_builds_the_store_from_the_flags(self):
    tmp_dir = epath.Path(self.enter_context(tempfile.TemporaryDirectory()))
    worker = self._build_worker([
        "--sampler=vanilla",
        "--worker_id=rollout-0",
        "--trajectory_store_enabled",
        f"--trajectory_store_root_dir={tmp_dir}",
        "--trajectory_store_run_id=shared_run",
    ])
    store = worker._trajectory_store  # pylint: disable=protected-access
    try:
      self.assertIsInstance(store, file_store.FileTrajectoryStore)
      self.assertEqual(store.root_dir, tmp_dir / "shared_run")
    finally:
      worker.stop()

  def test_vanilla_worker_has_no_store_when_the_flag_is_absent(self):
    worker = self._build_worker(["--sampler=vanilla", "--worker_id=rollout-0"])
    self.assertIsNone(worker._trajectory_store)  # pylint: disable=protected-access
    worker.stop()

  def test_missing_run_id_fails_at_startup(self):
    # A worker launched with the store on but no run_id must fail loudly
    # here rather than write into an unscoped root_dir.
    tmp_dir = epath.Path(self.enter_context(tempfile.TemporaryDirectory()))
    with self.assertRaisesRegex(ValueError, "run_id"):
      self._build_worker([
          "--sampler=vanilla",
          "--trajectory_store_enabled",
          f"--trajectory_store_root_dir={tmp_dir}",
      ])


if __name__ == "__main__":
  absltest.main()
