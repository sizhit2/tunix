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

"""Tests for ClusterOrchestrator's Trajectory Store construction and shutdown."""

import tempfile
from unittest import mock

from absl.testing import absltest
from etils import epath
from tunix.experimental.orchestrator import orchestrator
from tunix.experimental.trajectory import config as trajectory_config_lib
from tunix.experimental.trajectory import file_store
from tunix.experimental.trajectory import trajectory_testing


def _orchestrator(**kwargs) -> orchestrator.ClusterOrchestrator:
  mock_registry = mock.MagicMock()
  mock_registry.worker_ids.return_value = []
  mock_registry.infos.return_value = []
  mock_registry.group.return_value.members.return_value = []
  return orchestrator.ClusterOrchestrator(
      registry=mock_registry,
      lifecycle_driver=mock.MagicMock(),
      monitor=mock.MagicMock(),
      **kwargs,
  )


class ClusterOrchestratorTrajectoryStoreTest(absltest.TestCase):

  def test_no_config_means_no_store(self):
    orch = _orchestrator()
    self.assertIsNone(orch.trajectory_store)
    orch.shutdown()

  def test_disabled_config_means_no_store(self):
    orch = _orchestrator(
        trajectory_store_config=trajectory_config_lib.TrajectoryStoreConfig(
            enabled=False
        )
    )
    self.assertIsNone(orch.trajectory_store)
    orch.shutdown()

  def test_enabled_file_backend_builds_store_once(self):
    tmp_dir = epath.Path(self.enter_context(tempfile.TemporaryDirectory()))
    orch = _orchestrator(
        trajectory_store_config=trajectory_config_lib.TrajectoryStoreConfig(
            enabled=True,
            backend="file",
            root_dir=str(tmp_dir),
            run_id="cluster_run",
        )
    )
    self.assertIsInstance(orch.trajectory_store, file_store.FileTrajectoryStore)
    orch.shutdown()

  def test_shutdown_closes_the_store(self):
    tmp_dir = epath.Path(self.enter_context(tempfile.TemporaryDirectory()))
    orch = _orchestrator(
        trajectory_store_config=trajectory_config_lib.TrajectoryStoreConfig(
            enabled=True,
            backend="file",
            root_dir=str(tmp_dir),
            run_id="cluster_run",
        )
    )
    store = orch.trajectory_store
    orch.shutdown()
    with self.assertRaises(RuntimeError):
      store.add_step(
          trajectory_testing.STEP_1_1, trajectory_testing.METADATA_1
      )

  def test_shutdown_closes_the_store_even_when_a_prior_step_raises(self):
    orch = _orchestrator()
    orch.trajectory_store = mock.MagicMock()
    orch.lifecycle_driver.shutdown = mock.MagicMock(
        side_effect=RuntimeError("lifecycle shutdown failed")
    )
    with self.assertRaises(RuntimeError):
      orch.shutdown()
    orch.trajectory_store.close.assert_called_once()

  def test_shutdown_without_a_store_does_not_raise(self):
    orch = _orchestrator()
    orch.shutdown()

  def test_run_threads_the_orchestrators_store_into_an_auto_built_program(
      self,
  ):
    # Tier 1 (`run()`, no `program=` supplied): the auto-constructed
    # StandardRLProgram should receive the orchestrator's own instance, not
    # build a second one of its own.
    orch = _orchestrator(
        trajectory_store_config=trajectory_config_lib.TrajectoryStoreConfig(
            enabled=True, backend="memory"
        )
    )
    orch.engine = mock.MagicMock()
    orch.bring_up_workers = mock.MagicMock()

    with mock.patch.object(
        orchestrator.rl_program, "StandardRLProgram"
    ) as mock_program_cls:
      orch.run(
          algo=mock.MagicMock(max_packed_len=16),
          dataset=["prompt_0"],
          reward_fns=[lambda x: 1.0],
      )

    self.assertIs(
        mock_program_cls.call_args.kwargs["trajectory_store"],
        orch.trajectory_store,
    )


if __name__ == "__main__":
  absltest.main()
