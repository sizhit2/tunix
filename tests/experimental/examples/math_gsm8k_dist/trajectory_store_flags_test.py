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

"""Tests for the GSM8K demo's shared Trajectory Store flags."""

import argparse
import tempfile

from absl.testing import absltest
from etils import epath
from tunix.experimental.examples.math_gsm8k_dist import trajectory_store_flags
from tunix.experimental.trajectory import factory
from tunix.experimental.worker import rollout_worker as rollout_worker_lib


def _parse(argv: list[str]) -> argparse.Namespace:
  parser = argparse.ArgumentParser()
  trajectory_store_flags.add_arguments(parser)
  return parser.parse_args(argv)


class TrajectoryStoreFlagsTest(absltest.TestCase):

  def test_disabled_by_default(self):
    args = _parse([])
    self.assertFalse(args.trajectory_store_enabled)
    self.assertIsNone(trajectory_store_flags.config_from_args(args))

  def test_enabled_flags_build_a_config(self):
    args = _parse([
        "--trajectory_store_enabled",
        "--trajectory_store_root_dir=/tmp/traj",
        "--trajectory_store_run_id=run_abc",
    ])
    config = trajectory_store_flags.config_from_args(args)
    self.assertTrue(config.enabled)
    self.assertEqual(config.backend, "file")
    self.assertEqual(config.root_dir, "/tmp/traj")
    self.assertEqual(config.run_id, "run_abc")

  def test_enabled_without_run_id_fails_loudly(self):
    # The whole point of requiring it: a missing run_id must surface here,
    # at startup, not as silently overwritten trajectories later.
    args = _parse([
        "--trajectory_store_enabled",
        "--trajectory_store_root_dir=/tmp/traj",
    ])
    with self.assertRaisesRegex(ValueError, "run_id"):
      trajectory_store_flags.config_from_args(args)

  def test_memory_backend_needs_neither_root_dir_nor_run_id(self):
    args = _parse([
        "--trajectory_store_enabled",
        "--trajectory_store_backend=memory",
    ])
    config = trajectory_store_flags.config_from_args(args)
    self.assertEqual(config.backend, "memory")

  def test_run_id_env_var_default(self):
    # launcher.sh exports the run id it generated; the flags pick it up so a
    # caller that omits the flag still lands on the run's shared id.
    with absltest.mock.patch.dict(
        "os.environ", {"TRAJECTORY_STORE_RUN_ID": "run_from_env"}
    ):
      parser = argparse.ArgumentParser()
      trajectory_store_flags.add_arguments(parser)
      args = parser.parse_args(["--trajectory_store_enabled"])
    self.assertEqual(args.trajectory_store_run_id, "run_from_env")


class SharedRunIdReachesBothSidesTest(absltest.TestCase):
  """The orchestrator and a rollout worker must land on the same directory."""

  def test_same_flags_resolve_to_one_shared_path(self):
    tmp_dir = epath.Path(self.enter_context(tempfile.TemporaryDirectory()))
    argv = [
        "--trajectory_store_enabled",
        f"--trajectory_store_root_dir={tmp_dir}",
        "--trajectory_store_run_id=shared_run",
    ]

    # The orchestrator process builds its store straight from the config.
    orchestrator_store = factory.build_trajectory_store(
        trajectory_store_flags.config_from_args(_parse(argv)),
        owner="orchestrator",
    )
    # A rollout worker process carries the same config through RolloutConfig.
    worker_config = rollout_worker_lib.RolloutConfig(
        trajectory_store_config=trajectory_store_flags.config_from_args(
            _parse(argv)
        )
    )
    worker_store = factory.build_trajectory_store(
        worker_config.trajectory_store_config, owner="rollout-0"
    )

    try:
      self.assertEqual(
          orchestrator_store.root_dir, worker_store.root_dir
      )
      self.assertEqual(
          orchestrator_store.root_dir, tmp_dir / "shared_run"
      )
      # Separate instances, one shared path — no cross-process object.
      self.assertIsNot(orchestrator_store, worker_store)
    finally:
      orchestrator_store.close()
      worker_store.close()

  def test_a_worker_writing_is_readable_by_the_orchestrator(self):
    # End-to-end for the read/write split: the worker writes, and the
    # orchestrator reading the same root_dir/run_id sees it.
    tmp_dir = epath.Path(self.enter_context(tempfile.TemporaryDirectory()))
    argv = [
        "--trajectory_store_enabled",
        f"--trajectory_store_root_dir={tmp_dir}",
        "--trajectory_store_run_id=shared_run",
    ]
    from tunix.experimental.trajectory import trajectory_testing  # pylint: disable=g-import-not-at-top

    worker_store = factory.build_trajectory_store(
        trajectory_store_flags.config_from_args(_parse(argv)), owner="rollout-0"
    )
    worker_store.add_step(
        trajectory_testing.STEP_1_1, trajectory_testing.METADATA_1
    )
    worker_store.flush()

    orchestrator_store = factory.build_trajectory_store(
        trajectory_store_flags.config_from_args(_parse(argv)),
        owner="orchestrator",
    )
    try:
      metas = orchestrator_store.get_trajectories_metadata()
      self.assertEqual(
          [m.trajectory_id for m in metas],
          [trajectory_testing.TRAJECTORY_ID_1],
      )
    finally:
      worker_store.close()
      orchestrator_store.close()


if __name__ == "__main__":
  absltest.main()
