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

"""Tests for StandardRLProgram's Trajectory Store construction and shutdown."""

import tempfile
from unittest import mock

from absl.testing import absltest
from etils import epath
from tunix.experimental.orchestrator import algorithm_adapter
from tunix.experimental.orchestrator import batch_assembly
from tunix.experimental.orchestrator import rl_program
from tunix.experimental.trajectory import config as trajectory_config_lib
from tunix.experimental.trajectory import file_store
from tunix.experimental.trajectory import trajectory_testing


class StandardRLProgramTrajectoryStoreTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    self.mock_algo = mock.MagicMock(spec=algorithm_adapter.AlgorithmAdapter)
    self.mock_algo.group_size = 2
    self.mock_algo.mini_batch_size = 1
    self.mock_algo.max_packed_len = 16
    self.assembler = batch_assembly.SequencePackedBatchAssembler(
        max_packed_len=16
    )

  def _create_program(self, **kwargs) -> rl_program.StandardRLProgram:
    return rl_program.StandardRLProgram(
        dataset=["prompt_0"],
        algo=self.mock_algo,
        reward_fns=[lambda x: 1.0],
        assembler=self.assembler,
        **kwargs,
    )

  def test_no_trajectory_store_config_means_no_store(self):
    program = self._create_program()
    self.assertIsNone(program._trajectory_store)  # pylint: disable=protected-access
    program.close()

  def test_disabled_trajectory_store_config_means_no_store(self):
    program = self._create_program(
        trajectory_store_config=trajectory_config_lib.TrajectoryStoreConfig(
            enabled=False
        )
    )
    self.assertIsNone(program._trajectory_store)  # pylint: disable=protected-access
    program.close()

  def test_enabled_file_backend_builds_store_once(self):
    tmp_dir = epath.Path(self.enter_context(tempfile.TemporaryDirectory()))
    program = self._create_program(
        trajectory_store_config=trajectory_config_lib.TrajectoryStoreConfig(
            enabled=True,
            backend="file",
            root_dir=str(tmp_dir),
            run_id="orchestrator_run",
        )
    )
    self.assertIsInstance(
        program._trajectory_store, file_store.FileTrajectoryStore  # pylint: disable=protected-access
    )
    program.close()

  def test_close_closes_the_store(self):
    tmp_dir = epath.Path(self.enter_context(tempfile.TemporaryDirectory()))
    program = self._create_program(
        trajectory_store_config=trajectory_config_lib.TrajectoryStoreConfig(
            enabled=True,
            backend="file",
            root_dir=str(tmp_dir),
            run_id="orchestrator_run",
        )
    )
    store = program._trajectory_store  # pylint: disable=protected-access
    program.close()
    with self.assertRaises(RuntimeError):
      store.add_step(
          trajectory_testing.STEP_1_1, trajectory_testing.METADATA_1
      )

  def test_close_without_a_store_does_not_raise(self):
    program = self._create_program()
    program.close()


if __name__ == "__main__":
  absltest.main()
