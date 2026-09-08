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

"""Tests for TrajectoryStoreConfig validation."""

from absl.testing import absltest
from tunix.experimental.trajectory import config as config_lib


class TrajectoryStoreConfigTest(absltest.TestCase):

  def test_disabled_by_default(self):
    config = config_lib.TrajectoryStoreConfig()
    self.assertFalse(config.enabled)

  def test_disabled_config_skips_validation(self):
    # Nonsensical field values are fine as long as the store is off: nothing
    # will ever construct a store from this config.
    config_lib.TrajectoryStoreConfig(enabled=False, backend="bogus")

  def test_enabled_file_backend_requires_root_dir(self):
    with self.assertRaisesRegex(ValueError, "root_dir"):
      config_lib.TrajectoryStoreConfig(enabled=True, backend="file")

  def test_enabled_file_backend_with_root_dir_is_valid(self):
    config = config_lib.TrajectoryStoreConfig(
        enabled=True, backend="file", root_dir="/tmp/traj"
    )
    self.assertEqual(config.root_dir, "/tmp/traj")

  def test_enabled_memory_backend_does_not_require_root_dir(self):
    config = config_lib.TrajectoryStoreConfig(enabled=True, backend="memory")
    self.assertIsNone(config.root_dir)

  def test_memory_backend_with_resume_on_restart_rejected(self):
    with self.assertRaisesRegex(ValueError, "resume_on_restart"):
      config_lib.TrajectoryStoreConfig(
          enabled=True, backend="memory", resume_on_restart=True
      )

  def test_file_backend_with_resume_on_restart_is_valid(self):
    config = config_lib.TrajectoryStoreConfig(
        enabled=True,
        backend="file",
        root_dir="/tmp/traj",
        resume_on_restart=True,
    )
    self.assertTrue(config.resume_on_restart)

  def test_unknown_backend_rejected(self):
    with self.assertRaisesRegex(ValueError, "backend"):
      config_lib.TrajectoryStoreConfig(enabled=True, backend="bogus")


if __name__ == "__main__":
  absltest.main()
