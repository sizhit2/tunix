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

"""Builds the Trajectory Store instance a process should use, or None."""

from tunix.experimental.trajectory import config as config_lib
from tunix.experimental.trajectory import file_store
from tunix.experimental.trajectory import in_memory_store
from tunix.experimental.trajectory import store


def build_trajectory_store(
    config: config_lib.TrajectoryStoreConfig | None,
) -> store.TrajectoryReader | store.TrajectoryWriter | None:
  """Builds this process's Trajectory Store, or None if disabled.

  Call once per process — e.g. in the `__init__` of whatever owns that
  process's lifecycle — and hold onto the result. Calling this more than
  once per process would construct a second, independent store (and, for the
  file backend, a second `AsyncFileWriter` background thread); nothing here
  prevents that, since the guard is simply that callers construct it exactly
  once.

  Args:
    config: The store configuration, or None to disable.

  Returns:
    A `FileTrajectoryStore`, an `InMemoryTrajectoryStore`, or None when the
    store is disabled.
  """
  if config is None or not config.enabled:
    return None
  if config.backend == "file":
    return file_store.FileTrajectoryStore(
        root_dir=config.root_dir, run_id=config.run_id
    )
  return in_memory_store.InMemoryTrajectoryStore()
