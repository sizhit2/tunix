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

from absl import logging
from tunix.experimental.trajectory import config as config_lib
from tunix.experimental.trajectory import file_store
from tunix.experimental.trajectory import in_memory_store
from tunix.experimental.trajectory import store


def build_trajectory_store(
    config: config_lib.TrajectoryStoreConfig | None,
    *,
    owner: str = "unknown",
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
    owner: Identifies the calling process in the log line below (e.g.
      "orchestrator", or a worker id). absl log lines carry a thread id but
      no process, host or container identity, so without this there is no way
      to attribute a reported run_id to a process wherever several of them
      share one log stream — as rollout workers under a single launcher
      process do.

  Returns:
    A `FileTrajectoryStore`, an `InMemoryTrajectoryStore`, or None when the
    store is disabled.
  """
  if config is None or not config.enabled:
    return None
  if config.backend == "file":
    built = file_store.FileTrajectoryStore(
        root_dir=config.root_dir, run_id=config.run_id
    )
    # A run_id that differs between the orchestrator and its workers splits
    # one run across separate directory trees, which otherwise surfaces only
    # as an empty read on the orchestrator side. Logging the resolved path on
    # every process turns that into one grep.
    logging.info(
        "[trajectory-store] owner=%s backend=file path=%s", owner, built.root_dir
    )
    return built
  logging.info(
      "[trajectory-store] owner=%s backend=memory (process-local; not visible"
      " to any other process)",
      owner,
  )
  return in_memory_store.InMemoryTrajectoryStore()
