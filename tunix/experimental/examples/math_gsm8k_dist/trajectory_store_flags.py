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

"""Shared Trajectory Store flags for the distributed GSM8K demo.

The orchestrator and every rollout worker run as separate processes with
separate argument parsers, and they only end up writing to and reading from
one store if all of them are handed the *same* `run_id`. Defining the flags
once here, rather than once per entry point, is what keeps them from drifting
apart — a `run_id` that reaches the orchestrator but not the workers splits
one run across separate directory trees and surfaces only as an empty read.

`launcher.sh` generates the run_id once and passes it to every process it
starts; nothing generates one per process.
"""

import argparse
import os

from tunix.experimental.trajectory import config as config_lib


def add_arguments(parser: argparse.ArgumentParser) -> None:
  """Adds the Trajectory Store flags to an entry point's parser."""
  parser.add_argument(
      "--trajectory_store_enabled",
      action="store_true",
      help=(
          "Record rollout trajectories to a Trajectory Store. Off by default;"
          " when set, --trajectory_store_run_id must be the same for every"
          " process in the run."
      ),
  )
  parser.add_argument(
      "--trajectory_store_backend",
      type=str,
      default=os.getenv("TRAJECTORY_STORE_BACKEND", "file"),
      choices=("file", "memory"),
      help=(
          "file writes under root_dir/run_id and is shared by every process;"
          " memory is process-local and invisible to the other processes."
      ),
  )
  parser.add_argument(
      "--trajectory_store_root_dir",
      type=str,
      default=os.getenv("TRAJECTORY_STORE_ROOT_DIR", ""),
      help="Base directory for the file backend (local path or gs:// URI).",
  )
  parser.add_argument(
      "--trajectory_store_run_id",
      type=str,
      default=os.getenv("TRAJECTORY_STORE_RUN_ID", ""),
      help=(
          "Identifier scoping this run under root_dir. Generated once by"
          " launcher.sh and passed to every process; never defaulted"
          " per-process, since each process would then get a different one."
      ),
  )


def config_from_args(
    args: argparse.Namespace,
) -> config_lib.TrajectoryStoreConfig | None:
  """Builds the store config from parsed args, or None when disabled."""
  if not getattr(args, "trajectory_store_enabled", False):
    return None
  return config_lib.TrajectoryStoreConfig(
      enabled=True,
      backend=args.trajectory_store_backend,
      # Empty strings come from the flags' defaults when nothing was passed;
      # None makes the config raise with a message naming the missing field.
      root_dir=args.trajectory_store_root_dir or None,
      run_id=args.trajectory_store_run_id or None,
  )
