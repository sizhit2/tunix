# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Main entry point for distributed process execution.

Note that the "process" abstraction represents a logical distributed process
and does not map directly to an OS process; when multiple processes are
specified within a single invocation, they execute concurrently as OS threads
sharing the same `--process_executor`.

TODO: choose a dedicated name for "process" to avoid confusion. e.g. "LogicalProcess".

Usage Examples:
  1. Single "process" execution:
    ```shell
    python -m tunix.experimental.distributed.runtime.main \
        --process_main=basics.flag.main \
        --message="hello flag"
    ```

  2. Multi "process" execution separated by `--process`:
    ```shell
    python -m tunix.experimental.distributed.runtime.main \
        --process_executor=tunix.experimental.distributed.runtime.executor.LocalExecutor \
        --process \
            --process_main=basics.door.main \
            --discovery_id=door \
            --discovery_port=12345 \
        --process \
            --process_main=basics.knocker.main \
            --discovery_addrs=door:12345 \
            --say="open the door"
    ```
"""

import argparse
import concurrent.futures
import dataclasses
import importlib
import logging
import os
import sys
from typing import Any, Callable

from absl import logging as absl_logging


@dataclasses.dataclass(frozen=True)
class PreparedProcess:
  """Parsed configuration and entrypoint for a distributed process.

  Attributes:
    main_fn: Resolved callable entrypoint for the process.
    argv: Unrecognized command-line arguments forwarded to `main_fn`.
    context_args: Parsed runtime discovery and execution flags.
  """

  main_fn: Callable[..., Any]
  argv: list[str]
  context_args: argparse.Namespace


def import_symbol(fqn: str) -> Any:
  """Imports a symbol (class or function) from its fully qualified name.

  Args:
    fqn: Fully qualified dot-separated path to the symbol (e.g.
      'package.module.ClassName').

    Returns:
      The resolved symbol object.

    Raises:
      ValueError: If `fqn` does not contain a module path and symbol name.
      ModuleNotFoundError: If the module cannot be imported.
      AttributeError: If the symbol does not exist in the module.
  """
  if "." not in fqn:
    raise ValueError(f"invalid symbol path: {fqn}")
  module_path, *symbol_names = fqn.rsplit(".", maxsplit=1)
  symbol = importlib.import_module(module_path)
  for symbol_name in symbol_names:
    symbol = getattr(symbol, symbol_name)
  return symbol


def split_process_argv(argv: list[str]) -> list[list[str]]:
  """Splits command-line arguments into per-process slices separated by `--process`.

  Args:
    argv: List of command-line arguments after stripping the script path.

  Returns:
    A list of argument lists, where each entry corresponds to one process.
  """
  if "--process" not in argv:
    return [argv]

  slices: list[list[str]] = []
  current_slice: list[str] = []
  for token in argv:
    if token == "--process":
      if current_slice:
        slices.append(current_slice)
        current_slice = []
    else:
      current_slice.append(token)
  if current_slice:
    slices.append(current_slice)

  return slices


def prepare_process(argv: list[str]) -> PreparedProcess:
  """Parses discovery flags and imports the target entrypoint for a single process.

  Args:
    argv: Command-line arguments slice belonging to this process.

  Returns:
    A `PreparedProcess` containing the imported main function, application
    arguments, and parsed discovery context arguments.

  Raises:
    ValueError: If `--process_main` cannot be imported or resolved.
  """
  parser = argparse.ArgumentParser(
      description="process main", allow_abbrev=False, add_help=False
  )

  parser.add_argument(
      "--process_main",
      type=str,
      default="",
      help="Fully qualified name of the target main function to execute.",
  )
  parser.add_argument(
      "--discovery_id",
      type=str,
      default="",
      help="Id to identify the process. Id and port form a discovery address.",
  )
  parser.add_argument(
      "--discovery_port",
      type=int,
      default=0,
      help=(
          "Port of this process, that other processes register themselves to."
          " If non-zero, will run a service."
      ),
  )
  parser.add_argument(
      "--discovery_addrs",
      type=str,
      default="",
      help="Addresses of other processes, that this process registers to.",
  )

  context_args, process_argv = parser.parse_known_args(argv)

  try:
    process_main = import_symbol(context_args.process_main)
  except AttributeError as e:
    raise ValueError(
        f"Invalid --process_main={context_args.process_main}: {e}"
    ) from e

  return PreparedProcess(
      main_fn=process_main,
      argv=process_argv,
      context_args=context_args,
  )


# --- TEMPORARY BUILD-PROVENANCE INSTRUMENTATION (not part of PR #1983) ---
# Answers, from inside the running process: which image am I in, and which
# copy of the tunix source did this interpreter actually import?
def _log_build_provenance() -> None:
  """Logs which image build and which source tree this process is executing."""
  import hashlib  # pylint: disable=g-import-not-at-top
  import socket  # pylint: disable=g-import-not-at-top

  def _read(path: str) -> str:
    try:
      with open(path) as f:
        return f.read().strip()
    except OSError:
      return ""

  try:
    import tunix  # pylint: disable=g-import-not-at-top

    tunix_init = getattr(tunix, "__file__", "?") or "?"
    tunix_root = os.path.dirname(os.path.dirname(os.path.abspath(tunix_init)))
  except Exception as e:  # pylint: disable=broad-except
    tunix_init, tunix_root = f"IMPORT-FAILED: {e!r}", "?"

  # Fingerprint the orchestrator source this process actually resolves, via the
  # same import machinery the run uses. Compare with `sha256sum` on the host.
  import importlib.util  # pylint: disable=g-import-not-at-top

  try:
    spec = importlib.util.find_spec("tunix.experimental.orchestrator.rl_program")
    rl_path = spec.origin if spec else None
  except Exception:  # pylint: disable=broad-except
    rl_path = None
  if rl_path:
    try:
      with open(rl_path, "rb") as f:
        rl_sha = hashlib.sha256(f.read()).hexdigest()[:16]
    except OSError:
      rl_sha = "(unreadable)"
  else:
    rl_path, rl_sha = "(unresolved)", "(unresolved)"

  # Baked stamp is authoritative (written at image build time); the env
  # fallback is supplied by the launcher and only as trustworthy as it is.
  stamp = _read("/etc/tunix-build-stamp")
  if stamp:
    stamp = f"{stamp} [baked into image]"
  elif os.environ.get("TUNIX_BUILD_STAMP"):
    stamp = f'{os.environ["TUNIX_BUILD_STAMP"]} [from env, not baked]'
  else:
    stamp = "(unstamped image, no TUNIX_BUILD_STAMP env)"
  if tunix_root == "/app":
    verdict = "IMAGE-BAKED (/app from the image build)"
  elif os.path.isdir("/app/tunix"):
    verdict = f"BIND-MOUNT {tunix_root} SHADOWS the image's /app copy"
  else:
    verdict = f"{tunix_root} (no /app in image)"

  logging.info("=== TUNIX BUILD PROVENANCE ===")
  logging.info("provenance: host=%s pid=%d", socket.gethostname(), os.getpid())
  logging.info("provenance: image stamp   = %s", stamp)
  logging.info("provenance: python        = %s", sys.executable)
  logging.info("provenance: PYTHONPATH    = %s", os.environ.get("PYTHONPATH", "(unset)"))
  logging.info("provenance: tunix.__file__= %s", tunix_init)
  logging.info("provenance: rl_program.py = %s", rl_path)
  logging.info("provenance: rl_program.py sha256[:16] = %s", rl_sha)
  logging.info("provenance: EXECUTING     = %s", verdict)
  logging.info("=== END BUILD PROVENANCE ===")
# --- END TEMPORARY INSTRUMENTATION ---


def main(argv: list[str]) -> None:
  """Main entry point for distributed process runtime execution.

  Parses global executor settings, splits per-process argument slices delimited
  by `--process`, and executes the configured processes using the executor.

  Args:
    argv: System command-line arguments (`sys.argv`).
  """
  parser = argparse.ArgumentParser(
      description="distributed main", allow_abbrev=False, add_help=False
  )

  # The same executor is used for all processes started in one invocation.
  parser.add_argument(
      "--process_executor",
      type=str,
      default="tunix.experimental.distributed.runtime.executor.LocalExecutor",
      help="Fully qualified class name of the process executor implementation.",
  )

  main_args, processes_argv = parser.parse_known_args(argv)
  # Strip the first argument (program path), which should be hidden from processes.
  processes_argv = processes_argv[1:]

  process_executor = import_symbol(main_args.process_executor)()

  # force=True here previously pinned every worker process at INFO, which
  # silently discarded the logging.debug() calls already present in the rollout
  # path (e.g. TrajectoryCollectEngine's "model_call starting/done"). Make the
  # level selectable so those can be turned on without editing code.
  log_level_name = os.environ.get("TUNIX_LOG_LEVEL", "INFO").upper()
  log_level = getattr(logging, log_level_name, None)
  if not isinstance(log_level, int):
    log_level = logging.INFO
  logging.basicConfig(
      level=log_level,
      format="%(asctime)s [%(filename)s:%(lineno)d] %(levelname)s %(message)s",
      force=True,
  )
  # absl gates its own debug() calls on verbosity independently of the root
  # logger's level, so both must be raised for absl logging.debug to emit.
  absl_logging.set_verbosity(
      absl_logging.DEBUG if log_level <= logging.DEBUG else absl_logging.INFO
  )

  _log_build_provenance()

  prepared_processes = [
      prepare_process(slice_argv)
      for slice_argv in split_process_argv(processes_argv)
  ]

  if len(prepared_processes) == 1:
    prepared = prepared_processes[0]
    process_executor.run(prepared.main_fn, prepared.argv, prepared.context_args)
  else:
    with concurrent.futures.ThreadPoolExecutor(
        max_workers=len(prepared_processes)
    ) as executor:
      futures = [
          executor.submit(
              process_executor.run,
              prepared.main_fn,
              prepared.argv,
              prepared.context_args,
          )
          for prepared in prepared_processes
      ]
      try:
        for future in concurrent.futures.as_completed(futures):
          future.result()
      except SystemExit as e:
        code = (
            e.code if isinstance(e.code, int) else (0 if e.code is None else 1)
        )
        if code != 0:
          logging.exception(
              "Distributed process exited with non-zero code. Forcefully"
              " terminating application."
          )
        os._exit(code)
      except BaseException:
        logging.exception(
            "Distributed process execution failed or interrupted. Forcefully"
            " terminating application."
        )
        os._exit(1)


if __name__ == "__main__":
  main(sys.argv)
