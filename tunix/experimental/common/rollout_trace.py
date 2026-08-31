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

"""Opt-in tracing of the rollout path.

Logs one line at each hop a prompt makes on its way through the orchestrator:

  DATASET   the item as it comes off the dataset, before any normalization
  REQUEST   the `RolloutRequest` actually put on the wire to a rollout worker
  RESPONSE  the `RolloutResponse` that came back, before conversion
  DTO       the `TrajectoryItem` handed to the queue manager

Together these separate "the dataset is wrong", "we sent the wrong thing", and
"the worker returned the wrong thing", which are otherwise indistinguishable
from the DTO alone.

Tracing is off unless `TUNIX_TRACE_ROLLOUTS` is set to a truthy value, since it
emits O(num_prompts * group_size) lines per step and includes prompt and
completion text.
"""

import os
from typing import Any

from absl import logging
import numpy as np


_ENV_VAR = "TUNIX_TRACE_ROLLOUTS"
_TRUTHY = frozenset(("1", "true", "yes", "on"))

# Prompt and completion text are truncated to keep a single trace line
# greppable; the untruncated text still reaches the trainer.
_TEXT_CHARS = 600

# Token arrays are summarized by length plus a short head, so a trace line does
# not carry a few thousand integers.
_TOKEN_HEAD = 12


class _Missing:
  """Distinguishes an absent key from one holding an empty value."""

  def __repr__(self) -> str:
    return "<absent>"


_MISSING = _Missing()


def enabled() -> bool:
  """Returns whether rollout tracing is turned on for this process."""
  return os.environ.get(_ENV_VAR, "").strip().lower() in _TRUTHY


def _preview(value: Any) -> str:
  """Renders a value as a repr truncated to `_TEXT_CHARS`.

  An empty completion is a real and interesting result, so it is rendered as
  `''` rather than skipped; a key that was never set renders as `<absent>`, so
  the two cannot be confused.
  """
  if isinstance(value, _Missing):
    return repr(value)
  text = value if isinstance(value, str) else repr(value)
  if len(text) <= _TEXT_CHARS:
    return repr(text) if isinstance(value, str) else text
  return f"{text[:_TEXT_CHARS]!r}...[+{len(text) - _TEXT_CHARS} chars]"


def _tokens(arr: Any) -> str:
  """Renders a token array as `len=N head=[...]`."""
  if arr is None:
    return "none"
  try:
    flat = np.asarray(arr).reshape(-1)
  except (TypeError, ValueError):
    return f"unrenderable({type(arr).__name__})"
  head = flat[:_TOKEN_HEAD].tolist()
  suffix = "..." if flat.size > _TOKEN_HEAD else ""
  return f"len={flat.size} head={head}{suffix}"


def trace_dataset_item(index: int, item: Any) -> None:
  """Logs a dataset item at the point the dispatch stage picks it up."""
  if not enabled():
    return
  if isinstance(item, dict):
    keys = sorted(item)
    prompt = item.get("prompt", item)
    prompt_id = item.get("prompt_id")
  else:
    keys = sorted(k for k in vars(item)) if hasattr(item, "__dict__") else []
    prompt = getattr(item, "prompt", item)
    prompt_id = getattr(item, "prompt_id", None)
  logging.info(
      "[rollout-trace] DATASET idx=%d prompt_id=%s type=%s keys=%s prompt=%s",
      index,
      prompt_id,
      type(item).__name__,
      keys,
      _preview(prompt),
  )


def trace_request(req: Any) -> None:
  """Logs a RolloutRequest immediately before it is dispatched to a worker."""
  if not enabled():
    return
  logging.info(
      "[rollout-trace] REQUEST request_id=%s prompt_id=%s group_index=%s"
      " target_pv=%s max_turns=%s gen_kwargs=%s metadata=%s prompt=%s",
      getattr(req, "request_id", None),
      getattr(req, "prompt_id", None),
      getattr(req, "group_index", None),
      getattr(req, "target_policy_version", None),
      getattr(req, "max_turns", None),
      getattr(req, "generation_kwargs", None),
      _redact_text(getattr(req, "metadata", None)),
      _preview(getattr(req, "prompt", None)),
  )


def trace_response(resp: Any) -> None:
  """Logs a RolloutResponse as received, before conversion to a DTO."""
  if not enabled():
    return
  segments = getattr(resp, "segments", None) or []
  seg_desc = [
      f"{getattr(s, 'source', '?')}:{_tokens(getattr(s, 'tokens', None))}"
      for s in segments
  ]
  metadata = getattr(resp, "metadata", None) or {}
  logging.info(
      "[rollout-trace] RESPONSE request_id=%s prompt_id=%s status=%s pv=%s"
      " env_reward=%s error=%s prompt_tokens=(%s) segments=%s metadata=%s"
      " text=%s",
      getattr(resp, "request_id", None),
      getattr(resp, "prompt_id", None),
      getattr(resp, "status", None),
      getattr(resp, "policy_version", None),
      getattr(resp, "env_reward", None),
      getattr(resp, "error", None),
      _tokens(getattr(resp, "prompt_tokens", None)),
      seg_desc,
      _redact_text(metadata),
      _preview(metadata.get("text", _MISSING)),
  )


def trace_trajectory_item(item: Any) -> None:
  """Logs the TrajectoryItem handed to the raw trajectory queue."""
  if not enabled():
    return
  traj = getattr(item, "traj", None)
  metadata = getattr(item, "metadata", None) or {}
  logging.info(
      "[rollout-trace] DTO prompt_id=%s group_index=%s status=%s pv=%s"
      " reward=%s prompt_tokens=(%s) completion_tokens=(%s) steps=%d text=%s",
      getattr(item, "prompt_id", None),
      getattr(item, "group_index", None),
      getattr(traj, "status", None),
      getattr(item, "policy_version", None),
      getattr(traj, "reward", None),
      _tokens(getattr(item, "prompt_tokens", None)),
      _tokens(getattr(item, "completion_tokens", None)),
      len(getattr(traj, "steps", None) or []),
      _preview(metadata.get("text", _MISSING)),
  )


def _redact_text(metadata: Any) -> Any:
  """Drops the bulky `text` key, which every caller logs separately."""
  if not isinstance(metadata, dict):
    return metadata
  return {k: v for k, v in metadata.items() if k != "text"}
