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

"""TrajectoryQueueManager specialization of GroupQueueManager for TrajectoryItem."""

from collections.abc import Callable, Hashable, Sequence
from typing import Any, Optional
from absl import logging
from tunix.experimental.common import datatypes
from tunix.rl.agentic.queue_manager import group_queue_manager

TrajectoryItem = datatypes.TrajectoryItem
GroupFn = group_queue_manager.GroupFn[TrajectoryItem]
FilterFn = group_queue_manager.FilterFn[TrajectoryItem]


class TrajectoryQueueManager(group_queue_manager.GroupQueueManager):
  """Specialized GroupQueueManager holding TrajectoryItems with ACK and abort support."""

  def __init__(
      self,
      *,
      group_size: Optional[int] = None,
      group_fn: Optional[GroupFn] = None,
      filter_fn: Optional[FilterFn] = None,
      key_fn: Optional[Callable[[datatypes.TrajectoryItem], Hashable]] = None,
  ):
    """Initializes TrajectoryQueueManager.

    Args:
      group_size: Optional target number of trajectories per ready group when
        using default grouping.
      group_fn: Optional custom grouping function. If None, `group_size` must be
        provided.
      filter_fn: Optional pluggable function to filter candidate groups.
      key_fn: Optional function to extract grouping key. Defaults to prompt_id
        fallback.
    """
    if key_fn is None and group_fn is None:

      def _default_key_fn(item: datatypes.TrajectoryItem) -> Hashable:
        prompt_id = getattr(item, "prompt_id", None)
        if prompt_id is not None and prompt_id != "":
          return prompt_id
        return id(item)

      key_fn = _default_key_fn

    super().__init__(
        group_size=group_size,
        group_fn=group_fn,
        filter_fn=filter_fn,
        key_fn=key_fn,
    )

  @classmethod
  def create(
      cls,
      group_size: int = 1,
      max_staleness: int = 0,
      current_policy_version: Callable[[], int] | None = None,
      filter_fn: Any | None = None,
  ) -> "TrajectoryQueueManager":
    """Creates a grouped trajectory queue with optional policy staleness filtering.

    Args:
      group_size: Target number of trajectories per ready group.
      max_staleness: Maximum tolerated lag, in policy versions, between the
        weights a trajectory was generated with and the current policy. `0`
        means strictly on-policy: only trajectories generated with the current
        policy version are admitted.
      current_policy_version: Callable returning the trainer's current policy
        version. Staleness filtering is enabled whenever this is provided;
        without it there is no reference point and no filtering is possible.
      filter_fn: Optional additional filter applied to groups that pass the
        staleness check.

    Returns:
      A `TrajectoryQueueManager` enforcing the requested staleness bound.
    """
    assert max_staleness >= 0, "max_staleness must be non-negative."
    combined_filter = filter_fn
    # NB: `max_staleness == 0` is the *strictest* setting, not "filtering
    # disabled" -- `--max_staleness` is documented as "0 means queue-level
    # on-policy training". Gating on `max_staleness > 0` here would silently
    # admit arbitrarily stale trajectories in exactly that configuration.
    if current_policy_version is not None:

      def _staleness_filter(group: Sequence[Any]) -> Any:
        min_allowed = current_policy_version() - max_staleness
        valid = [
            item
            for item in group
            if getattr(item, "policy_version", 0) >= min_allowed
        ]
        filtered = [
            item
            for item in group
            if getattr(item, "policy_version", 0) < min_allowed
        ]
        if filtered:
          # Group-relative algorithms (e.g. GRPO) derive each advantage from a
          # baseline over the whole group, and reshape rewards to
          # `(-1, group_size)` to do so. Admitting the surviving members of a
          # partially stale group would both skew that baseline and break the
          # reshape, so staleness is all-or-nothing: drop the entire group.
          logging.warning(
              "Dropping trajectory group (prompt_id=%s): %d of %d items are"
              " staler than the %d-version bound (min_allowed=%d,"
              " versions=%s).",
              getattr(group[0], "prompt_id", "<unknown>") if group else None,
              len(filtered),
              len(group),
              max_staleness,
              min_allowed,
              [getattr(item, "policy_version", 0) for item in group],
          )
          return [], list(group)
        if filter_fn is not None:
          return filter_fn(valid)
        return valid, filtered

      combined_filter = _staleness_filter

    return cls(
        group_size=group_size,
        filter_fn=combined_filter,  # pyrefly: ignore[bad-argument-type]
    )

  def __aiter__(self) -> "TrajectoryQueueManager":
    return self

  async def __anext__(self) -> list[datatypes.TrajectoryItem]:
    group = await self.get_group()
    if not group:
      raise StopAsyncIteration
    return group

  async def get_group(self) -> list[datatypes.TrajectoryItem]:
    """Retrieves a single ready group of TrajectoryItems."""
    return await self._get_one_ready_group()

  async def get_batch(
      self,
      batch_size: int | None = None,
      num_groups: int | None = None,
  ) -> list[datatypes.TrajectoryItem]:
    """Retrieves items by either batch_size or num_groups."""
    # TODO: why do we need both batch_size and num_groups?
    # TODO: should this be in parent class?
    if num_groups is not None:
      out: list[datatypes.TrajectoryItem] = []
      for _ in range(num_groups):
        g = await self._get_one_ready_group()
        if not g:
          break
        out.extend(g)
      return out
    actual_batch_size = (
        batch_size if batch_size is not None else self.group_size
    )
    return await super().get_batch(batch_size=actual_batch_size)  # pyrefly: ignore[bad-argument-type]

  def commit(self, step: int, groups: Sequence[Any] | None = None) -> None:
    """Commits in-flight groups after a successful global step boundary."""
    # TODO: implement the commit and keep track of uncommited items.
    # might be worth putting in parent class.
    pass

  @property
  def ready_groups_count(self) -> int:
    """Returns the count of ready complete groups in the queue."""
    return len(self._ready_groups)

  @property
  def incomplete_buckets_count(self) -> int:
    """Returns the count of incomplete buckets currently buffering items."""
    return len(self._buckets)

  async def abort(self, exc: BaseException) -> None:
    """Aborts queue and unblocks all waiting consumers with the given exception."""
    if isinstance(exc, Exception):
      await self.put_exception(exc)
    await self.prepare_clear()
