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

"""Every VanillaRollout.generate() call samples with its own seed."""

from unittest import mock

from absl.testing import absltest
from tunix.rl.rollout import base_rollout
from tunix.rl.rollout import vanilla_rollout


class VanillaRolloutSeedTest(absltest.TestCase):

  def _rollout(self):
    with mock.patch.object(vanilla_rollout.sampler, "Sampler"):
      return vanilla_rollout.VanillaRollout(
          model=mock.MagicMock(),
          tokenizer=mock.MagicMock(),
          cache_config_or_size=base_rollout.CacheConfig(
              cache_size=8, num_layers=1, num_kv_heads=1, head_dim=1
          ),
      )

  def test_unset_seed_still_differs_per_call(self):
    """seed=None is what `Sampler` maps to a fixed PRNGKey(0)."""
    rollout = self._rollout()
    seeds = [rollout._next_seed(None) for _ in range(4)]
    self.assertLen(set(seeds), 4)

  def test_configured_seed_is_the_base_and_the_run_is_reproducible(self):
    rollout = self._rollout()
    first = [rollout._next_seed(42) for _ in range(4)]
    self.assertLen(set(first), 4)
    self.assertTrue(all(s > 42 for s in first))
    # A second rollout built the same way replays the same sequence.
    self.assertEqual([self._rollout()._next_seed(42) for _ in range(1)], [43])

  def test_each_rollout_counts_its_own_calls(self):
    a, b = self._rollout(), self._rollout()
    a._next_seed(0)
    a._next_seed(0)
    self.assertEqual(b._next_seed(0), a._next_seed(0) - 2)


if __name__ == "__main__":
  absltest.main()
