# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Tests for VanillaRollout's RNG seeding."""

from absl.testing import absltest
from flax import nnx
from tunix.rl.rollout import base_rollout
from tunix.rl.rollout import vanilla_rollout
from tunix.tests import test_common as tc


class VanillaRolloutRngTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    self.vocab = tc.MockVocab()
    self.cache_config = base_rollout.CacheConfig(
        cache_size=64, num_layers=4, num_kv_heads=4, head_dim=16
    )

  def _rollout(self, sampling_rng_seed=0):
    return vanilla_rollout.VanillaRollout(
        tc.ToyTransformer(
            config=tc.ModelConfig(vocab_size=self.vocab.GetPieceSize()),
            rngs=nnx.Rngs(42),
        ),
        self.vocab,
        self.cache_config,
        sampling_rng_seed=sampling_rng_seed,
    )

  def _gen(self, rollout):
    return rollout.generate(
        ["input string"],
        base_rollout.RolloutConfig(
            max_tokens_to_generate=8,
            max_prompt_length=8,
            temperature=1.0,
            top_p=1.0,
        ),
    ).text[0]

  def test_rng_seed_pins_the_stream(self):
    self.assertEqual(
        self._gen(self._rollout(sampling_rng_seed=3)),
        self._gen(self._rollout(sampling_rng_seed=3)),
    )

  def test_different_rng_seeds_diverge(self):
    self.assertNotEqual(
        self._gen(self._rollout(sampling_rng_seed=3)),
        self._gen(self._rollout(sampling_rng_seed=4)),
    )

  def test_unseeded_rollouts_advance(self):
    rollout = self._rollout()
    self.assertLen({self._gen(rollout) for _ in range(4)}, 4)


if __name__ == "__main__":
  absltest.main()
