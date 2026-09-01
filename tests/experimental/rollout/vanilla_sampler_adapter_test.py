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

"""Tests for VanillaSamplerAdapter with Tunix JAX Sampler."""

import asyncio
import types
from unittest import mock
from absl.testing import absltest
from flax import nnx
import numpy as np
from tunix.experimental.rollout import sampler as base_sampler_lib
from tunix.experimental.rollout import vanilla_sampler_adapter
from tunix.experimental.weight_sync import weight_sync
from tunix.generate import sampler as generate_sampler_lib
from tunix.tests import test_common as tc


class VanillaSamplerAdapterTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    self.vocab = tc.MockVocab()
    self.transformer = tc.ToyTransformer(
        config=tc.ModelConfig(vocab_size=self.vocab.GetPieceSize()),
        rngs=nnx.Rngs(42),
    )
    self.cache_config = generate_sampler_lib.CacheConfig(
        cache_size=64,
        num_layers=4,
        num_kv_heads=4,
        head_dim=16,
    )
    self.vanilla_sampler = vanilla_sampler_adapter.VanillaSamplerAdapter(
        server_id="tpu_slice_01",
        transformer=self.transformer,
        tokenizer=self.vocab,
        cache_config=self.cache_config,
    )
    self.vanilla_sampler.initialize()

  def test_single_sampling_request(self):
    req = base_sampler_lib.SamplingRequest(
        request_id="req_01",
        prompt="input string",
        sampling_params=base_sampler_lib.SamplingParams(
            max_tokens=10,
            temperature=0.0,
        ),
    )
    response = asyncio.run(self.vanilla_sampler.sample(req))
    self.assertIsInstance(response, base_sampler_lib.SamplingResponse)
    self.assertEqual(response.request_id, "req_01")
    self.assertIsNotNone(response.text)
    self.assertGreater(response.prompt_token_ids.size, 0)

  def test_sampling_request_with_logprobs(self):
    req = base_sampler_lib.SamplingRequest(
        request_id="req_logprobs",
        prompt="input string",
        sampling_params=base_sampler_lib.SamplingParams(
            max_tokens=10,
            temperature=0.0,
            return_logprobs=True,
        ),
    )
    response = asyncio.run(self.vanilla_sampler.sample(req))
    self.assertIsInstance(response, base_sampler_lib.SamplingResponse)
    self.assertEqual(response.request_id, "req_logprobs")
    self.assertIsNotNone(response.logprobs)

  def test_batch_sampling_requests(self):
    reqs = [
        base_sampler_lib.SamplingRequest(
            request_id="req_a",
            prompt="input string 1",
            sampling_params=base_sampler_lib.SamplingParams(
                max_tokens=8,
                temperature=0.0,
            ),
        ),
        base_sampler_lib.SamplingRequest(
            request_id="req_b",
            prompt="hello world 2",
            sampling_params=base_sampler_lib.SamplingParams(
                max_tokens=8,
                temperature=0.0,
            ),
        ),
    ]
    responses = asyncio.run(self.vanilla_sampler.sample(reqs))
    self.assertIsInstance(responses, list)
    self.assertLen(responses, 2)
    self.assertEqual(responses[0].request_id, "req_a")
    self.assertEqual(responses[1].request_id, "req_b")
    self.assertIsNotNone(responses[0].text)
    self.assertIsNotNone(responses[1].text)
    self.assertGreater(responses[0].prompt_token_ids.size, 0)
    self.assertGreater(responses[1].prompt_token_ids.size, 0)

  def _req(self, request_id, prompt="input string", seed=None):
    return base_sampler_lib.SamplingRequest(
        request_id=request_id,
        prompt=prompt,
        sampling_params=base_sampler_lib.SamplingParams(
            max_tokens=8, temperature=1.0, top_p=1.0, seed=seed
        ),
    )

  def test_unseeded_group_members_decode_differently(self):
    """A GRPO group is issued as separate calls; they must not all match."""
    texts = [
        asyncio.run(self.vanilla_sampler.sample([self._req(f"r{i}")]))[0].text
        for i in range(4)
    ]
    self.assertLen(set(texts), 4)

  def test_seeded_request_is_reproducible_across_calls(self):
    first = asyncio.run(self.vanilla_sampler.sample([self._req("a", seed=11)]))
    second = asyncio.run(self.vanilla_sampler.sample([self._req("b", seed=11)]))
    self.assertEqual(first[0].text, second[0].text)

  def test_rng_seed_reaches_the_underlying_sampler(self):
    """sampling_rng_seed must reach the Sampler, not be silently dropped."""

    def build(sampling_rng_seed):
      adapter = vanilla_sampler_adapter.VanillaSamplerAdapter(
          server_id="tpu_slice_seeded",
          transformer=self.transformer,
          tokenizer=self.vocab,
          cache_config=self.cache_config,
          sampling_rng_seed=sampling_rng_seed,
      )
      adapter.initialize()
      return adapter

    first = asyncio.run(build(5).sample([self._req("a")]))[0].text
    second = asyncio.run(build(5).sample([self._req("b")]))[0].text
    self.assertEqual(first, second)
    other = asyncio.run(build(6).sample([self._req("c")]))[0].text
    self.assertNotEqual(first, other)

  def test_construct_with_integer_cache_size(self):
    sampler_adapter_direct = vanilla_sampler_adapter.VanillaSamplerAdapter(
        server_id="tpu_slice_02",
        transformer=self.transformer,
        tokenizer=self.vocab,
        cache_config=64,
    )
    req = base_sampler_lib.SamplingRequest(
        request_id="req_direct",
        prompt="direct prompt",
        sampling_params=base_sampler_lib.SamplingParams(
            max_tokens=6,
            temperature=0.0,
        ),
    )
    response = asyncio.run(sampler_adapter_direct.sample(req))
    self.assertIsInstance(response, base_sampler_lib.SamplingResponse)
    self.assertEqual(response.request_id, "req_direct")
    self.assertIsNotNone(response.text)
    self.assertEqual(response.prompt_token_ids.dtype, np.int32)

  def test_uninitialized_sampler_raises(self):
    uninit_sampler = vanilla_sampler_adapter.VanillaSamplerAdapter(
        server_id="empty"
    )
    with self.assertRaises(RuntimeError):
      asyncio.run(
          uninit_sampler.sample(
              base_sampler_lib.SamplingRequest(prompt="hello")
          )
      )

  def test_weight_sync_without_raiden_delegate(self):
    self.assertIsNone(asyncio.run(self.vanilla_sampler.bind_weight_sync()))
    self.assertTrue(asyncio.run(self.vanilla_sampler.pre_weight_sync()))
    new_state = self.vanilla_sampler.sampler.transformer_state
    req = base_sampler_lib.WeightSyncRequest(weights=new_state)
    self.assertTrue(
        asyncio.run(self.vanilla_sampler.weight_sync(sync_request=req))
    )
    self.assertTrue(asyncio.run(self.vanilla_sampler.post_weight_sync()))
    with self.assertRaises(NotImplementedError):
      asyncio.run(self.vanilla_sampler.get_weight_sync_metadata())

  def test_weight_sync_with_raiden_delegate(self):
    mock_delegate = mock.MagicMock()
    mock_delegate.is_bounded.return_value = False
    mock_delegate.bind_weight_sync = mock.AsyncMock(return_value=True)
    mock_delegate.get_weight_sync_metadata = mock.AsyncMock(
        return_value=[{"unit": "rollout"}]
    )
    mock_delegate.pre_weight_sync = mock.AsyncMock(return_value=True)
    mock_delegate.weight_sync = mock.AsyncMock(return_value=10)
    mock_delegate.post_weight_sync = mock.AsyncMock(return_value=True)

    sampler_with_raiden = vanilla_sampler_adapter.VanillaSamplerAdapter(
        server_id="tpu_slice_raiden",
        transformer=self.transformer,
        tokenizer=self.vocab,
        cache_config=self.cache_config,
        config=types.SimpleNamespace(
            weight_sync_mode=weight_sync.WeightSyncMode.RAIDEN
        ),
        raiden_sync_delegate=mock_delegate,
    )
    sampler_with_raiden.initialize()

    sync_req = base_sampler_lib.WeightSyncRequest(policy_version=10)

    # 1. bind_weight_sync
    asyncio.run(sampler_with_raiden.bind_weight_sync(sync_req))
    mock_delegate.bind_weight_sync.assert_awaited_once_with(
        sync_request=sync_req,
        state=sampler_with_raiden.sampler.transformer_state,
    )
    mock_delegate.is_bounded.return_value = True

    # 2. get_weight_sync_metadata
    metadata = asyncio.run(sampler_with_raiden.get_weight_sync_metadata())
    self.assertEqual(metadata, [{"unit": "rollout"}])

    # 3. pre_weight_sync
    self.assertTrue(asyncio.run(sampler_with_raiden.pre_weight_sync(sync_req)))
    mock_delegate.pre_weight_sync.assert_awaited_once_with(
        sync_request=sync_req
    )

    # 4. weight_sync
    version = asyncio.run(sampler_with_raiden.weight_sync(sync_req))
    self.assertEqual(version, 10)
    mock_delegate.weight_sync.assert_awaited_once_with(sync_request=sync_req)

    # 5. post_weight_sync
    self.assertTrue(asyncio.run(sampler_with_raiden.post_weight_sync(sync_req)))
    mock_delegate.post_weight_sync.assert_awaited_once_with(
        sync_request=sync_req
    )


if __name__ == "__main__":
  absltest.main()
