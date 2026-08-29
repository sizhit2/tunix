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

"""Vanilla Sampler adapter using Tunix JAX Sampler."""

import abc
import contextlib
import os

from flax import nnx
import jax
import numbers
from typing import Any, List, Sequence
from absl import logging
from flax import nnx
import jax
import numpy as np
from tunix.experimental.rollout import sampler as base_sampler_lib
from tunix.experimental.weight_sync import weight_sync
from tunix.generate import sampler as generate_sampler_lib

Sampler = base_sampler_lib.Sampler


def _infer_mesh(transformer: Any) -> Any:
  """Best-effort recovery of the mesh a model's parameters are sharded over.

  Used when the caller does not pass `mesh=` explicitly. Returns None if no
  sharded array with a mesh is found, in which case sampling runs without a
  mesh context exactly as before.
  """
  if transformer is None:
    return None
  try:
    import jax  # pylint: disable=g-import-not-at-top
    from flax import nnx  # pylint: disable=g-import-not-at-top

    try:
      leaves = jax.tree.leaves(nnx.state(transformer))
    except Exception:  # pylint: disable=broad-except
      leaves = jax.tree.leaves(transformer)
    for leaf in leaves:
      value = getattr(leaf, "value", leaf)
      sharding = getattr(value, "sharding", None)
      mesh = getattr(sharding, "mesh", None)
      if mesh is not None and getattr(mesh, "devices", None) is not None:
        return mesh
  except Exception:  # pylint: disable=broad-except
    return None
  return None


class VanillaSamplerAdapter(Sampler, abc.ABC):
  """Standalone TPU/GPU slice running Tunix Vanilla JAX Sampler.

  Constructs or wraps a Tunix generate_sampler_lib.Sampler instance and
  executes sampling requests.

  Supported Weight Synchronization Modes:
    1. Raiden Mode (`weight_sync_mode == RAIDEN`):
       Delegates high-performance decentralized P2P / DCN weight synchronization
       to `RaidenWeightSyncDelegate`. Binds destination memory buffers using
       `self.sampler.transformer_state` and executes phased synchronization
       lifecycle hooks (`bind`, `pre`, `sync`, `post`).
    2. Fallback Direct Mode (`weight_sync_mode == FALLBACK`):
       Synchronizes weights in-place without Raiden transport. When a weight
       payload is received in `sync_request.weights`, updates the underlying
       JAX sampler directly via `self.sampler.transformer_state`.
  """

  def __init__(
      self,
      server_id: str,
      transformer: Any = None,
      tokenizer: Any = None,
      cache_config: generate_sampler_lib.CacheConfig | int | None = None,
      image_processor: Any = None,
      model: Any = None,
      config: Any = None,
      raiden_sync_delegate: Any = None,
      mesh: Any = None,
      **kwargs,
  ):
    self.server_id = server_id
    self.transformer = transformer if transformer is not None else model
    self.tokenizer = tokenizer
    self.image_processor = image_processor
    self.config = config
    self.raiden_sync_delegate = raiden_sync_delegate
    # The mesh the model's parameters are sharded over. Generation must run
    # inside it: the caller builds the model under `with mesh:` but that only
    # covers construction, and sample() runs later on the worker's event loop
    # with no mesh installed. Named-axis shardings then fail to resolve with
    #   "Resource axis: tp of P('tp', None) is not found in mesh: ()".
    # Mirrors the non-distributed path, which wraps its generate calls in
    # _get_mesh_and_logical_axis_rules_cm(Role.ROLLOUT).
    self.mesh = mesh if mesh is not None else _infer_mesh(transformer)

    # Sampling RNG stream for requests that do not carry their own seed.
    #
    # torch keeps a stateful global RNG, so vLLM gets this for free: an
    # unseeded request draws from the default generator, which advances on
    # every draw (v1/sample/ops/topk_topp_sampler.py's random_sample calls
    # q.exponential_() with no generator). JAX is functional -- a PRNGKey is a
    # value, and sampler.py reuses PRNGKey(0) for every unseeded call, so each
    # call replays the same stream and a GRPO group decodes identically.
    #
    # Hold the equivalent state here and split it per call. Seeded from
    # ROLLOUT_RNG_SEED when set, so a run can be reproduced end to end;
    # otherwise from OS entropy, so separate workers and restarts diverge.
    # jax.random has no global stateful RNG -- keys are values, so there is
    # nothing to advance implicitly. nnx.Rngs is the stateful equivalent in the
    # ecosystem this repo already uses, and holds the state for us.
    env_seed = os.getenv("ROLLOUT_RNG_SEED")
    root_seed = (
        int(env_seed)
        if env_seed is not None and env_seed.strip()
        else int.from_bytes(os.urandom(4), "little")
    )
    self._rngs = nnx.Rngs(root_seed)
    self.weight_sync_mode = getattr(
        config, "weight_sync_mode", weight_sync.WeightSyncMode.FALLBACK
    )
    self.enable_raiden = (
        self.weight_sync_mode == weight_sync.WeightSyncMode.RAIDEN
    )

    if self.enable_raiden and self.raiden_sync_delegate is None:
      from tunix.experimental.weight_sync import raiden_weight_sync_delegate  # pylint: disable=g-import-not-at-top

      self.raiden_sync_delegate = (
          raiden_weight_sync_delegate.RaidenWeightSyncDelegate()
      )

    if not self.enable_raiden and self.raiden_sync_delegate:
      logging.warning(
          "VanillaSamplerAdapter [%s] raiden_sync_delegate is set but"
          " enable_raiden is False.",
          self.server_id,
      )

    if self.transformer is not None and self.tokenizer is not None:
      self.sampler = self._build_generate_sampler(cache_config)
    else:
      self.sampler = None

  def _build_generate_sampler(
      self, cache_config: generate_sampler_lib.CacheConfig | int | None
  ) -> generate_sampler_lib.Sampler:
    """Helper to construct generate_sampler_lib.Sampler from model and tokenizer."""
    if isinstance(cache_config, generate_sampler_lib.CacheConfig):
      cache_cfg = cache_config
    else:
      cfg = getattr(self.transformer, "config", None)
      if cfg:
        num_layers = getattr(
            cfg, "num_layers", getattr(cfg, "num_hidden_layers", 4)
        )
        num_kv_heads = getattr(
            cfg, "num_kv_heads", getattr(cfg, "num_key_value_heads", 4)
        )
        head_dim = getattr(cfg, "head_dim", getattr(cfg, "head_dimension", 16))
        cache_size = (
            cache_config
            if isinstance(cache_config, int)
            else getattr(cfg, "max_position_embeddings", 1024)
        )
      else:
        num_layers = 4
        num_kv_heads = 4
        head_dim = 16
        cache_size = cache_config if isinstance(cache_config, int) else 1024

      cache_cfg = generate_sampler_lib.CacheConfig(
          cache_size=cache_size,
          num_layers=num_layers,
          num_kv_heads=num_kv_heads,
          head_dim=head_dim,
      )

    return generate_sampler_lib.Sampler(
        transformer=self.transformer,
        tokenizer=self.tokenizer,
        cache_config=cache_cfg,
        image_processor=self.image_processor,
    )

  def initialize(self) -> None:
    """Initializes sampler if needed."""
    if (
        self.sampler is None
        and self.transformer is not None
        and self.tokenizer is not None
    ):
      self.sampler = self._build_generate_sampler(None)

  def _unpadded_prompt_tokens(self, padded_tokens: Any) -> np.ndarray:
    """Returns sampler-tokenized prompt ids without backend left padding."""
    arr = np.asarray(padded_tokens, dtype=np.int32).reshape(-1)
    pad_id = getattr(self.tokenizer, "pad_token_id", None)
    if pad_id is None:
      pad_id = getattr(self.tokenizer, "eos_token_id", None)
    if not isinstance(pad_id, numbers.Integral):
      return arr
    non_pad = np.flatnonzero(arr != pad_id)
    if non_pad.size == 0:
      return np.zeros(0, dtype=np.int32)
    return arr[non_pad[0] :]

  # --- Lifecycle & Topology ---
  async def start(self, **kwargs) -> str | None | Any:
    """Starts the sampling engine or local loop."""
    del kwargs
    return True

  async def stop(self, **kwargs) -> str | None | Any:
    del kwargs
    return True

  async def pause(self, **kwargs) -> str | None | Any:
    """Pauses inference processing on this worker slice."""
    del kwargs
    return True

  async def resume(self, **kwargs) -> str | None | Any:
    """Resumes inference processing on this worker slice."""
    del kwargs
    return True

  def _next_seed(self) -> int:
    """Advances the worker's sampling RNG and returns the next seed.

    Calling the Rngs stream advances it, so consecutive requests draw from
    fresh keys and a fixed ROLLOUT_RNG_SEED reproduces the whole sequence.
    """
    return int(jax.random.randint(self._rngs(), (), 0, 2**31 - 1))

  async def get_mesh(self, **kwargs) -> Any:
    """Returns the underlying device mesh topology."""
    del kwargs
    if hasattr(self.sampler, "get_mesh"):
      return self.sampler.get_mesh()
    return None

  # --- Inference ---
  async def sample(
      self,
      sampling_requests: (
          base_sampler_lib.SamplingRequest
          | Sequence[base_sampler_lib.SamplingRequest]
          | Any
          | Sequence[Any]
      ),
      **kwargs,
  ) -> (
      base_sampler_lib.SamplingResponse
      | List[base_sampler_lib.SamplingResponse]
      | Any
  ):
    """Standard completion call using external Tunix JAX Sampler model."""
    if not self.sampler:
      raise RuntimeError(
          f"VanillaSamplerAdapter [{self.server_id}] sampler is not"
          " initialized."
      )

    if sampling_requests is None:
      raise ValueError("sampling_requests cannot be None.")

    if isinstance(sampling_requests, base_sampler_lib.SamplingRequest):
      requests: List[Any] = [sampling_requests]
      is_sequence = False
    elif isinstance(sampling_requests, (list, tuple)):
      requests = list(sampling_requests)
      is_sequence = True
    else:
      requests = [sampling_requests]
      is_sequence = False

    prompts = []
    max_gen_steps_list = []
    temps = []
    top_ps = []
    top_ks = []
    seeds = []
    return_logprobs_list = []
    return_logits_list = []
    beam_sizes = []

    for req in requests:
      prompt = req.prompt if hasattr(req, "prompt") else req
      prompts.append(prompt)
      sp = (
          req.sampling_params
          if hasattr(req, "sampling_params") and req.sampling_params is not None
          else base_sampler_lib.SamplingParams()
      )
      assert sp is not None

      max_gen_steps_list.append(sp.max_tokens)
      temps.append(sp.temperature)
      top_ps.append(sp.top_p)
      top_ks.append(sp.top_k)
      seeds.append(sp.seed)
      return_logprobs_list.append(sp.return_logprobs)
      return_logits_list.append(sp.return_logits)
      if sp.beam_size is not None:
        beam_sizes.append(sp.beam_size)

    max_generation_steps = (
        max(max_gen_steps_list) if max_gen_steps_list else 64
    )
    temperature = temps[0] if temps else 0.0
    top_p = top_ps[0] if top_ps else None
    top_k = top_ks[0] if top_ks else None
    seed = seeds[0] if seeds else None
    if seed is None:
      # Advance the worker's stream, mirroring torch's global RNG. An explicit
      # per-request seed is still honoured untouched, so seeded requests stay
      # reproducible.
      seed = self._next_seed()
    return_logprobs = any(return_logprobs_list) or kwargs.get(
        "return_logprobs", False
    )
    return_logits = any(return_logits_list) or kwargs.get(
        "return_logits", False
    )
    beam_size = beam_sizes[0] if beam_sizes else None

    # Enter the Mesh object itself, exactly as rl_cluster's
    # _get_mesh_and_logical_axis_rules_cm does via
    # stack.enter_context(role_to_mesh[role]).
    #
    # NB: do NOT "modernise" this to jax.set_mesh(mesh) despite the deprecation
    # warning jax 0.11 prints. The two are not interchangeable here: entering
    # the Mesh installs the concrete physical mesh that the legacy check_pspec
    # path uses when the model applies P('tp', None) inside jitted prefill,
    # whereas jax.set_mesh installs only the abstract mesh for sharding-in-
    # types. With jax.set_mesh the model still fails with
    #   ValueError: Resource axis: tp of P('tp', None) is not found in mesh: ().
    # Verified both ways against Qwen3-0.6B on a (fsdp=1, tp=8) mesh.
    mesh_cm = self.mesh if self.mesh is not None else contextlib.nullcontext()
    with mesh_cm:
      sampler_output = self.sampler(
          input_strings=prompts,
          max_generation_steps=max_generation_steps,
          temperature=temperature,
          top_p=top_p,
          top_k=top_k,
          seed=seed,
          beam_size=beam_size,
          return_logits=return_logits,
          return_logprobs=return_logprobs,
      )

    responses = []
    for i, req in enumerate(requests):
      req_id = getattr(req, "request_id", "")

      txt = (
          sampler_output.text[i]
          if isinstance(sampler_output.text, list)
          else sampler_output.text
      )
      toks = (
          sampler_output.tokens[i]
          if isinstance(sampler_output.tokens, list)
          else sampler_output.tokens
      )
      lps = None
      if sampler_output.logprobs and isinstance(sampler_output.logprobs, list):
        lps = sampler_output.logprobs[i]

      tok_ids = (
          np.array(toks, dtype=np.int32)
          if toks is not None
          else np.zeros(0, dtype=np.int32)
      )
      prompt_token_ids = self._unpadded_prompt_tokens(
          sampler_output.padded_prompt_tokens[i]
      )
      log_ps = np.array(lps, dtype=np.float32) if lps is not None else None

      responses.append(
          base_sampler_lib.SamplingResponse(
              request_id=req_id,
              text=txt,
              prompt_token_ids=prompt_token_ids,
              token_ids=tok_ids,
              logprobs=log_ps,
              finish_reason="stop",
          )
      )

    if is_sequence:
      return responses
    return responses[0]

  # --- Weight Synchronization ---
  async def get_transfer_status(self, req_id: str | Any, **kwargs) -> str | Any:
    """Queries status of an ongoing weight transfer or KV-cache migration."""
    del req_id, kwargs
    return "SUCCESS"

  async def get_load_info(self, **kwargs) -> base_sampler_lib.LoadInfo:
    """Returns best-effort local sampler load information."""
    del kwargs
    return base_sampler_lib.LoadInfo()

  # --- Weight Synchronization ---
  def _check_weight_sync_boundness(
      self,
  ):
    """Verifies that the Raiden delegate has been bound before executing sync phases."""
    if not self.enable_raiden:
      return

    if not self.raiden_sync_delegate.is_bounded():
      raise RuntimeError(
          f"VanillaSamplerAdapter [{self.server_id}] weight sync delegate"
          " is not bounded."
      )

  async def get_weight_sync_metadata(self, **kwargs) -> Any:
    """Returns sharding specs and layout metadata across devices for weights."""
    self._check_weight_sync_boundness()

    # In Raiden mode, retrieve transport endpoint and tensor shard layout
    # metadata.
    if self.enable_raiden:
      return await self.raiden_sync_delegate.get_weight_sync_metadata(**kwargs)

    # In Fallback mode, metadata query is not supported.
    raise NotImplementedError(
        f"VanillaSamplerAdapter [{self.server_id}] does not support"
        " get_weight_sync_metadata when Raiden is disabled."
    )

  async def bind_weight_sync(
      self,
      sync_request: base_sampler_lib.WeightSyncRequest | Any = None,
      **kwargs,
  ) -> Any:
    """Binds destination-side transport resources for weight transfer."""
    if self.enable_raiden:
      # In Raiden mode, register destination sampler transformer_state memory
      # buffers.
      if not hasattr(self.sampler, "transformer_state"):
        raise RuntimeError(
            f"VanillaSamplerAdapter [{self.server_id}] sampler does not expose"
            " transformer_state for Raiden weight sync."
        )

      if self.raiden_sync_delegate.is_bounded():
        # TODO(tunix-dev): bind_weight_sync should be called once, currently
        # it's called every time in weight_sync_coordinator, need to fix it.
        return True

      state = self.sampler.transformer_state
      return await self.raiden_sync_delegate.bind_weight_sync(
          sync_request=sync_request, state=state, **kwargs
      )
    # In Fallback mode, no transport binding is required.
    return None

  def get_target_state(self) -> Any:
    """Returns target state shape/dtype pytree for weight conversion."""
    if self.sampler is None:
      raise RuntimeError(
          f"VanillaSamplerAdapter [{self.server_id}] sampler is not initialized."
      )
    if hasattr(self.sampler, "get_target_state"):
      return self.sampler.get_target_state()
    if hasattr(self.sampler, "transformer_state"):
      state = self.sampler.transformer_state
      return jax.tree.map(
          lambda x: nnx.Param(jax.ShapeDtypeStruct(shape=x.shape, dtype=x.dtype)),
          state,
          is_leaf=lambda x: isinstance(x, nnx.Variable),
      )
    raise AttributeError(
        f"VanillaSamplerAdapter [{self.server_id}] cannot extract target_state."
    )

  async def pre_weight_sync(
      self,
      sync_request: base_sampler_lib.WeightSyncRequest | Any = None,
      **kwargs,
  ) -> str | None | Any:
    """Prepares staging handshake prior to policy weight update."""
    self._check_weight_sync_boundness()

    # In Raiden mode, execute the pre-synchronization barrier via delegate.
    if self.enable_raiden:
      self._log_param_stats("pre-sync")
      return await self.raiden_sync_delegate.pre_weight_sync(
          sync_request=sync_request, **kwargs
      )
    # In Fallback mode, acts as a no-op returning True.
    return True

  async def weight_sync(
      self,
      sync_request: base_sampler_lib.WeightSyncRequest | Any = None,
      **kwargs,
  ) -> str | None | Any:
    """Updates model weights in-place from the specified controller or request."""
    self._check_weight_sync_boundness()

    # Raiden mode: Invoke Raiden transport to stream weights into bound memory
    # buffers.
    if self.enable_raiden:
      return await self.raiden_sync_delegate.weight_sync(
          sync_request=sync_request, **kwargs
      )
    else:
      # Fallback mode: Directly assign source weights from sync_request.weights.
      if sync_request is None:
        raise ValueError(
            "VanillaSamplerAdapter Fallback mode [%s] weight_sync:"
            " sync_request is None."
            % self.server_id
        )
      if self.sampler and hasattr(self.sampler, "update_params"):
        weights = getattr(sync_request, "weights", None)
        if weights is None:
          raise ValueError(
              "VanillaSamplerAdapter [%s] weight_sync: weights not found"
              " in sync_request."
              % self.server_id
          )
        self.sampler.update_params(weights)
      else:
        raise RuntimeError(
            f"VanillaSamplerAdapter [{self.server_id}] does not support"
            " Raiden weight sync, while the fallback path missing required"
            " components."
        )
      return True

  def _log_param_stats(self, phase: str) -> None:
    """Logs summary statistics of a few rollout parameters.

    Raiden streams bytes straight into the bound buffers, so a transfer can
    commit while writing the wrong layout. Name/shape/item_size preflight does
    not catch that. Comparing these numbers either side of a round, and against
    the trainer, shows whether the weights that landed are plausible.
    """
    try:
      state = self.sampler.transformer_state
      leaves = jax.tree.leaves(state)
      for idx in (0, len(leaves) // 2):
        if idx >= len(leaves):
          continue
        v = getattr(leaves[idx], "value", leaves[idx])
        arr = jax.numpy.asarray(v, dtype=jax.numpy.float32)
        logging.info(
            "[weight-sync %s] leaf[%d] shape=%s mean=%.6g std=%.6g"
            " absmax=%.6g finite=%s",
            phase,
            idx,
            getattr(v, "shape", None),
            float(jax.numpy.mean(arr)),
            float(jax.numpy.std(arr)),
            float(jax.numpy.max(jax.numpy.abs(arr))),
            bool(jax.numpy.all(jax.numpy.isfinite(arr))),
        )
    except Exception as e:  # pylint: disable=broad-except
      logging.warning("[weight-sync %s] could not read param stats: %r", phase, e)

  async def post_weight_sync(
      self,
      sync_request: base_sampler_lib.WeightSyncRequest | Any = None,
      **kwargs,
  ) -> str | None | Any:
    """Finalizes and switches active policy weights after transfer completion."""
    # Raiden mode: Commit newly transferred weights and execute post-sync
    # barrier.
    if self.enable_raiden:
      result = await self.raiden_sync_delegate.post_weight_sync(
          sync_request=sync_request, **kwargs
      )
      self._log_param_stats("post-sync")
      return result
    # Fallback mode: acts as a no-op returning True.
    return True

  # --- KV-cache Migration ---
  async def migrate_kv_cache(
      self,
      source_server_id: str,
      target_server_id: str,
      token_ids: List[int],
      **kwargs,
  ) -> bool:
    """Triggers Raiden P2P KV-cache transfer across TPU slices."""
    del source_server_id, target_server_id, token_ids, kwargs
    return True
