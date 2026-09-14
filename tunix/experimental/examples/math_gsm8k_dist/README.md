# Distributed GSM8K GRPO (Qwen3-1.7B)

`launcher.sh` starts an orchestrator plus trainer / rollout / (optional)
reference workers on one host. A converging run on a v5e-8 is:

```bash
SAMPLER=vanilla WEIGHT_SYNC_MODE=raiden MAXTEXT_MODEL_NAME="" \
RUN_INFERENCE_NODE=1 CHAT_PARSER=raw EOS_TOKENS='<|im_end|>,<|endoftext|>' \
TRAINER_TPU_CHIPS=0,1,2,3 TRAINER_TP=4 TRAINER_CHIP_BOUNDS=1,4,1 \
ROLLOUT_TPU_CHIPS=4,5 INFERENCE_TPU_CHIPS=6,7 \
MAX_STEPS=100 LEARNING_RATE=1e-6 WARMUP_STEPS=10 TRAIN_MICRO_BATCH_SIZE=2 \
USE_ROLLOUT_LOGPS=false \
EVAL_EVERY_N_STEPS=1000000 CHECKPOINT_SAVE_INTERVAL_STEPS=1000000 \
bash launcher.sh
```

## What still has to be overridden, and why

Everything not listed here is left at `launcher.sh`'s default (`BETA=0.04`,
`EPSILON=0.2`, AdamW `0.9 / 0.999 / 1e-8`, `WEIGHT_DECAY=0.01`,
`MAX_GRAD_NORM=1.0`, `SCHEDULE_TYPE=warmup_cosine_decay_schedule`,
`LR_DECAY_STEPS=500`, `BATCH_SIZE=4`, `NUM_GENERATIONS=8`, prompt/response
1024, `SEED=42`, `SHUFFLE=true`, `USE_LORA=0`).

| override | default | why |
|---|---|---|
| `SAMPLER=vanilla` | `inprocess_vllm` | the stack under test; vLLM needs a separate install |
| `WEIGHT_SYNC_MODE=raiden` | `none` | disaggregated trainer/rollout must sync weights every step |
| `MAXTEXT_MODEL_NAME=""` | lowercased `MODEL_NAME` | a non-empty value selects the MaxText trainer |
| `TRAINER_TPU_CHIPS=0,1,2,3`, `TRAINER_TP=4`, `TRAINER_CHIP_BOUNDS=1,4,1` | `0,1`, `2`, `1,2,1` | 4 chips for the trainer |
| `ROLLOUT_TPU_CHIPS=4,5` | `2,3` | rollout on its own 2 chips |
| `RUN_INFERENCE_NODE=1`, `INFERENCE_TPU_CHIPS=6,7` | `0`, empty | `BETA != 0` needs a reference worker for the KL term. `launcher.sh` already refuses to start otherwise, and `INFERENCE_ADDR` is the alternative if the reference runs elsewhere |
| `MAX_STEPS=100` | `1` | |
| `LEARNING_RATE=1e-6` | `2.0e-7` | see below |
| `WARMUP_STEPS=10` | `LR_DECAY_STEPS/10` = 50 | at the default the whole 100-step run is warmup |
| `TRAIN_MICRO_BATCH_SIZE=2` | `1` | 2 rows × 2048 tokens fits and roughly halves step time |
| `USE_ROLLOUT_LOGPS=false` | `true` | what `examples/math_gsm8k/qwen3_grpo_demo.py` uses: the trainer re-scores before the update instead of taking the ratio against the sampler's logps |
| `EVAL_EVERY_N_STEPS=1000000` | `50` | this flag drives the **trainer node's** SFT-style eval loop, which the RL orchestrator does not use; leaving it at 50 only costs time |
| `CHECKPOINT_SAVE_INTERVAL_STEPS=1000000` | `1` | a checkpoint per step is ~6 GB per 50 steps |

`CHAT_PARSER=raw` and `EOS_TOKENS='<|im_end|>,<|endoftext|>'` are also required
(see "Defaults that are recipe-specific" below); they are proposed as launcher
defaults separately — see "Defaults that are recipe-specific"
below.

## Measured behaviour

100 steps, 4 prompts × 8 generations = 32 rollouts and one optimizer update per
step, on a v5e-8 (~25 s/step once clipping falls):

| by 20 steps | 0–19 | 20–39 | 40–59 | 60–79 | 80–99 |
|---|---|---|---|---|---|
| `rewards/mean` | 0.498 | 0.827 | 0.840 | 0.889 | 0.866 |
| `completions/clip_ratio` | 0.335 | 0.048 | 0.020 | 0.014 | 0.006 |
| `rollout/completion_length_mean` | 645 | 330 | 296 | 280 | 287 |
| `advantage/nonzero_frac` | 0.892 | 0.475 | 0.450 | 0.400 | 0.287 |

Reading the numbers: the reward gain is the policy learning to **close its tags
and stop inside the budget** — over the same run the format rate goes
82.7% → 99.5% while answer accuracy moves 85.0% → 85.8%. By step 40 most groups
are degenerate (all 8 samples identical reward → group-relative advantage 0),
which is why the curve flattens: the training signal is exhausted, not stalled.

`LEARNING_RATE=2.0e-7` (the launcher default, and the single-host recipe's
value) is correct for a 200+ step run but does not move a 100-step one: the
cumulative LR over 100 warmup-heavy steps is under 10% of the reference
recipe's budget, and the curve stays flat at ~0.31.

## Defaults that are recipe-specific

`launcher.sh` only runs the GSM8K VTC recipe, so two of its settings encode that
recipe rather than a neutral choice:

- **`CHAT_PARSER=raw`** — the prompt ends with an opened `<reasoning>` tag for
  the model to continue. A chat template would close the user turn and open an
  assistant turn after it, so the format check could never pass.
- **`EOS_TOKENS='<|im_end|>,<|endoftext|>'`** — the tokenizer's own
  `eos_token_id` is `<|im_end|>` (151645), so an unset `--eos_tokens` leaves the
  sampler stopping on exactly the token this recipe never produces. Measured
  over 16 raw-prompt completions, stopping only on `<|im_end|>`: **0/16**
  terminate, **0/16** contain `<|im_end|>` anywhere, and 4/16 contain
  `<|endoftext|>` *mid-stream* with a median of 618 (max 764) further tokens
  generated after it. Adding `<|endoftext|>` makes those 4 terminate and no
  rollout carries a mid-stream `<|endoftext|>` any more. The tokens after the
  model has already finished are not just wasted compute: the rollout hits the
  response budget, so the collect engine marks it
  `MAX_CONTEXT_LIMIT_REACHED` and skips `env.step`, and it is never scored.
  `<|im_end|>` stays in the set because it is the terminator under
  `CHAT_PARSER=auto`, and because vLLM merges both ids from
  `generation_config.json` regardless.

## Knobs this run needs that are not in this tree

These need changes that are not in this tree; the measured run above used them:

- **fp32 master weights for the actor** (`--model_dtype float32` on the trainer
  node plus a bf16 copy for Raiden, which pairs tensors by name *and* byte
  width). The single-host recipe loads its actor in fp32 and only the reference
  in bf16. Without it the actor is bf16, where a 2e-7 Adam update is a fraction
  of one ULP; bf16 convergence at 1e-6 has not been measured here.
- `generation/completions/clip_ratio` and a truthful `rollout/success_rate`,
  both derived from the collect engine's `TrajectoryStatus`.
- Held-out eval on the GSM8K test split (the numbers above are train reward).
- `kl` / `entropy` / `pg_clipfrac` from the loss, and the sampler-vs-trainer
  log-prob agreement metrics.
