# Raiden drops shards of norm weights

Branch `sizhi-metrics-pr2023` · 2026-08-31 · Qwen3-0.6B on v5litepod-8

Investigation into why GSM8K rollouts collapse to token soup one step after
every weight sync. One fix holds, one headline claim is retracted, and the
surviving finding is narrower than first reported but measured.

---

## Bottom line

The goal was a converging reward curve in 4-8 hours. **That was not
delivered.** Raiden weight sync corrupts the rollout on every configuration
tried, so no long run was worth starting.

What exists instead: the preflight bug that had blocked every run since
2026-08-27 is fixed and verified, and the remaining defect is characterised
precisely enough to hand to whoever owns `tpu_sync`.

---

## Claim ledger

Six claims were made over the session. Three did not survive scrutiny. They
are listed because the retractions matter as much as the findings.

### HOLDS - Preflight fix (`83b8040e`)

112 phantom "sharding spec differs" failures -> 0 blocking, 0 advisory.
Verified by replaying the production manifest offline, plus 76 passing tests.
Not run-dependent, so nondeterminism cannot explain it. This unblocked weight
sync for the first time since 08-27 -- and revealed that the preflight had been
masking a real transfer defect.

### SUPPORTED AT tp=2 ONLY - Raiden drops whole shards of `P('tp')`-sharded rank-1 norm weights

Strong and repeated at `tp=2` (5 runs): only norm tensors are ever affected,
loss ratios centre on 0.500. The generalisation to `tp=1` and `tp=4` rests on
**a single run each** and should not be filed as reproduced until repeated.

### RETRACTED - Tiling fix recovered 72% of weight mass (`02e09f3c`)

Claimed `-72% -> -0.18%` from two runs 20 minutes apart. A controlled A/B on
one binary gives `-0.12%` (old path) vs `-0.35%` (new path) -- no effect. The
`-71.7%` outlier is not reproducible. The code change may still be more
correct, but it does not fix what was claimed.
**The commit message needs amending.**

### RETRACTED - "This is a `tpu_sync` bug"

Asserted before eliminating our own code. Several Tunix-side suspects were
found afterwards. Elimination narrows it toward the transport, but that is not
the same as demonstrating it.

### RETRACTED - "Checkpoint flags are silently ignored"

Raised from an orbax debug line showing `save_interval_steps=1`. The flags are
honoured correctly via `FixedIntervalPolicy(50)` and `LatestN(1)`. No bug, no
disk risk.

### OVERSTATED - "The drops are nondeterministic"

Between rounds the affected set shifts (Jaccard 0.55), but **27 of 33 shared
tensors lose the identical fraction every round**. It is mostly deterministic.
This misreading is what motivated the buffer-lock hypothesis, which then
measured as a dead end.

---

## The surviving finding

Arm B, `fsdp=1/tp=2`, per-tensor checksums on both sides of one sync round:

| Tensor class              | In model | Materially wrong | Rate |
|---------------------------|---------:|-----------------:|-----:|
| norm weights (rank-1)     |      113 |           **53** |  47% |
| everything else (rank>=2) |      197 |            **0** |   0% |

Every projection matrix, MLP weight and the embedding table arrive
**bit-exact**. Only norms break. The loss ratios say how:

```
0.000 x8    <- both shards lost
0.372 0.400 0.428 0.441 0.446 0.449 0.453 0.462 0.470 0.472
0.478 0.481 0.481 0.481 0.482 0.484 0.486 0.486 0.488 0.491
0.497 0.498 0.500 0.510 0.511 0.511 0.513 0.519 0.525 0.530 ...

median = 0.500     mean = 0.507     41 of 45 within 0.40-0.60
```

A median of exactly **0.500 at tp=2** is one shard of two never arriving.

Norms are the only rank-1 tensors in the model -- `tunix/models/qwen3/model.py:125`
sets `rms_norm_weight=P('tp')`, splitting them along their only axis, so a
`k_norm` shard is 64 float32s. Every other tensor is 2-D with dims that are
clean multiples of 8 and 128.

**Why this is fatal despite being ~0.3% of weight mass:** RMSNorm scales every
channel. A zeroed `layers[0].input_layernorm` or
`layers[27].post_attention_layernorm` kills that sublayer outright. Result:
entropy 4.97 nats, no EOS, 512-token output, `distinct_rewards=1`,
`grad_norm 0` -- zero advantage, so training cannot recover.

---

## Measured deltas, every run

Destination vs source `__grand_total__` (sum of `abs()` over all 310 tensors).
Source is 13,217,973 throughout.

| Run             | Mesh      | skip_tiling | Destination   | Delta      |
|-----------------|-----------|-------------|--------------:|-----------:|
| trace+checksum  | fsdp1/tp4 | all-False   |    13,213,279 |     -0.04% |
| isolate-tp2     | fsdp1/tp2 | all-False   | **3,739,804** | **-71.7%** |
| fix-derived     | fsdp1/tp2 | derived     |    13,193,767 |     -0.18% |
| tp1-no-parallel | fsdp1/tp1 | derived     |    13,206,413 |     -0.09% |
| buffer-lock     | fsdp1/tp2 | derived     |    13,191,707 |     -0.20% |
| A/B arm A       | fsdp1/tp2 | all-False   |    13,201,961 |     -0.12% |
| A/B arm B       | fsdp1/tp2 | derived     |    13,171,642 |     -0.35% |

Every run lands between -0.04% and -0.35% except one outlier at -71.7%. Arm A
repeats that outlier's exact configuration and returns -0.12%. That outlier is
the entire basis of the retracted tiling claim.

---

## Ruled out

| Suspect                            | Test                             | Result                              |
|------------------------------------|----------------------------------|-------------------------------------|
| Mesh / sharding mismatch           | preflight across all 310 tensors | identical specs, 0 advisories       |
| Tensor parallelism as trigger      | tp=4, tp=2, tp=1                 | corrupts at all three               |
| skip_tiling declaration            | controlled A/B, one binary       | no effect                           |
| unsafe_skip_buffer_lock            | lock enabled                     | no change, no deadlock              |
| Async h2d completion               | re-read 9 s later                | identical -- copy completes, wrong  |
| Precision / bf16 round-trip        | per-tensor equality              | 302 bit-exact -- not rounding       |
| Trainer updating between snapshots | Adam step <= 2e-7/param          | 130x too small to explain           |
| Preflight fix & log changes        | corruption predates both         | not the cause                       |

---

## How to reproduce in five minutes

```bash
VERIFY_WEIGHTS=true VERIFY_WEIGHTS_SAMPLE=400   # both sides, per-tensor
SAMPLER=vanilla WEIGHT_SYNC_MODE=raiden SYNC_WEIGHTS=1
TRAINER_FSDP=1 TRAINER_TP=2   ROLLOUT_FSDP=1 ROLLOUT_TP=2
MODEL_NAME=Qwen3-0.6B MAX_STEPS=2
```

Compare `source checksums` in `trainer.log` against `destination checksums` in
`rollout.log`. The invariant is simply: **after a transfer, destination must
equal source.** Both are `RaidenSynchronizer.checksums()` -- the same function
on both sides.

Knobs added this session for A/B-ing:
- `RAIDEN_HOST_STAGED_TILING` -- `1` forces all-False skip_tiling, `0` derived
- `RAIDEN_SKIP_BUFFER_LOCK` -- `0` enables the buffer lock
- `VERIFY_WEIGHTS_SAMPLE` -- tensors per checksum dump (use 400 for all)

**Do not call `get_host_buffer()` from inside a worker.** It segfaulted the
trainer, leaving it wedged in uninterruptible (`D`) state on the TPU driver,
which survives SIGKILL and blocks all further runs until the driver times out
on its own.

The cause is *not* established. An earlier note here claimed the trainer "has
no host staging buffer because `host_stage=False`" -- that is wrong.
`host_stage` only controls whether JAX arrays are pulled to CPU before binding
(`to_host_cpu_state`, a pathways-proxy workaround); the C++ WeightSynchronizer
has a host buffer either way, since `d2h()` is defined as "Device-to-Host copy
of current weights to Host buffer". Two untested candidates for the real cause:
`d2h()` is asynchronous and the buffer was read mid-DMA, or the
`layer_idx`/`shard_idx` ranges taken from `num_layers`/`num_shards` were out of
range for a zero-copy accessor that does not bounds-check.

---

## Commits on the branch

| SHA        | Subject                                                      | Status              |
|------------|--------------------------------------------------------------|---------------------|
| `83b8040e` | Preflight: pair mesh_axes with the unit mesh; placement advisory | verified         |
| `c1858d2c` | Launcher: resolve max_steps from dataset; checkpoint interval | uncontroversial     |
| `02e09f3c` | Stop declaring device-resident weights untiled               | message overstates  |
| `a5018d50` | Make unsafe_skip_buffer_lock settable + negative result       | diagnostic knob     |

Local only, not pushed.

`origin/main` carries **no fix** for any of this. Its GSM8K commits are the
copybara twin of work already on this branch -- `score_gsm8k_completion` is
line-for-line equivalent, differing only in variable names -- and it still
passes `make_host_staged_transfer_options()` unconditionally. A merge conflicts
in 7 files.

---

## Next

- **Repeat tp=4 and tp=1** two or three times each. Both are currently n=1, and
  the shard-granularity mechanism is only properly established at tp=2.
  Predicted ratios: 0.75/0.5/0.25 at tp=4; all-or-nothing at tp=1.
- **Amend `02e09f3c`** so the repo stops carrying the -72% claim.
- **Correct the memory notes** -- they record the shard-loss theory as refuted,
  which arm B reverses.
- **Merge main** -- take its canonical form for the example files, re-apply the
  weight-sync work. Note it renames `WEIGHT_SYNC_BACKEND` -> `WEIGHT_SYNC_MODE`
  and maps `fallback` to `NullHandler`.

A converging curve is not reachable through Raiden until the norm-shard loss is
fixed. The one untested escape is `WEIGHT_SYNC_MODE=fallback` on the vanilla
sampler, which routes through `reshard_pytree` instead of the byte-streaming
path -- explicitly excluded this session.

---

*All figures from `RaidenSynchronizer.checksums()` on both sides of a live
round. Qwen3-0.6B, 310 tensors (113 norm, 197 other).*
