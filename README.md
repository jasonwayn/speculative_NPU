# speculative_NPU

This repository documents my work on porting **DFlash**, a block-diffusion speculative decoding method, to the **Rebellions ATOM+ NPU** and optimizing it under a static-graph execution model.

Unless stated otherwise, all NPU measurements were taken on a **single ATOM+ card**. Both the target model and the drafter run on the same card from prefill through decoding.

The main goal of this project was not just to make DFlash run on the NPU, but to understand what actually limits speculative decoding on a compiled accelerator: static graph shapes, KV-cache handling, weight streaming, host overhead, runtime restrictions, and numerical correctness.

---

## 1. Setup

| Item                              | Configuration                                                                          |
| --------------------------------- | -------------------------------------------------------------------------------------- |
| NPU                               | Rebellions ATOM+ (RBLN-CA22) × 4, 15.7 GiB per card                                    |
| Software                          | `rebel` / `optimum-rbln` / `rebel-compiler` 0.10.2, PyTorch 2.9.1+cpu, Python 3.13     |
| Target model                      | Qwen3-4B, fp16, `max_seq_len=4096`                                                     |
| Drafter                           | DFlash, 5 layers, block size B=16                                                      |
| Measured hardware characteristics | ~225 GB/s weight streaming, ~61 µs dispatch overhead, 17–18 W idle, 48–69 W under load |

---

## 2. Important notes about the reported results

Two issues were discovered after the first round of experiments.

### Host-side bottlenecks

The original measurements were affected by three host-side problems:

1. `NTHREADS=2` caused PyTorch OpenMP threads to busy-wait and starve the RBLN runtime.
2. The scorer physically copied GQA heads, reaching 268 MB at a 16K context.
3. Key-storage buffers were reallocated every decoding round.

After fixing them, stock execution improved by roughly **8–9%**, while CMR improved by **3.5–3.7×**. The acceptance length, `tau`, did not change.

At a context length of 16,384, NPU CMR became beneficial for the first time, reaching approximately **1.02× to 1.09×** the stock throughput.

See:

`docs/HOST_THREAD_STARVATION.md`

### RoPE correctness issue

I also found a long-generation accuracy bug in the compiled trigonometric operations used by RoPE.

Before the fix, drafter quality degraded as generation became longer. This reduced `tau` and therefore made NPU throughput look worse than it actually was.

For example, on `math500`:

* old NPU `tau`: **4.295**
* corrected NPU `tau`, preliminary 5-sample result: **7.293**
* GPU `tau`: **6.661**

The older `slim_results` tables and the GPU comparison below were measured before this fix. They are kept here as historical optimization results, but should not be used as final post-fix performance numbers.

Corrected measurements are stored under:

* `results/rr_results/`
* `results/rr_long/`

Details of the bug are in:

`docs/ROPE_ROOT_CAUSE.md`

---

## 3. DFlash performance on ATOM+

The following table uses the original 20-sample DFlash paper workloads with `MAXNEW=2048`, B=16, and one NPU card.

These measurements are useful for comparing the optimization stages, but the `tau` and throughput values were collected before the RoPE fix described above.

| Dataset   |   tau |  Draft |  Verify | lm_head |       Round |      tok/s |   J/token |
| --------- | ----: | -----: | ------: | ------: | ----------: | ---------: | --------: |
| gsm8k     | 5.840 | 5.3 ms | 38.8 ms |  9.0 ms | **59.5 ms** |  **98.12** | **0.593** |
| humaneval | 6.258 |    5.2 |    39.1 |     9.0 |    **59.9** | **104.45** | **0.561** |
| math500   | 4.295 |    5.7 |    39.7 |     9.0 |    **62.0** |  **69.32** | **0.832** |
| mbpp      | 5.730 |    5.4 |    39.8 |     9.2 |    **62.6** |  **91.51** | **0.633** |
| mt-bench  | 2.287 |    5.6 |    40.1 |     9.1 |    **62.6** |  **36.51** | **1.570** |

One useful observation is that the **round time stays almost flat at 59–63 ms across all five datasets**.

The large throughput differences mainly come from differences in `tau`, or the number of accepted tokens per speculative round. For example, `mt-bench` is not slow because the hardware takes longer to execute a round. Its round time is almost identical to the other workloads, but its `tau` is only 2.287.

---

## 4. Optimization progress

The table below shows the main optimization steps on `gsm8k`.

|           |    Original | + Drafter KV cache | + Unaligned resume | + Dual graph | + Output slicing |
| --------- | ----------: | -----------------: | -----------------: | -----------: | ---------------: |
| Draft     |     15.7 ms |                5.4 |                5.2 |          5.3 |          **5.3** |
| Verify    |        63.8 |               63.4 |               47.7 |         41.2 |         **38.8** |
| lm_head   |         9.2 |                9.1 |                8.9 |          8.9 |          **9.0** |
| **Round** | **96.5 ms** |               86.5 |               67.3 |         62.3 |      **59.5 ms** |
| tau       |       4.628 |              5.699 |              5.699 |        5.840 |            5.840 |
| **tok/s** |       47.96 |              65.86 |              84.71 |        93.77 |        **98.12** |
| J/token   |       1.236 |              0.917 |              0.732 |        0.611 |        **0.593** |

For `gsm8k`, the final implementation reduced the decoding round from **96.5 ms to 59.5 ms**, a **1.62× reduction in round latency**.

Measured throughput increased from **47.96 to 98.12 tok/s**, and energy per generated token dropped from **1.236 to 0.593 J/token**.

The throughput improvement is larger than the raw latency improvement because `tau` also changed during the optimization process. For hardware-only comparisons, **round latency is the more meaningful metric**.

Other workloads showed similar latency improvements:

* `math500`: **126.1 → 62.0 ms**
* `mt-bench`: **89.3 → 62.6 ms**

More detailed measurements are in:

`docs/RESULTS.md`

---

## 5. Comparison with an RTX PRO 6000 Blackwell

For comparison, I ran the same DFlash configuration on an **RTX PRO 6000 Blackwell** with 20 samples, `MAXNEW=2048`, B=16, and batch size 1.

The GPU was otherwise idle during measurement.

Again, the NPU results in this table were measured before the RoPE fix, so throughput comparisons involving `tau`, especially `math500`, should be treated with caution.

| Dataset   | NPU tok/s | GPU tok/s | Throughput gap | NPU round | GPU round | NPU J/token | GPU J/token |
| --------- | --------: | --------: | -------------: | --------: | --------: | ----------: | ----------: |
| gsm8k     |     98.12 |    257.26 |           2.6× |   59.5 ms |   22.9 ms |   **0.593** |       0.692 |
| humaneval |    104.45 |    296.80 |           2.8× |      59.9 |      20.9 |   **0.561** |       0.721 |
| mbpp      |     91.51 |    276.38 |           3.0× |      62.6 |      20.4 |   **0.633** |       0.792 |
| mt-bench  |     36.51 |    116.52 |           3.2× |      62.6 |      20.8 |   **1.570** |       1.965 |
| math500   |     69.32 |    310.46 |           4.5× |      62.0 |      21.5 |       0.832 |   **0.668** |

The more stable comparison is round latency.

Across the five workloads, the NPU round is consistently about **2.7–3.0× slower** than the GPU round. Before the NPU optimizations, this gap was approximately 4.6×.

The throughput gap varies more because it also depends on `tau`.

Energy tells a different story. In these measurements, the NPU consumed less energy per generated token on four of the five datasets. The NPU generally ran around 57 W, while the GPU consumed roughly 180–230 W during DFlash execution.

### DFlash on NPU vs. autoregressive decoding on GPU

An interesting comparison is NPU DFlash against normal autoregressive generation on the GPU.

| System             |      Throughput |   J/token |    Power |
| ------------------ | --------------: | --------: | -------: |
| GPU autoregressive |     69.01 tok/s |     3.724 |    258 W |
| **NPU DFlash**     | **98.12 tok/s** | **0.593** | **57 W** |

On this `gsm8k` measurement, NPU DFlash was **1.42× faster** than GPU autoregressive decoding while using about **6.3× less energy per token**.

For reference, DFlash itself gave a **3.73× speedup** over autoregressive decoding on the GPU.

### The old `math500` discrepancy

Before fixing RoPE, `math500` had a large mismatch in `tau`:

* NPU: 4.295
* GPU: 6.661

The other four datasets were within roughly 1–5%.

Since `tau` should mostly depend on model behavior rather than accelerator speed, this was a sign that something was wrong with the NPU execution rather than a hardware performance difference.

The later RoPE investigation confirmed a numerical problem during long generation. After the fix, preliminary NPU measurements reached a `tau` of 7.293.

This means the old **4.5× `math500` throughput gap should not be interpreted as an NPU hardware gap**. The corresponding round-time gap was only about 2.9×, consistent with the other datasets.

---

## 6. Where the NPU time goes

The target model streams its weights from device memory during execution.

A decoding round reads approximately **9.11 GB** of weights. With a measured streaming bandwidth of approximately **225 GB/s**, weight reads alone require around:

```text
9.11 GB / 225 GB/s ≈ 40.5 ms
```

The optimized `gsm8k` round takes **59.5 ms**.

This means a large part of the remaining latency is already close to the weight-streaming floor. Under the same model and decoding structure, the rough fully optimized upper bound is around **144 tok/s**.

This also explains why reducing graph size indefinitely does not help: smaller graphs increase the number of executions, but each execution still has to pay a substantial weight-streaming cost.

---

## 7. Final execution structure

The final implementation uses three main static-graph stages.

```text
[1] Target prefill
    input_ids [1, 256]
    Executed once per sample

[2] Drafter
    Append: hidden state [1, 16, 12800] -> update drafter KV cache
    Block:  noise [1, 16, 2560]        -> generate 15 proposals

[3] Target verify
    input_ids [1, 17]
    Executed every speculative round
    16 candidate tokens + 1 padding position
```

The target prefill and target verification graphs use different input chunk sizes, but **share the same target KV cache**.

I verified numerical consistency between the two paths with cosine similarity:

```text
cosine similarity = 0.999982
```

The drafter uses separate Append and Block graphs, which share their own KV cache.

The KV caches remain resident on the device:

```text
Target:
[1, 8, 4096, 128] × 72

Drafter:
[1, 8, 4096, 128] × 10
```

This structure allows the decoding loop to remain dynamic at the algorithm level while each individual accelerator graph stays statically compiled.

---

## 8. Optimizations

### 1. Keep the drafter KV cache on device

The initial implementation resent the drafter context every round.

Making the drafter stateful and keeping its KV cache resident reduced draft latency from:

```text
15.7 ms -> 5.4 ms
```

At a 16K context, the effect was much larger:

```text
158.9 ms -> 13.6 ms
```

This was one of the most important changes for making long-context speculative decoding practical.

### 2. Resume from unaligned offsets

The original wrapper only allowed prefix-cache reuse at chunk-aligned positions.

This forced the model to recompute the tail of the previous chunk whenever the accepted prefix ended at an arbitrary position.

The underlying kernel did not actually require this restriction.

After bypassing the Python-side guard, verification latency dropped from:

```text
63.4 ms -> 47.7 ms
```

### 3. Reduce the verify chunk from 64 to 17

DFlash proposes at most 16 tokens per round.

With a verify graph of length 64, most of the graph was padding. I compiled a 17-token verify graph instead:

```text
16 proposed tokens + 1 extra position
```

Verification latency dropped further:

```text
47.7 ms -> 41.2 ms
```

### 4. Increase the prefill chunk from 64 to 256

The verify graph benefits from being small, but prefill benefits from fewer invocations.

I therefore compiled target graphs with different sequence lengths while sharing the same KV cache:

```text
Prefill: 256 tokens
Verify:   17 tokens
```

The larger prefill chunk reduced the number of prefill calls by 4× without forcing verification to use a large graph.

### 5. Return only the hidden states DFlash needs

With `output_hidden_states=True`, the vendor wrapper returned hidden states from all 37 layers to the host.

DFlash only needed six of them.

Reducing the outputs from:

```text
37 hidden states -> 6 hidden states
```

reduced verify latency from:

```text
41.2 ms -> 38.8 ms
```

This was mostly unnecessary device-to-host output traffic introduced by the wrapper rather than a limitation of the accelerator itself.

### 6. Move RoPE angle reduction to the host

Long-generation experiments showed that compiled `sin` and `cos` operations became numerically inaccurate for large input values.

The error grew with the magnitude of the trigonometric input and eventually destroyed drafter accuracy.

I moved the angle-range reduction step to the host before sending values into the compiled graph.

For long generation, this recovered `tau` from approximately:

```text
1.93 -> 6.16–6.46
```

and improved throughput by roughly:

```text
2.5–2.8×
```

The full root-cause analysis is in:

`docs/ROPE_ROOT_CAUSE.md`

---

## 9. Vendor/runtime limitations found during the port

I documented ten main issues in:

`docs/VENDOR_LIMITS.md`

The short version is:

1. **Prefill length must be a chunk multiple**

   * Python-side check only
   * underlying kernel can run without it

2. **Unaligned prefix-cache resume is rejected**

   * harmless with `use_attention_mask=False`
   * Python-side restriction can be bypassed

3. **`logits_to_keep > 1` is unsupported**

4. **Argmax cannot be compiled and top-k cannot be executed correctly**

5. **`decoder_batch_sizes` breaks for small batches**

   * `block_tables` shape mismatch

6. **The prefill interface has no batch dimension**

7. **int4/int8 quantization silently produces incorrect output**

   * no usable calibration API was available

8. **`output_hidden_states=True` copies every layer output back to the host**

   * can be bypassed by exposing only the required layers

9. **`mark_static_address` requires an undocumented three-argument form**

10. **Compiled `sin` / `cos` become badly inaccurate for large arguments**

    * maximum error observed relative to CPU fp32 was on the order of tens of millions
    * worked around by doing angle reduction on the host

The most difficult problems were not the restrictions that raised errors.

They were the cases where the runtime **successfully executed and returned numerically wrong results**.

For example, an early stateful-drafter implementation called PAGED attention with `q_len != k_len`. The graph ran without an exception, but the resulting cosine similarity was only about:

```text
0.57
```

This made explicit numerical equivalence checks necessary throughout the port.

---

## 10. Things that did not help

Several optimizations that looked promising either had no effect or made performance worse.

| Attempt                  | Result                                                                            |
| ------------------------ | --------------------------------------------------------------------------------- |
| Flash attention          | **5–15% slower**; partition size 8192 was much larger than the working context    |
| Chunk sizes 32 / 16 / 8  | Slower because each invocation still pays roughly the same weight-streaming floor |
| Verify chunks 128 / 256  | Slower                                                                            |
| Wider Append graph       | Not yet tested; initial context loading still requires P/16 calls                 |
| Removing `.float()`      | No measurable difference; earlier claim was incorrect                             |
| int4 / int8 quantization | Numerical output collapsed                                                        |

One general lesson from these experiments is that GPU-style intuition does not always transfer to this accelerator.

Reducing arithmetic or graph size is not necessarily useful when **weight streaming dominates each invocation**.

---

## 11. Measurement discipline

Measurements on this server have to be run with only one active experiment.

Although the NPU cards themselves are independent, they share host resources:

* 8 CPU cores
* 64 GiB host memory limit
* approximately 19.5 GiB RSS per worker
* serialized model loading through `flock`

Running experiments on different cards at the same time caused severe host contention.

For example, contaminated `humaneval` and `mbpp` runs increased verify latency from roughly:

```text
63.9 ms -> 225–245 ms
```

For reliable numbers, all measurements in this project therefore use:

```text
one NPU card
one experiment
no concurrent accelerator workload on the server
```

---

## 12. Repository structure

| Path                                                               | Description                                                                                                                                     |
| ------------------------------------------------------------------ | ----------------------------------------------------------------------------------------------------------------------------------------------- |
| [`bench/`](bench/)                                                 | Benchmark harnesses for each optimization stage: `bench2.py` → `bench_sf_paper.py` → `bench_sf_ua.py` → `bench_sf_dual.py` → `bench_sf_slim.py` |
| [`compile/`](compile/)                                             | Graph compilation scripts. `slim_1_256.py` contains the final 17-token verify + 256-token prefill shared-cache configuration                    |
| [`generators/`](generators/)                                       | Harness patch generators; useful for seeing the changes between optimization stages                                                             |
| [`checks/`](checks/)                                               | Numerical equivalence and debugging checks such as `d17_check.py`, `tau_diag.py`, and `verify_stateful.py`                                      |
| [`run/`](run/)                                                     | Execution and waiting scripts                                                                                                                   |
| [`probes/`](probes/)                                               | One-off exploratory experiments; kept mainly as a development record                                                                            |
| [`results/`](results/)                                             | Raw benchmark JSON files                                                                                                                        |
| [`docs/OPTIMIZATIONS.md`](docs/OPTIMIZATIONS.md)                   | Detailed description of the optimization steps                                                                                                  |
| [`docs/VENDOR_LIMITS.md`](docs/VENDOR_LIMITS.md)                   | Runtime/compiler limitations found during the port                                                                                              |
| [`docs/RESULTS.md`](docs/RESULTS.md)                               | Full benchmark results and GPU comparison                                                                                                       |
| [`docs/ROPE_ROOT_CAUSE.md`](docs/ROPE_ROOT_CAUSE.md)               | Root cause and fix for the long-context RoPE numerical failure                                                                                  |
| [`docs/LONG_CONTEXT_CMR.md`](docs/LONG_CONTEXT_CMR.md)             | Long-context tau degradation and CMR comparison between NPU and GPU                                                                             |
| [`docs/HOST_THREAD_STARVATION.md`](docs/HOST_THREAD_STARVATION.md) | Analysis showing that much of the original CMR overhead came from the host implementation                                                       |
| [`docs/CMR_TUNING.md`](docs/CMR_TUNING.md)                         | CMR parameter analysis and why dynamic search was ineffective for DFlash                                                                        |

---

## 13. Reproducing the setup

Compiled `.rbln` graphs and model weights are not included in this repository because they are several gigabytes in size.

They need to be rebuilt using the scripts under `compile/`.

```bash
# 1. Build the stateful drafter graphs
python3 compile/build_stateful.py

# 2. Build target verify-17 and prefill-256 graphs with a shared KV cache
CHUNKS=17,256 python3 compile/dual_1_256.py

# 3. Run the benchmark
DSET=gsm8k NSAMP=20 MAXNEW=2048 DEV=0 python3 bench/bench_sf_dual.py
```

Several filesystem paths are currently hard-coded near the top of the scripts, including:

```text
SRC
TGT
D
```

These need to be changed for a different server setup.

Server access information is kept in internal lab documentation and is intentionally not included in this repository.

---

## 14. Takeaways

Porting DFlash to a compiled accelerator exposed a different set of bottlenecks from the ones I initially expected.

The speculative decoding algorithm itself was not the difficult part. Most of the work came from fitting a dynamic decoding loop into a static execution model and separating actual hardware limits from runtime and wrapper restrictions.

The biggest improvements came from:

* keeping KV state on the device,
* allowing execution to resume at arbitrary cache positions,
* using different graph shapes for prefill and verification while sharing the same cache,
* avoiding unnecessary host outputs,
* removing host-side scheduling and memory overhead,
* and validating numerical correctness instead of assuming that a successfully executed graph was correct.

After these changes, the target verification path went from **63.8 ms to 38.8 ms**, and the full `gsm8k` decoding round went from **96.5 ms to 59.5 ms**.

At that point, the remaining latency was largely explained by the accelerator's measured weight-streaming bandwidth rather than obvious software overhead.

The project also uncovered several cases where vendor-side checks were stricter than the underlying kernels, as well as numerical failures that produced no runtime error. In practice, understanding those software boundaries was just as important as optimizing the model itself.
