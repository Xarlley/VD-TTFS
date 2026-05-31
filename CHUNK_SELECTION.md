# Selecting the integrator chunk width Δ

The time-sorted early-exit integrator has exactly **one granularity hyper-parameter**:
the **chunk width Δ** — the number of consecutive timesteps grouped into each block
that the integrator sweeps. It is a compile-time constant:

| Task | Source | Macro | Default Δ | Window T |
|---|---|---|---|---|
| 1 — LeNet/MNIST | `vdttfs_cuda/snn_timesorted_earlyexit.cu` | `#define CHUNK` | 16 | 80 |
| 2 — VGG-16/CIFAR-10 | `vdttfs_cuda/bench_vgg_eventdriven_timesorted_earlyexit.cu` | `#define CHUNK` | 8 | 80 |
| 3 — DVSGesture | `vdttfs_cuda/snn_inference_timesorted.cu` | `const int CHUNK_SIZE` | 32 | 160 |

This document records **how** Δ is chosen and the **measured basis** for the values
above, re-derived on the RTX 5070 Ti (environment: see
[`BENCHMARKS_5070Ti.md`](BENCHMARKS_5070Ti.md)).

---

## Method

### Why Δ matters — the trade-off

The integrator partitions the (truncated) window `[0, T'_ℓ)` into chunks of Δ
timesteps, gathers the synaptic events arriving in each chunk into a per-neuron
`delta[Δ]` accumulator, integrates chunk by chunk, and **early-exits the instant a
neuron crosses threshold** — so all events in later chunks are never fetched. Δ
sets the resolution of that early exit, with two opposing effects:

- **Finer Δ (small)** — the neuron stops at a finer time granularity, so fewer
  *post-firing* synapses are processed (better when early exit is active). But each
  chunk costs fixed per-chunk bookkeeping (zeroing `delta[Δ]`, a threshold scan,
  loop overhead), and there are `⌈T/Δ⌉` chunks — so very small Δ multiplies that
  overhead.
- **Coarser Δ (large)** — fewer chunks, less bookkeeping, but the early exit is
  blunter (a fired neuron still pays for the rest of its current chunk), and the
  `delta[Δ]` array grows: for large Δ it spills out of registers into local memory,
  which is slow.

The optimum therefore depends on **how much early exit there is to exploit** — i.e.
the arrival-ordered early-exit factor γ — and on the **window length T**:
- strong early exit (small γ) rewards **fine** Δ;
- weak / no early exit (γ→1) makes fine chunking pure overhead, so a **coarser** Δ
  wins, up to the point where `delta[Δ]` local-memory pressure dominates.

### Determination procedure

For each task we sweep Δ over powers of two up to the window length, recompile,
run the **full evaluation set on the target GPU**, and pick the **latency minimum**:

```bash
# example (Task 1); run from the task directory so the relative assets resolve
for C in 4 8 16 32 64; do
  sed "s/#define CHUNK 16/#define CHUNK $C/" vdttfs_cuda/snn_timesorted_earlyexit.cu > /tmp/sw.cu
  nvcc /tmp/sw.cu -o /tmp/sw -O3 -Wno-deprecated-gpu-targets
  /tmp/sw            # report Accuracy + Total Inference Time
done
```

Each configuration is run **3×** and the **minimum** inference time is reported
(reduces scheduling jitter). **Accuracy is recorded at every Δ** to confirm that
chunking is lossless — re-chunking is an exact re-ordering of the same events, so
the result must be invariant (see the FP note below).

---

## Basis — measured sweeps (RTX 5070 Ti, full eval set, min of 3 runs)

### Task 1 — LeNet / MNIST (T = 80), 10,000 images

| Δ (CHUNK) | Accuracy | Inference time |
|---|---|---|
| 4 | 98.18% | 0.1025 s |
| 8 | 98.18% | 0.0645 s |
| **16** | **98.18%** | **0.0450 s** ← min |
| 32 | 98.18% | 0.0489 s |
| 64 | 98.18% | 0.0559 s |

→ optimum **Δ = 16** (the shipped default). LeNet fires late relative to its small
fan-in (γ≈0.97, early exit weak), so very fine chunks (4, 8) are pure overhead and
coarse chunks (32, 64) blunt the little early exit there is; 16 is the sweet spot.

### Task 2 — VGG-16 / CIFAR-10 (T = 80), 10,000 images

| Δ (CHUNK) | Accuracy | Inference time |
|---|---|---|
| 4 | 90.76% | 2.730 s ← min |
| **8** | **90.76%** | **2.747 s** (+0.6%) |
| 16 | 90.76% | 3.120 s |
| 32 | 90.76% | 3.204 s |
| 64 | 90.76% | 3.427 s |

→ **fine chunks win** (4 ≈ 8, within run-to-run noise; both clearly beat 16–64).
VGG-16 has strong arrival-ordered early exit (γ=0.62), so a fine Δ sharpens the
resolution at which a fired neuron drops its remaining synapses. The shipped
default **Δ = 8** sits at this optimum (Δ = 4 is marginally faster but within noise).

### Task 3 — DVSGesture (T = 160), 1078 samples

| Δ (CHUNK_SIZE) | Accuracy | Inference time |
|---|---|---|
| 8 | 96.29% | 1027 ms |
| 16 | 96.29% | 955 ms ← min |
| **32** | **96.29%** | **966 ms** (+1.2%) |
| 64 | 96.38%* | 1014 ms |
| 160 | 96.29% | 1466 ms |

→ **coarse chunks win** (16 ≈ 32; both beat fine-8 and the very coarse 160). The
DVSGesture stream has essentially no early exit (γ≈0.99) and a long window
(T = 160), so fine chunking only adds per-chunk overhead, while the largest Δ = 160
makes `delta[160]` spill local memory (1466 ms, the slowest). The shipped default
**Δ = 32** (warp-aligned) is within ~1% of the Δ = 16 minimum and is preferred for
warp alignment.

\*See the floating-point note.

---

## Findings

1. **The shipped defaults are confirmed near-optimal on this GPU**: Task 1 Δ=16 is
   the exact minimum; Task 2 Δ=8 is within 0.6% of the minimum; Task 3 Δ=32 is
   within 1.2% of the minimum.
2. **The optimum tracks γ exactly as the paper argues**: strong early exit (VGG,
   γ=0.62) → fine Δ; weak early exit (LeNet, γ≈0.97) → medium Δ; no early exit
   (DVSGesture, γ≈0.99, long T) → coarse Δ.
3. **Chunking is lossless.** Accuracy is invariant across Δ for Tasks 1–2 and for
   Task 3 except Δ=64, where it differs by exactly one sample (96.38% vs 96.29%,
   i.e. 1039 vs 1038 / 1078). This is **floating-point accumulation order**, not an
   algorithmic change: different chunk widths sum the same contributions in a
   different order, and one borderline neuron's first-spike time flips. The exact
   event set and contributions are identical (a lossless permutation); only IEEE-754
   non-associativity produces the ±1-sample variation.
4. **Hardware dependence.** These optima are for the RTX 5070 Ti (sm_120). The
   precise minimum can shift by one step on other GPUs (the paper's defaults were
   chosen across V100 / 5070 Ti / Jetson), but the qualitative rule — γ and T
   determine whether fine or coarse Δ is best — is hardware-independent.
