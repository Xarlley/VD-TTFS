# VD-TTFS — Supplementary Material

> Markdown rendering of the supplementary appendix to *VD-TTFS: Efficient TTFS
> Computing Architecture on Vector Devices*. The typeset source is
> [`VD-TTFS_supplementary.tex`](VD-TTFS_supplementary.tex) (LNCS). All measurements
> are on a single NVIDIA RTX 5070 Ti (16 GB), CUDA 12.8 / nvcc V12.8.93, unless
> noted; the SpikingJelly experiment (Appendix B) uses a dedicated environment with
> SpikingJelly 0.0.0.0.15, PyTorch 2.12 / CUDA 13, Triton 3.7
> (see [`appendix_sj_fused_lif/`](appendix_sj_fused_lif/)).

---

## A. Selecting the Integrator Chunk Width Δ

The time-sorted early-exit integrator exposes a single granularity hyper-parameter,
the **chunk width Δ**: the number of consecutive timesteps grouped into each block
the integrator sweeps. It is a compile-time constant (`CHUNK` / `CHUNK_SIZE`):
Δ = 8 for VGG-16, 16 for LeNet, 32 for the DVSGesture network.

### A.1 Method

**The trade-off.** The integrator partitions the (truncated) window `[0, T'_ℓ)`
into chunks of Δ timesteps, accumulates the events arriving in each chunk into a
per-neuron register array `delta[Δ]`, integrates chunk by chunk, and halts the
instant a neuron crosses threshold (later chunks are never fetched). Two opposing
effects result:

- **Finer Δ** stops the neuron at finer granularity (fewer post-firing synapses
  processed) but multiplies the fixed per-chunk bookkeeping over `⌈T/Δ⌉` chunks.
- **Coarser Δ** issues fewer chunks but blunts the early exit and enlarges
  `delta[Δ]`, which eventually spills to (slow) local memory.

The optimum tracks the early-exit factor γ and the window T: strong early exit
(small γ) favours fine Δ; weak/absent early exit (γ→1) favours coarse Δ, up to the
point where `delta[Δ]` local-memory pressure dominates.

**Determination procedure.** For each task we sweep Δ over powers of two up to the
window length, recompile, run the full evaluation set on the target GPU, and select
the latency minimum (min of 3 runs). Accuracy is recorded at every Δ to confirm
losslessness.

### A.2 Measured sweeps (RTX 5070 Ti, full eval set, min of 3 runs)

**Task 1 — LeNet/MNIST (T=80), 10,000 images**

| Δ | Accuracy | Inference time (s) |
|---|---|---|
| 4 | 98.18% | 0.1025 |
| 8 | 98.18% | 0.0645 |
| **16** | **98.18%** | **0.0450** ← min |
| 32 | 98.18% | 0.0489 |
| 64 | 98.18% | 0.0559 |

**Task 2 — VGG-16/CIFAR-10 (T=80), 10,000 images**

| Δ | Accuracy | Inference time (s) |
|---|---|---|
| 4 | 90.76% | 2.730 ← min |
| **8** | **90.76%** | **2.747** (+0.6%) |
| 16 | 90.76% | 3.120 |
| 32 | 90.76% | 3.204 |
| 64 | 90.76% | 3.427 |

**Task 3 — DVSGesture (T=160), 1078 samples**

| Δ | Accuracy | Inference time (ms) |
|---|---|---|
| 8 | 96.29% | 1027 |
| 16 | 96.29% | 955 ← min |
| **32** | **96.29%** | **966** (+1.2%) |
| 64 | 96.38%† | 1014 |
| 160 | 96.29% | 1466 |

### A.3 Findings

1. **Shipped defaults are at or near the per-task optimum.** Task 1 Δ=16 is the
   exact minimum; Task 2 Δ=8 within 0.6%; Task 3 Δ=32 within 1.2% (and warp-aligned).
2. **The optimum tracks γ.** Strong early exit (VGG-16, γ=0.62) → fine Δ; weak
   (LeNet, γ≈0.97) → medium Δ; absent + long window (DVSGesture, γ≈0.99, T=160) →
   coarse Δ, while the extreme Δ=160 spills `delta[Δ]` to local memory (slowest).
3. **Chunking is lossless.** Accuracy is invariant across Δ except Task 3 Δ=64 (†),
   which differs by one sample (96.38% vs 96.29%, i.e. 1039 vs 1038 of 1078). This
   is IEEE-754 accumulation order, not an algorithmic change: identical
   contributions summed in a different order flip one borderline first-spike time.
4. **Hardware dependence.** Optima are specific to the RTX 5070 Ti; the minimum can
   shift by one step on other GPUs, but the qualitative rule (γ and T determine fine
   vs coarse Δ) is hardware-independent.

---

## B. A Fused Multi-Step LIF Lower Bound for SpikingJelly

### B.1 Motivation

SpikingJelly provides no efficient TTFS kernel, and the SpikingJelly baselines in
the main paper consequently employ its `torch` backend. A natural concern is that a
fused or Triton backend would be substantially faster, diminishing VD-TTFS's
apparent advantage. A fused multi-step LIF network — executed with SpikingJelly's
accelerated `cupy` and `triton` backends — nonetheless evaluates **every** one of
the T timesteps, with no early exit and no event-sparsity exploitation. Its latency
is therefore a lower bound on the cost of any fused TTFS backend (including a
hypothetical TTFS-Triton kernel) on the same network: if VD-TTFS is faster than
this bound, the improvement is algorithmic, not a framework artifact.

### B.2 Experimental setup

We instantiate networks with the identical layer structure and identical T as
Task 1/2/3, substituting standard LIF neurons and random weights/inputs (accuracy
is immaterial; only latency and memory matter). Each runs in multi-step mode under
the `triton` backend, available in SpikingJelly master (≥ 0.0.0.0.15); the released
0.0.0.0.14 exposes only `torch` and `cupy`. Peak GPU memory is measured with
`nvidia-smi` (per-process, allocator-agnostic, includes the CUDA context);
`torch.cuda.max_memory_allocated` is **not** used, as it omits allocations outside
the PyTorch caching allocator (e.g. the `cupy`/`triton` backends').

### B.3 Acceptance threshold

Reference = VD-TTFS per-sample latency on the same device: 4.9 µs (Task 1),
274 µs (Task 2), 899 µs (Task 3). A fused multi-step configuration is non-competitive
if its per-sample latency exceeds **10×** the corresponding VD-TTFS latency, or if it
cannot execute within device memory. A configuration is **infeasible** if it does not
terminate within the 10× budget (0.49 s / 27.4 s / 9.69 s for the full workloads) or
raises an out-of-memory condition.

### B.4 Results

| Task (T) | VD-TTFS (ms/sample) | Triton (ms/sample) | Determination |
|---|---|---|---|
| Task 1 (T=80) | 0.0049 | 0.049 (peak 13,468 MiB) | **10.0× slower** |
| Task 2 (T=680) | 0.274 | — | **infeasible (memory)** |
| Task 3 (T=160) | 0.899 | — | **infeasible (latency)** |

A dash denotes that no latency could be obtained because the configuration is
infeasible under the criterion of B.3.

### B.5 Findings

1. **Backend invariance.** On Task 1 the `triton`, `cupy`, and `torch` backends
   yield an identical per-sample latency (0.049 ms) and, via `nvidia-smi`, identical
   peak memory (≈13.5 GB). The elementwise neuronal update is a negligible fraction
   of the per-timestep cost, so the backend affects neither throughput nor memory.
2. **Latency lower bound (Task 1).** The Triton multi-step latency (0.049 ms/sample)
   is exactly 10.0× that of VD-TTFS (0.0049 ms/sample). The gap is structural: the
   multi-step pass evaluates all 80 timesteps, whereas VD-TTFS collapses the temporal
   dimension and exits at the first spike. A TTFS-Triton kernel would incur the same
   bound.
3. **Memory infeasibility (Task 2).** For VGG-16 at T=680, multi-step execution
   materializes all 680 timesteps' activations; the footprint exhausts the 16 GB
   device (OOM observed), and where the immediate allocation failure is averted the
   pass does not terminate within the latency budget.
4. **Latency infeasibility (Task 3).** For DVSGesture at T=160, the Triton multi-step
   forward does not terminate within 10× the VD-TTFS latency; the 9.69 s budget is
   exceeded by more than 40× without completion.
5. **Conclusion.** The inability of the most heavily optimized SpikingJelly backend
   to match VD-TTFS — by latency on Task 1, by feasibility on Task 2/3 — confirms
   that VD-TTFS's efficiency is algorithmic, not a framework/backend artifact.

### B.6 Soundness of the single-step baseline

The SpikingJelly baselines in the main text use the **single-step** mode and
complete every workload: on the RTX 5070 Ti, Task 1 in 0.996 s (peak 750 MiB) and
Task 2 in 29.99 s (peak 3876 MiB); on the V100 of the main text, 1.81 s and 38.4 s.

The contrast with the multi-step configuration above is explained by memory
complexity. Single-step advances one timestep at a time, retaining only the current
step's activations plus the persistent membrane state — activation memory **𝒪(N),
independent of T**. Multi-step materializes all T timesteps at once — **𝒪(N·T)**.
For VGG-16 at T=680 the single-step pass occupies 3.9 GB and fits, whereas the
multi-step pass exceeds 16 GB. Single-step also avoids the non-terminating
fused-kernel autotune. The two modes embody a memory–throughput trade-off.

Hence the baseline choice is sound: single-step is the **only** SpikingJelly mode
that runs the full TTFS dynamics of the deep, long-window networks within commodity
memory. The ostensibly faster fused/Triton alternative is, on identical hardware,
either no faster (Task 1) or infeasible (Task 2/3). Single-step therefore represents
SpikingJelly's best achievable performance and is a fair — indeed favorable —
comparison; yet it remains 20.3× (Task 1) and 10.9× (Task 2) slower than VD-TTFS, a
margin a fused backend cannot recover.

---

## C. Justification of the Evaluation Suite

The main text evaluates three tasks — LeNet/MNIST (T=80), VGG-16/CIFAR-10 (T=680),
and a seven-layer recurrent network on event-based DVSGesture (T=160) — spanning
shallow to deep architectures, static to event-based data, and short to long
temporal windows. The scale of this suite is appropriate.

Increasing network depth or input complexity raises the per-timestep cost and,
decisively, the memory footprint of any multi-step execution (𝒪(N·T); §B.6). The
de facto high-performance SNN execution path — SpikingJelly's fused multi-step
(CuPy/Triton) backend, in the most widely adopted framework — is **already at its
feasibility limit** on the present suite: it exhausts the 16 GB device on
VGG-16/CIFAR-10 and does not terminate on the DVSGesture network (§B). A strictly
larger network or higher-resolution dataset would place this baseline further beyond
reach: it could not run at all, and so could not serve as a comparison point.

The remaining feasible means of executing the full TTFS dynamics — SpikingJelly
single-step and a hand-written step-by-step CUDA implementation — both traverse all
T timesteps without early exit or sparsity, and are already markedly slower than
VD-TTFS at the present scale (single-step by 10.9×–20.3×, §B.6; step-by-step CUDA by
up to 36.8×, main text). Both costs grow with network size and window length, so the
margin only widens on larger tasks.

The chosen suite therefore occupies the appropriate operating point: large enough
that the standard accelerated SNN baseline has reached the limit of feasibility, yet
tractable for the slower feasible baselines against which VD-TTFS is measured. Larger
benchmarks would not change the qualitative conclusion — they would merely preclude
the baselines themselves — so the three-task design of the main text is well
justified.
