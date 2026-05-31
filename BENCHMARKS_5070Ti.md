# VD-TTFS Local Benchmarks — RTX 5070 Ti

All 11 experiments (Step-by-Step CUDA · Torch ANN · SpikingJelly · VD-TTFS CUDA
for each task; Task 3 has no ANN) were run locally on a single workstation. This
document records the **runtime environment**, **Round 1** (accuracy + latency),
and **Round 2** (GPU memory: average + peak).

> ⚠️ This machine is an **RTX 5070 Ti**, *not* the NVIDIA Tesla V100 used in the
> paper, so absolute latencies differ from the paper's tables. Relative trends
> (VD-TTFS vs. step-by-step) hold; see the notes at the end for important caveats
> (especially the Task 3 train/test-split and model differences).

---

## Runtime environment

| Component | Detail |
|---|---|
| OS | Ubuntu 24.04.2 LTS (kernel 6.14.0-34-generic) |
| CPU | 12th Gen Intel Core i7-12700K (12 cores / 20 threads, 1 socket) |
| System RAM | 31 GiB |
| GPU | NVIDIA GeForce RTX 5070 Ti, 16303 MiB (≈16 GB), driver 595.71.05 |
| CUDA toolkit | 12.8 (`nvcc` V12.8.93) |
| Host compiler | gcc 13.3.0 |
| CUDA build flags | `nvcc -O3 -Wno-deprecated-gpu-targets` |

**Software stacks used by the experiments**

| Stack | Used by | Versions |
|---|---|---|
| CUDA toolchain | all `*_cuda` (Step-by-Step, VD-TTFS) | CUDA 12.8 / nvcc V12.8.93 / driver 595.71.05 / gcc 13.3.0 |
| Base Python | Torch ANN baselines | Python 3.13.9, torch 2.9.1+cu128, numpy 2.3.5 |
| `spiking_env` | SpikingJelly baselines | Python 3.10.19, torch 2.9.1+cu128, **spikingjelly 0.0.0.0.14**, cupy 13.6.0, triton 3.5.1 |

The Task 3 SpikingJelly run additionally uses the source **FS\_Coding** framework
(`CuLIFNode`, h5py, matplotlib) on the same `spiking_env`.

---

## Round 1 — accuracy & inference latency

Latency is each program's self-reported **inference time** (excludes weight/data
loading). Speedup is relative to that task's Step-by-Step CUDA baseline.

### Task 1 — LeNet / MNIST (full 10,000-image test set)

| Method | Accuracy | Inference time | Speedup vs Step-by-Step |
|---|---|---|---|
| Step-by-Step (CUDA) | 98.18% | 0.0497 s | 1.0× |
| SpikingJelly | 98.23% | 0.996 s | 0.05× |
| Torch ANN | 99.43% | 0.0058 s | — (dense ANN ref) |
| **VD-TTFS (CUDA)** | **98.18%** | **0.0493 s** | **1.01×** |

> On this fast GPU, LeNet/MNIST is too small for the temporal-collapse advantage
> to register: both CUDA variants finish in ~0.05 s, dominated by fixed overhead.
> (On the paper's V100 the step-by-step is 1.39 s vs VD-TTFS 0.088 s = 15.8×.)

### Task 2 — VGG-16 / CIFAR-10 (full 10,000-image test set)

| Method | Accuracy | Inference time | Speedup vs Step-by-Step |
|---|---|---|---|
| Step-by-Step (CUDA) | 90.69% | 148.69 s | 1.0× |
| SpikingJelly | 90.70% | 29.99 s | 5.0× |
| Torch ANN | 91.32% | 0.230 s | — (dense ANN ref) |
| **VD-TTFS (CUDA)** | **90.76%** | **2.74 s** | **54.3×** |

### Task 3 — 7-layer SNN / DVSGesture

| Method | Accuracy | Samples / split | Inference time | Speedup |
|---|---|---|---|---|
| Step-by-Step (CUDA) | 96.29% | 1078 (train split) | 14.79 s | 1.0× |
| SpikingJelly (FS\_Coding) | 84.09% FS / 76.89% FR | 264 (test split) | 7.92 s | — (see note) |
| **VD-TTFS (CUDA)** | **96.29%** | 1078 (train split) | **0.969 s** | **15.3×** |

> The two CUDA programs evaluate the **1078-sample training export**
> (`train_data.bin`, from `DVS-Gesture-train10.hdf5`); the SpikingJelly entry runs
> the **original FS\_Coding model** (FS / rate readout) on the **264-sample test
> set**. Different split *and* different model — its accuracy is **not** directly
> comparable; it is included as the SpikingJelly framework/timing reference. See
> the final notes.

---

## Round 2 — GPU memory (average & peak)

Per-process GPU memory sampled via `nvidia-smi --query-compute-apps` at 50 ms
intervals over the run (matched to the process tree). **Peak** is the maximum
sample; **Avg** is the mean over samples in which the process held GPU memory.
Very short runs (Task 1) yield few samples, so their averages are coarse.

| Task | Method | Avg GPU mem | Peak GPU mem |
|---|---|---|---|
| 1 | Step-by-Step (CUDA) | 616 MiB | 812 MiB |
| 1 | SpikingJelly | 657 MiB | 750 MiB |
| 1 | Torch ANN | 1004 MiB | 3728 MiB |
| 1 | **VD-TTFS (CUDA)** | **758 MiB** | **1124 MiB** |
| 2 | Step-by-Step (CUDA) | 8660 MiB | 8668 MiB |
| 2 | SpikingJelly | 3840 MiB | 3876 MiB |
| 2 | Torch ANN | 1179 MiB | 6020 MiB |
| 2 | **VD-TTFS (CUDA)** | **1965 MiB** | **2050 MiB** |
| 3 | Step-by-Step (CUDA) | 6148 MiB | 6336 MiB |
| 3 | SpikingJelly (FS\_Coding) | 9871 MiB | 11146 MiB |
| 3 | **VD-TTFS (CUDA)** | **2418 MiB** | **3260 MiB** |

**Highlights.** On Task 2 the step-by-step baseline holds **8.7 GB** (it
materializes the full per-timestep membrane state for a batch of 5000), whereas
VD-TTFS needs **~2.0 GB** — a 4.2× smaller footprint. On Task 3, VD-TTFS uses
**3.3 GB peak** versus the step-by-step's 6.3 GB.

---

## Notes & caveats

1. **Hardware** — RTX 5070 Ti (16 GB, sm_120), not the paper's V100. Absolute
   latencies are therefore not comparable to the paper; the 5070 Ti is much faster
   in wall-clock, which is why Task 1's tiny LeNet shows no VD-TTFS speedup here.
2. **Batch / workload sizes** — Task 1: full 10k single batch (CUDA), ANN batch
   10000, SJ batch 2000. Task 2: VD-TTFS internal batching over 10k, Step-by-Step
   batch 5000, ANN/SJ batch 1000. Task 3 CUDA: 1078 samples; Task 3 SJ: 264.
3. **Task 3 train/test split** — the Task 3 CUDA programs (and the paper's Task 3
   numbers) evaluate the **1078-sample training export**, so 96.29% is a
   training-set figure; it is appropriate as a fixed *speed* workload but should
   not be read as test accuracy.
4. **Task 3 SpikingJelly model** — the only available SpikingJelly model is the
   original FS\_Coding network (CuLIF, FS/rate readout), evaluated on the 264-sample
   test set (84.09% FS / 76.89% FR). It is *not* a TTFS-converted SpikingJelly
   equivalent of the CUDA program, so its accuracy is not an apples-to-apples
   comparison with VD-TTFS.
5. **Memory methodology** — values are GPU memory held by the experiment's own
   process(es); for PyTorch/SpikingJelly this includes the CUDA context and the
   caching allocator's reserved pool (hence the higher peaks for short ANN runs).
