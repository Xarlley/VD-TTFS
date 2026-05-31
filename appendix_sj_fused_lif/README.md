# SpikingJelly fused multi-step LIF — "framework lower bound" benchmark

## Purpose

SpikingJelly ships **no** efficient TTFS kernel, so the paper's SpikingJelly
numbers use the `torch` backend, which invites the objection: *"a fused / Triton
backend would be far faster, so VD-TTFS's advantage is overstated."*

This experiment answers that objection. A fused multi-step LIF network — run with
SpikingJelly's accelerated **cupy** and **triton** backends — still traverses
**every one of the T timesteps** with **no early exit** and **no event-sparsity**
exploitation. Its latency is therefore a *lower bound* on what any fused TTFS
backend (including a hypothetical TTFS-Triton kernel) could achieve on the same
network. If VD-TTFS is still faster than this lower bound, the speedup provably
comes from the algorithm (temporal collapse + early exit + sparsity), not from
SpikingJelly merely being a slow framework.

We build SNNs with the **same layer structure** and **same number of timesteps T**
as VD-TTFS Task 1/2/3, but with plain **LIF** neurons and **random weights/inputs**
(accuracy is irrelevant — only latency/memory matter), and time the multi-step
forward under each available neuron backend (`torch`, `cupy`, `triton`).

## Network structures (match the three tasks)

| Task | Structure | T |
|---|---|---|
| 1 | Conv 1→12 (5×5) · pool · Conv 12→64 (5×5) · pool · FC 1024→10 (LeNet/MNIST) | 80 |
| 2 | VGG-16: 13× Conv 3×3 + 5 maxpool · FC 512→512→512→10 (CIFAR-10) | 680 |
| 3 | Conv 2→64 · Conv 64→128 · pool · Conv 128→128 · pool · FC 8192→128 · FC 128→10 (DVSGesture) | 160 |

LIF neurons replace the TTFS/CuLIF dynamics; the FC1 recurrence of Task 3 is
omitted (timing only). All convs are step-mode `'m'` (multi-step).

## Environment

A dedicated conda env is required because the **Triton neuron backend only exists
in SpikingJelly master (0.0.0.0.15+)**; the released 0.0.0.0.14 exposes only
`torch` and `cupy`.

```bash
conda create -y -n sj_triton python=3.11
conda run -n sj_triton pip install torch --index-url https://download.pytorch.org/whl/cu128
conda run -n sj_triton pip install numpy cupy-cuda12x pytest
conda run -n sj_triton pip install git+https://github.com/fangwei123456/spikingjelly.git
# verify: multi-step LIF supported_backends should include 'triton'
conda run -n sj_triton python -c "from spikingjelly.activation_based import neuron; n=neuron.LIFNode(); n.step_mode='m'; print(n.supported_backends)"
```

(On this machine the install resolved to torch 2.12.0+cu130 / triton 3.7.0 /
spikingjelly 0.0.0.0.15; `pytest` is needed or a `cupy.testing` import error
fires during introspection.)

## Usage

There is **one self-contained script per task** (each uses the `triton` backend
and measures peak GPU memory with `nvidia-smi`, allocator-agnostic, matching
`BENCHMARKS_5070Ti.md` — *not* `torch.cuda.max_memory_allocated`, which misses
the triton/cupy backends' own allocations):

```bash
ENV=/path/to/sj_triton/bin/python    # the conda env built above
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
$ENV -u task1_triton.py [batch] [iters]   # LeNet/MNIST,  T=80   (default batch 1000)
$ENV -u task2_triton.py [batch] [iters]   # VGG-16/CIFAR, T=680  (default batch 1)
$ENV -u task3_triton.py [batch] [iters]   # DVSGesture,   T=160  (default batch 8)
```

Each prints throughput (samples/s), per-sample latency, and peak GPU memory
(MiB, via `nvidia-smi`).

> **Why `nvidia-smi` and not `torch.cuda.max_memory_allocated()`** — the triton
> and cupy backends allocate outside PyTorch's caching allocator, so
> `max_memory_allocated` under-counts them (it also omits the CUDA context).
> `nvidia-smi` per-process memory counts everything and is what the main benchmark
> file uses, so the two are comparable.

## Observed behaviour on the RTX 5070 Ti (16 GB)

- **Task 1** completes under all three backends.
- The **`torch` backend** does not complete the multi-step convolutional forward of
  Task 2/3 within the time budget (the pure-PyTorch multi-step path is inefficient
  for large convolutional SNNs).
- The **`cupy` backend** exhibits a CUDA-13 / `cupy-cuda12x` version mismatch,
  resulting in CPU-bound kernel compilation that does not terminate.
- **Task 2 (T=680)** multi-step execution materializes the activations of all T
  timesteps and exhausts the 16 GB device, raising an out-of-memory condition; this
  confirms that a fused multi-step is not memory-free for deep networks at long T,
  and is the reason single-step execution is used for the SpikingJelly baselines.
- The **`triton` backend** incurs a large one-time kernel-autotune cost on the large
  per-timestep tensors of Task 2/3.

Measured results are recorded in the supplementary appendix
(`../../VD-TTFS_supplementary.tex`, Section B).
