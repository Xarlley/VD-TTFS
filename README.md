# VD-TTFS

**Efficient TTFS Computing Architecture on Vector Devices** — official code release.

> **Supplementary material:** [`VD-TTFS_supplementary.md`](VD-TTFS_supplementary.md)
> (readable markdown) · [`VD-TTFS_supplementary.tex`](VD-TTFS_supplementary.tex)
> (LNCS source). Three appendices:
> **(A)** how the integrator chunk width Δ is selected, with per-task latency sweeps;
> **(B)** a fused multi-step LIF lower bound — SpikingJelly's Triton backend is ~10×
> slower than VD-TTFS on Task 1 and infeasible (OOM / non-terminating) on Task 2–3;
> **(C)** justification of the three-task evaluation suite.

VD-TTFS recasts the inference of a Time-to-First-Spike (TTFS)-encoded Spiking
Neural Network from an iterative discrete-time simulation into a **single
arrival-ordered sparse forward pass** on commodity vector devices (GPUs). It
treats the "fire-once" constraint as a structural property rather than a dynamic
behaviour to be simulated, and is realised by three components:

1. **Look-up-table (LUT) linearization** of the neuronal dynamics — the
   exponential weight-decay kernel `α(t)` and the decaying threshold `Θ(t)` are
   tabulated per layer, so each synaptic event becomes one table read + one FMA.
2. **Input-aware temporal truncation** — a per-layer horizon `T'_ℓ` calibrated
   offline from layer-wise spike-time percentiles; the one *optional* component
   that trades a bounded accuracy budget for memory.
3. **Time-sorted, chunked early-exit integrator** — synaptic events are
   bucketized by arrival time (a lossless counting sort) and integrated chunk by
   chunk; the instant a neuron crosses threshold it records its first-spike time
   and early-exits, so all later-arriving synapses are never fetched. The
   early-exit decision propagates back into input generation itself.

## Method coverage in this release

The released CUDA programs implement the **lossless** VD-TTFS configuration and
faithfully reproduce each baseline's accuracy. A few points make the mapping
between the paper's three components and the code explicit:

- **(iii) Time-sorted, chunked early-exit integrator** — implemented in every
  task (`k_bucketize*` counting-sort into arrival chunks, then a chunked
  integrator that halts at the first spike). The chunk width `Δ` matches the
  paper per task (16 / 8 / 32 for Tasks 1 / 2 / 3).
- **(i) LUT linearization** — implemented for Tasks 1 and 2 (`update_LUTs` builds
  the per-layer decay and threshold tables `LUT_decay` / `LUT_th`). **Task 3
  applies only the time-sorted early-exit integrator:** its CuLIF neuron uses a
  constant per-step current decay and a constant threshold, so there is no
  per-timestep transcendental term to tabulate and the LUT is not applicable.
- **(ii) Input-aware temporal truncation** is an **offline configuration**, not a
  runtime stage. The integration window is the compile-time constant
  `TIME_WINDOW`, set to the full simulation window `T` in this lossless release.
  Truncation is applied by reducing this window — guided offline by the
  layer-wise spike-time percentiles described in the paper — so that
  late-arriving, low-salience spikes are discarded for a bounded accuracy budget.
  Because the default build uses the full window, the headline accuracy and
  speedup numbers below are the lossless operating point and are unaffected by
  this setting.

## Results (paper)

| Task | Network / Dataset | Speedup vs Step-by-Step CUDA | Ops saved vs ANN |
|------|-------------------|------------------------------|------------------|
| 1 | LeNet / MNIST | up to **15.8×** | **56%** |
| 2 | VGG-16 / CIFAR-10 | up to **36.8×** | **73%** |
| 3 | 7-layer SNN / DVSGesture | up to **15.40×** | — |

Direct power profiling of Task 2 on a Jetson AGX Xavier shows a **95.4%**
reduction in measured total energy. Accuracy loss stays below **1.3%**.

## Repository layout

Each task provides up to four implementations of the *same* inference workload,
so that throughput and energy can be compared on equal terms:

```
task1_lenet_mnist/
  vdttfs_cuda/              VD-TTFS (time-sorted chunked early-exit)   ← our method
  stepbystep_cuda/          naïve time-driven step-by-step baseline
  torch_ann/                dense PyTorch/cuDNN ANN reference (LeNet)
  spikingjelly/             structurally identical TTFS-SNN in SpikingJelly
  eventdriven_baseline_cuda/  corrected event-driven CUDA baseline (snn_eventdriven_baseline.cu)
  macs/                     operation-count (MAC) profiler
  exported_models/          (place snn_weights.bin here — not committed)
task2_vgg16_cifar10/
  vdttfs_cuda/              VD-TTFS                                    ← our method
  stepbystep_cuda/          naïve time-driven step-by-step baseline (+2 GB variant)
  stepbystep_lut_cuda/      step-by-step + LUT (ablation, §5.1)
  torch_ann/                dense PyTorch/cuDNN ANN reference
  spikingjelly/             structurally identical TTFS-SNN in SpikingJelly
  macs/                     MAC profiler + single-image correctness check
  exported_models/          (place snn_weights_vgg.bin here — 59 MB, see below)
task3_dvsgesture/
  vdttfs_cuda/              VD-TTFS                                    ← our method
  stepbystep_cuda/          step-by-step time-driven baseline (snn_dvs_stepbystep.cu)
  spikingjelly/             SpikingJelly reference (spkjelly/)
  macs/                     MAC profiler
  cuda_assets/              (place dvsgesture_weights.bin + train_labels.bin here — not committed)
```

> **Note (Task 3 has no ANN baseline)** — DVSGesture is an event stream with no
> natural dense-ANN equivalent, so only Step-by-Step / SpikingJelly / VD-TTFS are
> provided, as in the paper.

## Build & run (CUDA)

Requirements: CUDA Toolkit (tested with **12.4**, `nvcc`), an NVIDIA GPU.
All CUDA programs read their weights/data via **relative paths**, so compile
anywhere but **run from the task directory** that contains the asset folders.

```bash
# Task 2 — VD-TTFS on VGG-16/CIFAR-10
cd task2_vgg16_cifar10
nvcc vdttfs_cuda/bench_vgg_eventdriven_timesorted_earlyexit.cu -o vdttfs -O3 -Wno-deprecated-gpu-targets
./vdttfs                       # reads exported_models/snn_weights_vgg.bin + dataset_downloaded/cifar10_float/

# the step-by-step baseline it is compared against
nvcc stepbystep_cuda/bench_vgg_timedriven_baseline.cu -o stepbystep -O3 -Wno-deprecated-gpu-targets && ./stepbystep

# Task 1 — VD-TTFS on LeNet/MNIST
cd task1_lenet_mnist
nvcc vdttfs_cuda/snn_timesorted_earlyexit.cu -o vdttfs -O3 -Wno-deprecated-gpu-targets
./vdttfs                       # reads exported_models/snn_weights.bin + dataset_downloaded/mnist_test10k/

# Task 3 — VD-TTFS on DVSGesture
cd task3_dvsgesture
nvcc vdttfs_cuda/snn_inference_timesorted.cu -o vdttfs -O3 -Wno-deprecated-gpu-targets
./vdttfs                       # reads cuda_assets/{dvsgesture_weights,train_data,train_labels}.bin
```

`-Wno-deprecated-gpu-targets` silences the `sm_120` (Blackwell, e.g. RTX 5070 Ti)
deprecation warning; it does not affect results.

## Build & run (PyTorch / SpikingJelly baselines)

Python ≥ 3.10 with `torch`, `cupy-cuda12x`, `triton`, `spikingjelly`,
`numpy`. The Task 2 baselines read the same `snn_weights_vgg.bin`:

```bash
# Task 2 (VGG-16 / CIFAR-10)
cd task2_vgg16_cifar10
python torch_ann/baseline_torch_ann.py             [num_images] [batch] [fp16]
python spikingjelly/baseline_spikingjelly.py       [num_images] [batch_size]

# Task 1 (LeNet / MNIST)
cd task1_lenet_mnist
python torch_ann/baseline_torch_ann_mnist.py       [num_images] [batch] [fp16]
python spikingjelly/baseline_spikingjelly_mnist.py [num_images] [batch_size]
```

> The SpikingJelly TTFS neuron is a **custom single-step** node: SpikingJelly's
> fused CuPy/Triton multi-step backends are hand-written for its *built-in*
> neuron models only and cannot express the bespoke TTFS threshold/weight-decay
> dynamics, so this baseline necessarily runs on the single-step PyTorch path.

## Datasets & weights

**No weights or datasets are committed to this repository.** Obtain or regenerate
them and place them at the relative paths below, in the formats documented in this
section (and in each task's `weights_io.py`).

| Task | Weights | Test set (relative path expected at run time) |
|------|---------|------------------------------------------------|
| 1 | `exported_models/snn_weights.bin` | `dataset_downloaded/mnist_test10k/{0..9999}.bin` + `label_onehot` |
| 2 | `exported_models/snn_weights_vgg.bin` | `dataset_downloaded/cifar10_float/{0..9999}.bin` + `label_onehot` |
| 3 | `cuda_assets/dvsgesture_weights.bin` | `cuda_assets/train_data.bin` + `cuda_assets/train_labels.bin` |

**Binary layouts** (so the inputs can be regenerated and placed at the paths above):

- **MNIST** — `dataset_downloaded/mnist_test10k/{i}.bin`: 784 `float32`, row-major
  28×28, single channel; `label_onehot`: `N`×10 `float32`.
- **CIFAR-10** — `dataset_downloaded/cifar10_float/{i}.bin`: 3072 `float32`, HWC
  order `(h*32+w)*3+c`; `label_onehot`: `N`×10 `float32`.
- **DVSGesture** — `cuda_assets/train_data.bin`: `N`×2×32×32×160 `float32` event
  tensor; `cuda_assets/train_labels.bin`: `N`×10 `int32` one-hot. The CUDA programs
  evaluate the `N=1078` export drawn from the DVSGesture training split (this is the
  evaluation set used throughout; see `BENCHMARKS_5070Ti.md`).
- **Weights** — the binary format is documented in each task's `weights_io.py`
  (`int32 num_layers`, then per layer the kernel shape and data, bias, and the two
  TTFS time constants). All weights must be supplied by the user; none are committed.

The flat-binary inputs and weights are produced by the export step of the upstream
training pipelines (T2FSNN for Tasks 1–2, FS\_Coding for Task 3); those pipelines
are not part of this inference release.

## Status

All three tasks carry their full implementation set. The Task 1 **Step-by-Step
CUDA**, **Torch ANN**, and **SpikingJelly** baselines were originally authored on
a rented V100 server that has since been decommissioned; they were **reconstructed
from the local network/dynamics specification** (the VD-TTFS source and the Task 2
baselines) and validated to reproduce the paper's behaviour on the full 10,000-image
MNIST test set: Step-by-Step CUDA **98.18%** (identical to VD-TTFS), Torch ANN
**99.3%**, SpikingJelly **98.67%**.

## Citation

```bibtex
@inproceedings{vdttfs,
  title     = {VD-TTFS: Efficient TTFS Computing Architecture on Vector Devices},
  author    = {Liu, Shifeng and Yang, Zhijie and Wang, Dongsheng and Wang, Lei},
  year      = {2026}
}
```
