# VD-TTFS

**Efficient TTFS Computing Architecture on Vector Devices** — official code release.

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

## Results (paper)

| Task | Network / Dataset | Speedup vs Step-by-Step CUDA | Ops saved vs ANN |
|------|-------------------|------------------------------|------------------|
| 1 | LeNet / MNIST | up to **15.8×** | **56%** |
| 2 | VGG-16 / CIFAR-10 | up to **36.8×** | **73%** |
| 3 | 7-layer SNN / DVSGesture | up to **15.40×** | — |

Direct power profiling of Task 2 on a Jetson AGX Xavier shows a **95.4%**
reduction in measured total energy. Accuracy loss stays below **1.3%**.

## Repository layout

Each task provides up to four implementations of the *same* inference workload
so that throughput/energy can be compared apples-to-apples:

```
task1_lenet_mnist/
  vdttfs_cuda/              VD-TTFS (time-sorted chunked early-exit)   ← our method
  eventdriven_baseline_cuda/  corrected event-driven CUDA baseline (snn_new.cu)
  macs/                     operation-count (MAC) profiler
  exported_models/          snn_weights.bin (LeNet weights, 117 KB)
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
  stepbystep_cuda/          step-by-step TTFS baseline (snn_inference_TTFS.cu)
  spikingjelly/             SpikingJelly reference (spkjelly/)
  macs/                     MAC profiler
  cuda_assets/              dvsgesture_weights.bin + a single sample + labels
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
cd task2_vgg16_cifar10
python torch_ann/baseline_torch_ann.py        [num_images] [batch] [fp16]
python spikingjelly/baseline_spikingjelly.py  [num_images] [batch_size]
```

> The SpikingJelly TTFS neuron is a **custom single-step** node: SpikingJelly's
> fused CuPy/Triton multi-step backends are hand-written for its *built-in*
> neuron models only and cannot express the bespoke TTFS threshold/weight-decay
> dynamics, so this baseline necessarily runs on the single-step PyTorch path.

## Datasets & weights

| Task | Weights | Test set (relative path expected at run time) |
|------|---------|------------------------------------------------|
| 1 | `exported_models/snn_weights.bin` (included) | `dataset_downloaded/mnist_test10k/{0..9999}.bin` |
| 2 | `exported_models/snn_weights_vgg.bin` (**59 MB — not committed**) | `dataset_downloaded/cifar10_float/{0..9999}.bin` + `label_onehot` |
| 3 | `cuda_assets/dvsgesture_weights.bin` (included) | `cuda_assets/train_data.bin` (**1.4 GB — not committed**) + `train_labels.bin` (included) |

The CIFAR-10/MNIST test sets are exported to flat binary blobs from the standard
datasets (one file per image); the full DVSGesture test set (1078 samples) is
exported to `cuda_assets/train_data.bin`. A single DVSGesture sample
(`cuda_assets/single_data.bin`) is included for a quick correctness check.
See the project's export scripts (`binary_mnist_create.py`,
`binary_cifar10_create.py`, `export_dataset.py`) to regenerate these.

## Status

This release is assembled from the local working tree. The Task 1 **Step-by-Step
CUDA**, **Torch ANN**, and **SpikingJelly** baselines were authored on a rented
V100 server that has since been decommissioned and are **not yet re-added here**;
the Task 1 VD-TTFS method and its corrected event-driven CUDA baseline are
included. Task 2 and Task 3 carry their full implementation set.

## Citation

```bibtex
@inproceedings{vdttfs,
  title     = {VD-TTFS: Efficient TTFS Computing Architecture on Vector Devices},
  author    = {Liu, Shifeng and Yang, Zhijie and Wang, Dongsheng and Wang, Lei},
  year      = {2026}
}
```
