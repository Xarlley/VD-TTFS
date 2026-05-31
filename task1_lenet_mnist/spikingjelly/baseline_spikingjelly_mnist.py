"""
General-framework baseline (SpikingJelly) for Task 1: LeNet on MNIST.

Time-driven TTFS-SNN inference built with SpikingJelly's
`activation_based.layer` synaptic modules (step_mode='s') and a custom TTFS
integrate-and-fire rule, to reflect a "general SNN framework" implementation and
its overhead. Weights / dataset / TTFS dynamics are equivalent to the CUDA and
PyTorch-ANN baselines and to VD-TTFS.

Unlike the staggered VGG/CIFAR schedule, LeNet-MNIST uses NO inter-layer temporal
offset: the network is a feed-forward sweep where each layer integrates its input
spike-times over [0,T), records each neuron's first threshold crossing, and hands
the resulting spike-times to the next layer. The output layer fires like any other
and the prediction is the argmin of its first-spike times.

Usage:  python baseline_spikingjelly_mnist.py [num_images] [batch_size]
"""
import sys, time, math
import torch
import torch.nn as nn
import torch.nn.functional as F
from spikingjelly.activation_based import layer

from weights_io import load_layers
from data_io import load_images_labels

TW = 80                 # time window
VTH = 1.0               # V0
TC_IN, TD_IN = 17.452274, 0.0   # input encoding constants
SPK_EPS = 1e-5          # pixel below this never spikes
VTH_FLOOR = 1e-5        # threshold below this is treated as inactive
INF = 9999.0


def min_pool2x2(x):
    """2x2 stride-2 min-time pooling (earliest spike wins)."""
    return -F.max_pool2d(-x, 2)


@torch.no_grad()
def run_layer(spike_time, synapse, is_conv, bias,
              tc_i, td_i, tc_f, td_f):
    """Integrate one layer over [0,T); return per-output first-spike times (INF if none)."""
    t_min = math.ceil(td_f) if td_f > 0 else 0
    v = fired = result = bview = None
    for t in range(TW):
        spikes_t = ((torch.floor(spike_time) == t) & (spike_time < INF)).float()
        cur = synapse(spikes_t)
        if v is None:
            v = torch.zeros_like(cur)
            fired = torch.zeros_like(cur, dtype=torch.bool)
            result = torch.full_like(cur, INF)
            bview = torch.from_numpy(bias).to(cur.device)
            bview = bview.view(1, -1, 1, 1) if is_conv else bview.view(1, -1)
        kv = math.exp(-(t - td_i) / tc_i)
        v = v + cur * kv
        if t >= t_min:
            vth = VTH * math.exp(-(t - td_f) / tc_f)
            if vth >= VTH_FLOOR:
                newly = ((v + bview) >= vth) & (~fired)
                result = torch.where(newly, torch.full_like(result, float(t)), result)
                fired = fired | newly
    return result


class SJLeNet(nn.Module):
    def __init__(self, layers):
        super().__init__()
        self.layers_meta = layers
        c0 = layer.Conv2d(layers[0].kernel.shape[1], layers[0].kernel.shape[0], 5,
                          stride=1, padding=0, bias=False, step_mode="s")
        c1 = layer.Conv2d(layers[1].kernel.shape[1], layers[1].kernel.shape[0], 5,
                          stride=1, padding=0, bias=False, step_mode="s")
        fc = layer.Linear(layers[2].kernel.shape[1], layers[2].kernel.shape[0],
                          bias=False, step_mode="s")
        c0.weight.data = torch.from_numpy(layers[0].kernel)
        c1.weight.data = torch.from_numpy(layers[1].kernel)
        fc.weight.data = torch.from_numpy(layers[2].kernel)
        self.conv1, self.conv2, self.fc = c0, c1, fc

    @torch.no_grad()
    def run_batch(self, imgs):
        L = self.layers_meta
        # input encoding: pixel -> integer spike time (matches k_encode_image)
        tfloat = TD_IN - TC_IN * torch.log(imgs.clamp_min(1e-30))
        tspike = torch.ceil(tfloat.clamp_min(0.0))
        st0 = torch.where((imgs >= SPK_EPS) & (tspike <= TW), tspike,
                          torch.full_like(tspike, INF))

        # Conv1 (integ const = input encoding constants)
        st = run_layer(st0, self.conv1, True, L[0].bias, TC_IN, TD_IN, L[0].tc_fire, L[0].td)
        st = min_pool2x2(st)                                   # (N,12,12,12)
        # Conv2 (integ const = layer0 fire constants)
        st = run_layer(st, self.conv2, True, L[1].bias, L[0].tc_fire, L[0].td, L[1].tc_fire, L[1].td)
        st = min_pool2x2(st)                                   # (N,64,4,4)
        # flatten H,W,C to match the SNN FC ordering
        st = st.permute(0, 2, 3, 1).reshape(st.shape[0], -1)   # (N,1024)
        # FC (integ const = layer1 fire constants); output fires, predict argmin time
        st = run_layer(st, self.fc, False, L[2].bias, L[1].tc_fire, L[1].td, L[2].tc_fire, L[2].td)
        return st.argmin(dim=1)


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 10000
    bs = int(sys.argv[2]) if len(sys.argv) > 2 else 1000
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[spikingjelly baseline | LeNet-MNIST] device={dev} torch={torch.__version__} N={n} batch={bs}")

    layers = load_layers()
    net = SJLeNet(layers).to(dev)
    imgs, labels = load_images_labels(n, device=dev)

    correct, t_total = 0, 0.0
    nb = (n + bs - 1) // bs
    for b in range(nb):
        xb = imgs[b * bs:(b + 1) * bs]
        yb = labels[b * bs:(b + 1) * bs]
        if dev == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        pred = net.run_batch(xb)
        if dev == "cuda":
            torch.cuda.synchronize()
        t_total += time.perf_counter() - t0
        correct += (pred == yb).sum().item()
        print(f"  batch {b+1}/{nb}  acc_so_far={correct/((b+1)*bs)*100:.2f}%  elapsed={t_total:.1f}s", flush=True)

    print("\n" + "=" * 46)
    print(" SpikingJelly TTFS-SNN baseline (LeNet-MNIST)")
    print("=" * 46)
    print(f" Images       : {n}")
    print(f" Accuracy     : {correct/n*100:.2f} %")
    print(f" Total time   : {t_total:.3f} s   (inference only)")
    print(f" Throughput   : {n/t_total:.2f} img/s")
    print("=" * 46)


if __name__ == "__main__":
    main()
