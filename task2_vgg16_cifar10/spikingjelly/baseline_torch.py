"""
Baseline #1: pure PyTorch time-driven TTFS-SNN inference (VGG16, CIFAR-10).
Equivalent to the CUDA implementation (same weights, dataset, TTFS dynamics).
Usage: python baseline_torch.py [num_images] [batch_size]
"""
import sys, time, math
import torch
import torch.nn.functional as F
from weights_io import load_layers
from data_io import load_images_labels

TIME_STEPS = 680
TW = 80.0
TFS = 40.0
VTH = 1.0
TC_IN, TD_IN = 34.750164, 0.0
SPK_EPS, VTH_FLOOR = 1e-5, 1e-5

# VGG16 layer table: (type, C_in, C_out, H, W, pool after conv).
# 13 conv + 3 fc, 5 x 2x2 maxpool.
ARCH = [
    ("conv", 3,   64,  32, 32, False),
    ("conv", 64,  64,  32, 32, True),    # -> 16
    ("conv", 64,  128, 16, 16, False),
    ("conv", 128, 128, 16, 16, True),    # -> 8
    ("conv", 128, 256, 8,  8,  False),
    ("conv", 256, 256, 8,  8,  False),
    ("conv", 256, 256, 8,  8,  True),    # -> 4
    ("conv", 256, 512, 4,  4,  False),
    ("conv", 512, 512, 4,  4,  False),
    ("conv", 512, 512, 4,  4,  True),    # -> 2
    ("conv", 512, 512, 2,  2,  False),
    ("conv", 512, 512, 2,  2,  False),
    ("conv", 512, 512, 2,  2,  True),    # -> 1
    ("fc",   512, 512, 1,  1,  False),
    ("fc",   512, 512, 1,  1,  False),
    ("fc",   512, 10,  1,  1,  False),   # output layer
]


class TTFSNet:
    def __init__(self, layers, device):
        self.device = device
        self.L = layers
        self.W, self.bias, self.tc_f, self.td_f = [], [], [], []
        self.tc_i, self.td_i = [], []
        for i, (typ, cin, cout, H, Wd, pool) in enumerate(ARCH):
            self.W.append(torch.from_numpy(layers[i].kernel).to(device))
            self.bias.append(torch.from_numpy(layers[i].bias).to(device))
            self.tc_f.append(layers[i].tc_fire); self.td_f.append(layers[i].td)
            if i == 0:
                self.tc_i.append(TC_IN); self.td_i.append(TD_IN)
            else:
                self.tc_i.append(layers[i - 1].tc_fire); self.td_i.append(layers[i - 1].td)

    @torch.no_grad()
    def encode_spike_times(self, imgs):
        # imgs: [N,3,32,32]; return per-pixel spike step (-1 if no spike)
        x = imgs
        tfloat = (-TC_IN) * torch.log(x.clamp_min(1e-30)) + TD_IN
        tspike = torch.ceil(tfloat.clamp_min(0.0))
        valid = (x >= SPK_EPS) & (tspike <= TW)
        return torch.where(valid, tspike, torch.full_like(tspike, -1.0))

    @torch.no_grad()
    def run_batch(self, imgs):
        dev = self.device
        N = imgs.shape[0]
        spike_time = self.encode_spike_times(imgs)  # [N,3,32,32]

        # persistent per-layer state: vmem / fired
        vmem, fired = [], []
        for (typ, cin, cout, H, Wd, pool) in ARCH:
            shape = (N, cout, H, Wd) if typ == "conv" else (N, cout)
            vmem.append(torch.zeros(shape, device=dev))
            fired.append(torch.zeros(shape, dtype=torch.bool, device=dev))
        pool_fired = {}   # layer index -> [N,cout,H/2,W/2] bool
        for i, (typ, cin, cout, H, Wd, pool) in enumerate(ARCH):
            if pool:
                pool_fired[i] = torch.zeros((N, cout, H // 2, Wd // 2), dtype=torch.bool, device=dev)
        max_vmem = torch.full((N, 10), -1e9, device=dev)

        for t in range(TIME_STEPS):
            spikes = (spike_time == t).float()  # layer-0 input spikes [N,3,32,32]
            for i, (typ, cin, cout, H, Wd, pool) in enumerate(ARCH):
                t_integ = t - i * TFS
                t_fire = t - (i + 1) * TFS
                is_out = (i == len(ARCH) - 1)

                # ---- integrate ----
                if 0.0 <= t_integ < TW:
                    kv = math.exp(-(t_integ - self.td_i[i]) / self.tc_i[i])
                    if typ == "conv":
                        psp = F.conv2d(spikes, self.W[i], None, stride=1, padding=1)
                    else:
                        psp = F.linear(spikes, self.W[i], None)
                    vmem[i] += psp * kv

                # ---- fire / output ----
                if is_out:
                    if 0.0 <= t_integ < TW:
                        cur = vmem[i] + self.bias[i].view(1, -1)
                        max_vmem = torch.maximum(max_vmem, cur)
                    spikes = None
                else:
                    bview = self.bias[i].view(1, -1, 1, 1) if typ == "conv" else self.bias[i].view(1, -1)
                    out = torch.zeros_like(vmem[i])
                    start_fire = math.ceil(self.td_f[i]) if self.td_f[i] > 0 else 0
                    if start_fire <= t_fire < TW:
                        vth = VTH * math.exp(-(t_fire - self.td_f[i]) / self.tc_f[i])
                        if vth >= VTH_FLOOR:
                            newly = (vmem[i] + bview >= vth) & (~fired[i])
                            out = newly.float()
                            fired[i] |= newly
                    spikes = out
                    # ---- pooling (2x2 spike OR + latch) ----
                    if pool:
                        pooled = F.max_pool2d(spikes, 2)
                        newly_p = (pooled > 0) & (~pool_fired[i])
                        spikes = newly_p.float()
                        pool_fired[i] |= newly_p
                    if typ == "conv" and ARCH[i + 1][0] == "fc":
                        spikes = spikes.flatten(1)  # flatten before FC

        return max_vmem.argmax(dim=1)


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 10000
    bs = int(sys.argv[2]) if len(sys.argv) > 2 else 1000
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[torch baseline] device={dev} torch={torch.__version__} N={n} batch={bs}")

    layers = load_layers()
    net = TTFSNet(layers, dev)
    imgs, labels = load_images_labels(n, device=dev)

    correct = 0
    t_total = 0.0
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
    print(" PyTorch generic TTFS-SNN baseline")
    print("=" * 46)
    print(f" Images       : {n}")
    print(f" Accuracy     : {correct/n*100:.2f} %")
    print(f" Total time   : {t_total:.3f} s   (inference only, excl. data loading)")
    print(f" Throughput   : {n/t_total:.2f} img/s")
    print("=" * 46)


if __name__ == "__main__":
    main()
