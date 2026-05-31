"""
通用框架 baseline #1：纯 PyTorch 的时间驱动 TTFS-SNN 推理。

与手写 CUDA 实现等价：相同网络结构（VGG16）、相同权重
（exported_models/snn_weights_vgg.bin）、相同数据集（CIFAR-10 测试集）、
相同的 T2FSNN 时间驱动 TTFS 动力学。

动力学（与 cuda/bench_vgg_timedriven_baseline.cu 逐条对应）：
  全局时间步 t = 0..679，时间窗 TW=80，层间错峰 TFS=40。
  输入编码：t_spike = ceil(max(0, -tc_in*ln(pixel)))，pixel>=1e-5 且 t_spike<=80 才发放。
  第 i 层（depth=i+1）：
    t_integ = t - i*40，   积分核 kernel_val = exp(-(t_integ - td_integ)/tc_integ)
    t_fire  = t - (i+1)*40，阈值 vth = exp(-(t_fire - td_fire)/tc_fire)
    积分窗 [0,80) 内：vmem += conv/linear(当前步输入脉冲) * kernel_val
    发放窗 [start_fire,80) 内：vmem+bias>=vth 且 vth>=1e-5 -> 发放一次（latch）
    (tc_integ,td_integ)=上一层(tc_fire,td)，第 0 层用 (tc_in,td_in)。
  池化：2x2 脉冲 OR（latch）。输出层：取积分窗内 vmem+bias 的逐时刻最大值，argmax 为预测。

用法： python baseline_torch.py [num_images] [batch_size]
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

# VGG16 层表：(类型, C_in, C_out, H, W, 卷积后是否池化)
# 13 conv + 3 fc，5 个 2x2 maxpool。
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
    ("fc",   512, 10,  1,  1,  False),   # 输出层
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
        # imgs: [N,3,32,32]，返回每像素的发放步（不发放记 -1）
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

        # 持久状态：各层 vmem / fired
        vmem, fired = [], []
        for (typ, cin, cout, H, Wd, pool) in ARCH:
            shape = (N, cout, H, Wd) if typ == "conv" else (N, cout)
            vmem.append(torch.zeros(shape, device=dev))
            fired.append(torch.zeros(shape, dtype=torch.bool, device=dev))
        pool_fired = {}   # 层索引 -> [N,cout,H/2,W/2] bool
        for i, (typ, cin, cout, H, Wd, pool) in enumerate(ARCH):
            if pool:
                pool_fired[i] = torch.zeros((N, cout, H // 2, Wd // 2), dtype=torch.bool, device=dev)
        max_vmem = torch.full((N, 10), -1e9, device=dev)

        for t in range(TIME_STEPS):
            spikes = (spike_time == t).float()  # 第 0 层输入脉冲 [N,3,32,32]
            for i, (typ, cin, cout, H, Wd, pool) in enumerate(ARCH):
                t_integ = t - i * TFS
                t_fire = t - (i + 1) * TFS
                is_out = (i == len(ARCH) - 1)

                # ---- 积分 ----
                if 0.0 <= t_integ < TW:
                    kv = math.exp(-(t_integ - self.td_i[i]) / self.tc_i[i])
                    if typ == "conv":
                        psp = F.conv2d(spikes, self.W[i], None, stride=1, padding=1)
                    else:
                        psp = F.linear(spikes, self.W[i], None)
                    vmem[i] += psp * kv

                # ---- 发放 / 输出 ----
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
                    # ---- 池化（2x2 脉冲 OR + latch）----
                    if pool:
                        pooled = F.max_pool2d(spikes, 2)
                        newly_p = (pooled > 0) & (~pool_fired[i])
                        spikes = newly_p.float()
                        pool_fired[i] |= newly_p
                    if typ == "conv" and ARCH[i + 1][0] == "fc":
                        spikes = spikes.flatten(1)  # 进入 FC 前展平

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
    print(" PyTorch 通用 TTFS-SNN baseline")
    print("=" * 46)
    print(f" Images       : {n}")
    print(f" Accuracy     : {correct/n*100:.2f} %")
    print(f" Total time   : {t_total:.3f} s   (仅推理，不含数据加载)")
    print(f" Throughput   : {n/t_total:.2f} img/s")
    print("=" * 46)


if __name__ == "__main__":
    main()
