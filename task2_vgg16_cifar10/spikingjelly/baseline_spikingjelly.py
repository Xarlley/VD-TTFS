"""
通用框架 baseline #2：基于 SpikingJelly 的时间驱动 TTFS-SNN 推理。

与 baseline_torch.py 数学完全一致，但网络用 SpikingJelly 的
`activation_based.layer` 突触模块搭建，神经元用自定义的
`base.MemoryModule` 子类（带状态记忆，配合 functional.reset_net 复位），
以体现"通用 SNN 框架"的写法与开销。

权重 / 数据集 / TTFS 动力学与 CUDA 端、torch 端三者等价。

用法： python baseline_spikingjelly.py [num_images] [batch_size]
"""
import sys, time, math
import torch
import torch.nn as nn
from spikingjelly.activation_based import layer, base, functional

from weights_io import load_layers
from data_io import load_images_labels
from baseline_torch import ARCH, TIME_STEPS, TW, TFS, VTH, TC_IN, TD_IN, SPK_EPS, VTH_FLOOR


class TTFSNode(base.MemoryModule):
    """自定义 TTFS 神经元：时间窗内按 exp 核积分，过（随时间衰减的）阈值发放一次。"""
    def __init__(self, i, is_conv, is_out, bias, tc_i, td_i, tc_f, td_f):
        super().__init__()
        self.i, self.is_conv, self.is_out = i, is_conv, is_out
        self.register_buffer("bias", torch.from_numpy(bias))
        self.tc_i, self.td_i, self.tc_f, self.td_f = tc_i, td_i, tc_f, td_f
        self.register_memory("v", None)
        self.register_memory("fired", None)
        self.register_memory("max_v", None)

    def single_step_forward(self, x, t):
        # x 为本步突触电流；x=None 表示积分窗已关闭、只做发放判定（无新输入）。
        if self.v is None:
            self.v = torch.zeros_like(x)
            self.fired = torch.zeros_like(x, dtype=torch.bool)
            if self.is_out:
                self.max_v = torch.full_like(x, -1e9)
        t_integ = t - self.i * TFS
        t_fire = t - (self.i + 1) * TFS
        bview = self.bias.view(1, -1, 1, 1) if self.is_conv else self.bias.view(1, -1)

        if x is not None and 0.0 <= t_integ < TW:
            kv = math.exp(-(t_integ - self.td_i) / self.tc_i)
            self.v = self.v + x * kv

        if self.is_out:
            if 0.0 <= t_integ < TW:
                self.max_v = torch.maximum(self.max_v, self.v + bview)
            return None

        out = torch.zeros_like(self.v)
        start_fire = math.ceil(self.td_f) if self.td_f > 0 else 0
        if start_fire <= t_fire < TW:
            vth = VTH * math.exp(-(t_fire - self.td_f) / self.tc_f)
            if vth >= VTH_FLOOR:
                newly = (self.v + bview >= vth) & (~self.fired)
                out = newly.float()
                self.fired = self.fired | newly
        return out


class PoolLatch(base.MemoryModule):
    """2x2 脉冲 OR 池化，带 latch（每个池化位置只发放一次）。"""
    def __init__(self):
        super().__init__()
        self.pool = nn.MaxPool2d(2)
        self.register_memory("fired", None)

    def single_step_forward(self, x):
        p = self.pool(x)
        if self.fired is None:
            self.fired = torch.zeros_like(p, dtype=torch.bool)
        newly = (p > 0) & (~self.fired)
        self.fired = self.fired | newly
        return newly.float()


class SJNet(nn.Module):
    def __init__(self, layers):
        super().__init__()
        self.syn = nn.ModuleList()
        self.nodes = nn.ModuleList()
        self.pools = nn.ModuleDict()
        for i, (typ, cin, cout, H, Wd, pool) in enumerate(ARCH):
            if typ == "conv":
                m = layer.Conv2d(cin, cout, 3, stride=1, padding=1, bias=False, step_mode="s")
            else:
                m = layer.Linear(cin, cout, bias=False, step_mode="s")
            m.weight.data = torch.from_numpy(layers[i].kernel)
            self.syn.append(m)
            tc_i = TC_IN if i == 0 else layers[i - 1].tc_fire
            td_i = TD_IN if i == 0 else layers[i - 1].td
            self.nodes.append(TTFSNode(i, typ == "conv", i == len(ARCH) - 1,
                                       layers[i].bias, tc_i, td_i,
                                       layers[i].tc_fire, layers[i].td))
            if pool:
                self.pools[str(i)] = PoolLatch()

    @torch.no_grad()
    def run_batch(self, imgs):
        functional.reset_net(self)
        # 输入编码（与 torch 版一致）
        tfloat = (-TC_IN) * torch.log(imgs.clamp_min(1e-30)) + TD_IN
        tspike = torch.ceil(tfloat.clamp_min(0.0))
        valid = (imgs >= SPK_EPS) & (tspike <= TW)
        spike_time = torch.where(valid, tspike, torch.full_like(tspike, -1.0))

        out_node = self.nodes[-1]
        N = imgs.shape[0]
        # 各层"交给下一层"的零张量（当本层处于非活跃时间窗时直接复用，省掉无谓计算）
        zero_handoff = []
        for i, (typ, cin, cout, H, Wd, pool) in enumerate(ARCH):
            if i == len(ARCH) - 1:
                zero_handoff.append(None)
            elif pool:
                zero_handoff.append(torch.zeros((N, cout, H // 2, Wd // 2), device=imgs.device))
            elif typ == "conv" and ARCH[i + 1][0] == "fc":
                zero_handoff.append(torch.zeros((N, cout), device=imgs.device))
            elif typ == "conv":
                zero_handoff.append(torch.zeros((N, cout, H, Wd), device=imgs.device))
            else:
                zero_handoff.append(torch.zeros((N, cout), device=imgs.device))

        for t in range(TIME_STEPS):
            spikes = (spike_time == t).float()
            for i, (typ, cin, cout, H, Wd, pool) in enumerate(ARCH):
                # 本层活跃时间窗：积分窗 [i*40, i*40+80) ∪ 发放窗 [(i+1)*40, (i+1)*40+80)
                node_active = (i * TFS <= t < (i + 1) * TFS + TW)
                if not node_active:
                    if i != len(ARCH) - 1:
                        spikes = zero_handoff[i]
                    continue
                integ_active = (i * TFS <= t < i * TFS + TW)
                cur = self.syn[i](spikes) if integ_active else None
                out = self.nodes[i](cur, t)
                if i == len(ARCH) - 1:
                    break
                spikes = out
                if pool:
                    spikes = self.pools[str(i)](spikes)
                if typ == "conv" and ARCH[i + 1][0] == "fc":
                    spikes = spikes.flatten(1)
        return out_node.max_v.argmax(dim=1)


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 10000
    bs = int(sys.argv[2]) if len(sys.argv) > 2 else 1000
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[spikingjelly baseline] device={dev} torch={torch.__version__} N={n} batch={bs}")

    layers = load_layers()
    net = SJNet(layers).to(dev)
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
    print(" SpikingJelly 通用 TTFS-SNN baseline")
    print("=" * 46)
    print(f" Images       : {n}")
    print(f" Accuracy     : {correct/n*100:.2f} %")
    print(f" Total time   : {t_total:.3f} s   (仅推理，不含数据加载)")
    print(f" Throughput   : {n/t_total:.2f} img/s")
    print("=" * 46)


if __name__ == "__main__":
    main()
