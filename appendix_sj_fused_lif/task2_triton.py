"""
Task 2 (VGG-16/CIFAR-10, T=680) multi-step LIF, SpikingJelly TRITON backend.
Random weights/inputs; latency + peak GPU memory via nvidia-smi. Requires the
sj_triton env (SpikingJelly >= 0.0.0.0.15).

NOTE: multi-step materializes all T=680 timesteps' activations, which is very
memory-heavy for VGG-16; use a small batch (default 1) and expect possible OOM.

  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python -u task2_triton.py [batch] [iters]
"""
import sys, os, time, threading, subprocess
import torch, torch.nn as nn
from spikingjelly.activation_based import neuron, layer, functional

BATCH = int(sys.argv[1]) if len(sys.argv) > 1 else 1
ITERS = int(sys.argv[2]) if len(sys.argv) > 2 else 3
T, SHP = 680, (3, 32, 32)

class GPUSampler(threading.Thread):
    def __init__(self):
        super().__init__(daemon=True); self.peak = 0; self.run_flag = True; self.pid = os.getpid()
    def run(self):
        while self.run_flag:
            try:
                out = subprocess.run(["nvidia-smi", "--query-compute-apps=pid,used_memory",
                                      "--format=csv,noheader,nounits"], capture_output=True, text=True).stdout
                for line in out.strip().splitlines():
                    parts = [s.strip() for s in line.split(",")]
                    if len(parts) >= 2 and parts[0].isdigit() and int(parts[0]) == self.pid:
                        self.peak = max(self.peak, int(parts[1]))
            except Exception:
                pass
            time.sleep(0.05)

def build():
    cfg = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M',
           512, 512, 512, 'M', 512, 512, 512, 'M']
    layers, cin = [], 3
    for v in cfg:
        if v == 'M':
            layers.append(layer.MaxPool2d(2))
        else:
            layers += [layer.Conv2d(cin, v, 3, padding=1), neuron.LIFNode()]; cin = v
    layers += [layer.Flatten(),
               layer.Linear(512, 512), neuron.LIFNode(),
               layer.Linear(512, 512), neuron.LIFNode(),
               layer.Linear(512, 10),  neuron.LIFNode()]
    return nn.Sequential(*layers)

def main():
    print(f"[task2/triton] T={T} batch={BATCH} iters={ITERS}", flush=True)
    net = build().cuda().eval()
    functional.set_step_mode(net, 'm')
    functional.set_backend(net, 'triton', instance=neuron.LIFNode)
    x = torch.rand((T, BATCH) + SHP, device='cuda')
    smp = GPUSampler(); smp.start()
    try:
        with torch.no_grad():
            functional.reset_net(net); _ = net(x); torch.cuda.synchronize()
            t0 = time.perf_counter()
            for _ in range(ITERS):
                functional.reset_net(net); _ = net(x)
            torch.cuda.synchronize(); dt = (time.perf_counter() - t0) / ITERS
    except RuntimeError as e:
        smp.run_flag = False
        print(f"[task2/triton] FAILED: {str(e)[:120]}", flush=True); return
    smp.run_flag = False; time.sleep(0.15)
    print(f"[task2/triton] {dt:.4f} s/batch | {BATCH/dt:.1f} samples/s | "
          f"{dt/BATCH*1e3:.3f} ms/sample | peak {smp.peak} MiB (nvidia-smi)", flush=True)

if __name__ == "__main__":
    main()
