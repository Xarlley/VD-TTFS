"""
Task 3 (7-layer DVSGesture SNN, T=160) multi-step LIF, SpikingJelly TRITON backend.
Random weights/inputs; latency + peak GPU memory via nvidia-smi. Requires the
sj_triton env (SpikingJelly >= 0.0.0.0.15). FC1 recurrence omitted (timing only).

  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python -u task3_triton.py [batch] [iters]
"""
import sys, os, time, threading, subprocess
import torch, torch.nn as nn
from spikingjelly.activation_based import neuron, layer, functional

BATCH = int(sys.argv[1]) if len(sys.argv) > 1 else 8
ITERS = int(sys.argv[2]) if len(sys.argv) > 2 else 3
T, SHP = 160, (2, 32, 32)

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
    return nn.Sequential(
        layer.Conv2d(2, 64, 3, padding=1),  neuron.LIFNode(),
        layer.Conv2d(64, 128, 3, padding=1), neuron.LIFNode(), layer.MaxPool2d(2),
        layer.Conv2d(128, 128, 3, padding=1), neuron.LIFNode(), layer.MaxPool2d(2),
        layer.Flatten(),
        layer.Linear(128 * 8 * 8, 128), neuron.LIFNode(),
        layer.Linear(128, 10), neuron.LIFNode(),
    )

def main():
    print(f"[task3/triton] T={T} batch={BATCH} iters={ITERS}", flush=True)
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
        print(f"[task3/triton] FAILED: {str(e)[:120]}", flush=True); return
    smp.run_flag = False; time.sleep(0.15)
    print(f"[task3/triton] {dt:.4f} s/batch | {BATCH/dt:.1f} samples/s | "
          f"{dt/BATCH*1e3:.3f} ms/sample | peak {smp.peak} MiB (nvidia-smi)", flush=True)

if __name__ == "__main__":
    main()
