"""
General-framework baseline (PyTorch ANN) for Task 1: LeNet on MNIST.

Performs the SAME inference task as the TTFS-SNN but as a single dense ReLU
forward pass (no time-step simulation) -- i.e. what a general GPU framework can
do at best throughput on this network.

Equivalence: TTFS encoding is value-preserving (pixel x fires at t = td - tc*ln x,
and conv0's integration kernel decodes it back to x), so the SNN's conv0 membrane
potential equals the ANN pre-activation; this holds layer by layer. Running the
same snn_weights.bin as a plain ReLU ANN on the same mnist_test10k reproduces the
SNN's accuracy.

LeNet: Conv1 5x5 1->12 (valid) + ReLU, MaxPool2; Conv2 5x5 12->64 (valid) + ReLU,
MaxPool2; flatten in H,W,C order (to match the SNN's FC input ordering); FC 1024->10.

Usage:  python baseline_torch_ann_mnist.py [num_images] [batch] [fp16]
"""
import sys, time
import torch
import torch.nn as nn
from weights_io import load_layers
from data_io import load_images_labels

torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True


class LeNetANN(nn.Module):
    def __init__(self, layers):
        super().__init__()
        # layers[0]: conv 1->12 5x5, layers[1]: conv 12->64 5x5, layers[2]: fc 1024->10
        c0 = nn.Conv2d(layers[0].kernel.shape[1], layers[0].kernel.shape[0], 5, padding=0)
        c0.weight.data = torch.from_numpy(layers[0].kernel)
        c0.bias.data = torch.from_numpy(layers[0].bias)
        c1 = nn.Conv2d(layers[1].kernel.shape[1], layers[1].kernel.shape[0], 5, padding=0)
        c1.weight.data = torch.from_numpy(layers[1].kernel)
        c1.bias.data = torch.from_numpy(layers[1].bias)
        self.conv1, self.conv2 = c0, c1
        self.pool = nn.MaxPool2d(2)
        self.relu = nn.ReLU(inplace=True)
        fc = nn.Linear(layers[2].kernel.shape[1], layers[2].kernel.shape[0])
        fc.weight.data = torch.from_numpy(layers[2].kernel)
        fc.bias.data = torch.from_numpy(layers[2].bias)
        self.fc = fc

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))   # (N,12,24,24) -> (N,12,12,12)
        x = self.pool(self.relu(self.conv2(x)))   # (N,64,8,8)   -> (N,64,4,4)
        # flatten in (H, W, C) order -- the SNN's FC weights index cin = (h*4+w)*64 + c
        x = x.permute(0, 2, 3, 1).reshape(x.shape[0], -1)
        return self.fc(x)


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 10000
    bs = int(sys.argv[2]) if len(sys.argv) > 2 else 1000
    use_fp16 = len(sys.argv) > 3 and sys.argv[3] == "fp16"
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[torch ANN baseline | LeNet-MNIST] device={dev} torch={torch.__version__} N={n} batch={bs} fp16={use_fp16}")

    layers = load_layers()
    net = LeNetANN(layers).to(dev).eval()
    imgs, labels = load_images_labels(n, device=dev)

    nb = (n + bs - 1) // bs

    @torch.no_grad()
    def infer(xb):
        if use_fp16:
            with torch.autocast("cuda", dtype=torch.float16):
                return net(xb).argmax(1)
        return net(xb).argmax(1)

    with torch.no_grad():
        _ = infer(imgs[:min(bs, n)])
    if dev == "cuda":
        torch.cuda.synchronize()

    correct, t_total = 0, 0.0
    for b in range(nb):
        xb = imgs[b * bs:(b + 1) * bs]
        yb = labels[b * bs:(b + 1) * bs]
        if dev == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        pred = infer(xb)
        if dev == "cuda":
            torch.cuda.synchronize()
        t_total += time.perf_counter() - t0
        correct += (pred == yb).sum().item()

    print("\n" + "=" * 46)
    print(" PyTorch ANN baseline (LeNet-MNIST, equivalent task)")
    print("=" * 46)
    print(f" Images       : {n}")
    print(f" Accuracy     : {correct/n*100:.2f} %")
    print(f" Total time   : {t_total:.4f} s   (inference only)")
    print(f" Throughput   : {n/t_total:.1f} img/s")
    print("=" * 46)


if __name__ == "__main__":
    main()
