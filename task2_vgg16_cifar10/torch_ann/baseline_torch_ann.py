"""
通用框架 baseline（PyTorch · ANN 版）。

完成与 TTFS-SNN **等价的推理任务**：用 VGG16 对 CIFAR-10 分类。
但实现方式是标准 ANN 的**一次前向**（ReLU + maxpool），不做任何时间步模拟，
因此代表"通用 GPU 框架做这件事能达到的最佳吞吐"。

等价性依据：TTFS 编码是值保持的——像素 x 在 t=-tc·ln(x) 发放，conv0 的积分核把它解码回 x，
故 SNN 的 conv0 膜电位 = Σ w·x = ANN 的预激活。逐层同理。因此把同一份 snn_weights_vgg.bin
当作普通 ReLU ANN 跑、输入用同一份 cifar10_float，应复现≈90.7% 的精度。

用法： python baseline_torch_ann.py [num_images] [batch] [fp16]
"""
import sys, time
import torch
import torch.nn as nn
from weights_io import load_layers
from data_io import load_images_labels

# 性能开关：开启 TF32 / cuDNN autotune（最佳吞吐，精度无损）
torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

# VGG16：13 conv(3x3,pad1)+ReLU，5 个 2x2 maxpool；3 fc(512->512->512->10)
POOL_AFTER = {1, 3, 6, 9, 12}


class VGG16ANN(nn.Module):
    def __init__(self, layers):
        super().__init__()
        feats = []
        for i in range(13):
            cin, cout = layers[i].kernel.shape[1], layers[i].kernel.shape[0]
            conv = nn.Conv2d(cin, cout, 3, padding=1)
            conv.weight.data = torch.from_numpy(layers[i].kernel)
            conv.bias.data = torch.from_numpy(layers[i].bias)
            feats += [conv, nn.ReLU(inplace=True)]
            if i in POOL_AFTER:
                feats.append(nn.MaxPool2d(2))
        self.features = nn.Sequential(*feats)
        cls = []
        for j, i in enumerate([13, 14, 15]):
            cin, cout = layers[i].kernel.shape[1], layers[i].kernel.shape[0]
            lin = nn.Linear(cin, cout)
            lin.weight.data = torch.from_numpy(layers[i].kernel)
            lin.bias.data = torch.from_numpy(layers[i].bias)
            cls.append(lin)
            if i != 15:
                cls.append(nn.ReLU(inplace=True))
        self.classifier = nn.Sequential(*cls)

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        return self.classifier(x)


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 10000
    bs = int(sys.argv[2]) if len(sys.argv) > 2 else 1000
    use_fp16 = len(sys.argv) > 3 and sys.argv[3] == "fp16"
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[torch ANN baseline] device={dev} torch={torch.__version__} N={n} batch={bs} fp16={use_fp16}")

    layers = load_layers()
    net = VGG16ANN(layers).to(dev).eval()
    if dev == "cuda":
        net = net.to(memory_format=torch.channels_last)
    imgs, labels = load_images_labels(n, device=dev)
    if dev == "cuda":
        imgs = imgs.contiguous(memory_format=torch.channels_last)

    nb = (n + bs - 1) // bs

    @torch.no_grad()
    def infer(xb):
        if use_fp16:
            with torch.autocast("cuda", dtype=torch.float16):
                return net(xb).argmax(1)
        return net(xb).argmax(1)

    # 预热（触发 cuDNN autotune / 编译），不计时
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
    print(" PyTorch 通用 ANN baseline (VGG16, 等价推理任务)")
    print("=" * 46)
    print(f" Images       : {n}")
    print(f" Accuracy     : {correct/n*100:.2f} %")
    print(f" Total time   : {t_total:.4f} s   (仅推理，不含数据加载)")
    print(f" Throughput   : {n/t_total:.1f} img/s")
    print("=" * 46)


if __name__ == "__main__":
    main()
