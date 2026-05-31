"""读取 CIFAR-10 测试集（与 CUDA 端完全相同的二进制）以及标签。

每张图 dataset_downloaded/cifar10_float/{i}.bin = 3072 个 float32，
布局为 HWC：index = (h*32 + w)*3 + c。
标签 label_onehot：每张 10 个 float32 的 one-hot。
"""
import numpy as np
import torch

IMG_DIR = "dataset_downloaded/cifar10_float"


def load_images_labels(n, device="cpu"):
    imgs = np.empty((n, 3, 32, 32), dtype=np.float32)
    for i in range(n):
        with open(f"{IMG_DIR}/{i}.bin", "rb") as f:
            a = np.frombuffer(f.read(3072 * 4), dtype="<f4").reshape(32, 32, 3)
        imgs[i] = a.transpose(2, 0, 1)  # HWC -> CHW
    with open(f"{IMG_DIR}/label_onehot", "rb") as f:
        lbl = np.frombuffer(f.read(n * 10 * 4), dtype="<f4").reshape(n, 10)
    labels = lbl.argmax(axis=1).astype(np.int64)
    return (torch.from_numpy(imgs).to(device),
            torch.from_numpy(labels).to(device))
