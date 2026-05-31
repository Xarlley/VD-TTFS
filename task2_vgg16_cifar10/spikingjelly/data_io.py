"""Load CIFAR-10 test images and labels (same binary as the CUDA side).
Each image is 3072 float32 in HWC; labels are one-hot float32 (10 per image).
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
