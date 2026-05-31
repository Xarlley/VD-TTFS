"""Read the MNIST test set (the same binary the CUDA programs use) + labels.

Each image dataset_downloaded/mnist_test10k/{i}.bin = 784 float32 in [0,1],
laid out as a 28x28 single-channel image (row-major H,W).
label_onehot: 10 float32 one-hot per image.
"""
import numpy as np
import torch

IMG_DIR = "dataset_downloaded/mnist_test10k"


def load_images_labels(n, device="cpu"):
    imgs = np.empty((n, 1, 28, 28), dtype=np.float32)
    for i in range(n):
        with open(f"{IMG_DIR}/{i}.bin", "rb") as f:
            a = np.frombuffer(f.read(784 * 4), dtype="<f4").reshape(28, 28)
        imgs[i, 0] = a
    with open(f"{IMG_DIR}/label_onehot", "rb") as f:
        lbl = np.frombuffer(f.read(n * 10 * 4), dtype="<f4").reshape(n, 10)
    labels = lbl.argmax(axis=1).astype(np.int64)
    return (torch.from_numpy(imgs).to(device),
            torch.from_numpy(labels).to(device))
