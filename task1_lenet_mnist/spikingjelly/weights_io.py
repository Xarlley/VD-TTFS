"""Parse exported_models/snn_weights.bin (LeNet-MNIST).

Binary layout (little-endian, identical to the CUDA-side load_weights):
  int32 num_layers
  per layer:
    int32 ndim, ndim x int32 kernel shape
        ndim==4 -> [k_h, k_w, c_in, c_out]   (TensorFlow HWIO)
        ndim==2 -> [c_in, c_out]
    float32 * prod(shape)  kernel data (C order of the shape above)
    int32 ndim_b, int32 b_size, float32 * b_size   bias
    int32 ndim_tc, int32 tc_size(=1), float32 tc_fire
    int32 ndim_td, int32 td_size(=1), float32 td
"""
import struct
import numpy as np


class Layer:
    def __init__(self, kernel, bias, tc_fire, td, is_conv):
        self.kernel = kernel        # conv: OIHW (torch);  fc: [C_out, C_in]
        self.bias = bias            # [C_out]
        self.tc_fire = float(tc_fire)
        self.td = float(td)
        self.is_conv = is_conv

    def __repr__(self):
        return (f"Layer(conv={self.is_conv}, w={tuple(self.kernel.shape)}, "
                f"b={self.bias.shape[0]}, tc_fire={self.tc_fire:.4f}, td={self.td:.4f})")


def load_layers(path="exported_models/snn_weights.bin"):
    with open(path, "rb") as f:
        buf = f.read()
    off = 0

    def ri():
        nonlocal off
        v = struct.unpack_from("<i", buf, off)[0]; off += 4; return v

    def rf():
        nonlocal off
        v = struct.unpack_from("<f", buf, off)[0]; off += 4; return v

    def rarr(n):
        nonlocal off
        a = np.frombuffer(buf, dtype="<f4", count=n, offset=off).copy(); off += 4 * n; return a

    n_layers = ri()
    layers = []
    for _ in range(n_layers):
        nd = ri()
        shp = [ri() for _ in range(nd)]
        ksz = int(np.prod(shp))
        kflat = rarr(ksz)
        if nd == 4:
            kh, kw, cin, cout = shp
            # HWIO -> torch conv weight OIHW
            k = kflat.reshape(kh, kw, cin, cout).transpose(3, 2, 0, 1).copy()
            is_conv = True
        else:
            cin, cout = shp
            # [C_in, C_out] -> torch Linear weight [C_out, C_in]
            k = kflat.reshape(cin, cout).transpose(1, 0).copy()
            is_conv = False
        _ = ri(); bsz = ri(); bias = rarr(bsz)
        _ = ri(); _ = ri(); tc_fire = rf()
        _ = ri(); _ = ri(); td = rf()
        layers.append(Layer(k, bias, tc_fire, td, is_conv))
    assert off == len(buf), f"not fully read: off={off} size={len(buf)}"
    return layers


if __name__ == "__main__":
    L = load_layers()
    print(f"num_layers = {len(L)}")
    for i, l in enumerate(L):
        print(f"L{i:2d} {l}")
