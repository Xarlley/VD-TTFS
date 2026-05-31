"""
解析 exported_models/snn_weights_vgg.bin。

二进制布局（小端，与 CUDA 端 load_weights 完全一致）：
  int32 num_layers
  对每层:
    int32 ndim, ndim 个 int32 kernel shape
        ndim==4 -> [k_h, k_w, c_in, c_out]   (TensorFlow HWIO)
        ndim==2 -> [c_in, c_out]
    float32 * prod(shape)  kernel 数据（按上面 shape 的 C 顺序）
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


def load_layers(path="exported_models/snn_weights_vgg.bin"):
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
            # 存储为 HWIO -> 转成 torch 卷积权重 OIHW
            k = kflat.reshape(kh, kw, cin, cout).transpose(3, 2, 0, 1).copy()
            is_conv = True
        else:
            cin, cout = shp
            # 存储为 [C_in, C_out] -> 转成 torch Linear 权重 [C_out, C_in]
            k = kflat.reshape(cin, cout).transpose(1, 0).copy()
            is_conv = False
        _ = ri(); bsz = ri(); bias = rarr(bsz)
        _ = ri(); _ = ri(); tc_fire = rf()
        _ = ri(); _ = ri(); td = rf()
        layers.append(Layer(k, bias, tc_fire, td, is_conv))
    assert off == len(buf), f"未读完: off={off} size={len(buf)}"
    return layers


if __name__ == "__main__":
    L = load_layers()
    print(f"num_layers = {len(L)}")
    for i, l in enumerate(L):
        print(f"L{i:2d} {l}")
