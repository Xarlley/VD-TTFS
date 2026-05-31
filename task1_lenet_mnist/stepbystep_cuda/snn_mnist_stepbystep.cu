// ============================================================================
//  Step-by-Step (time-driven) TTFS-SNN baseline  —  Task 1: LeNet on MNIST
//
//  This is the naive discrete-time-step reference that VD-TTFS is compared
//  against. It uses the SAME network, weights, data, and TTFS dynamics as
//  vdttfs_cuda/snn_timesorted_earlyexit.cu, but performs NO arrival-ordered
//  bucketize and NO early exit: every output neuron materializes the full
//  per-timestep increment array delta[T] and then integrates over the ENTIRE
//  window [0,T), so the work scales with T regardless of when (or whether) the
//  neuron fires. The full-window delta[T] (T=80) lives in (slow) local memory;
//  removing exactly this traversal is what the time-sorted chunked integrator
//  of VD-TTFS does.
//
//  Network (LeNet-MNIST):
//    encode(28x28x1) -> Conv1 5x5 1->12 (valid) 24x24x12 -> Pool 12x12x12
//                    -> Conv2 5x5 12->64 (valid) 8x8x64   -> Pool 4x4x64 (=1024)
//                    -> FC 1024->10   ; prediction = argmin spike time.
//  Dynamics (identical to the VD-TTFS file):
//    decay  : LUT_decay[t] = exp(-(t - td_integ)/tc_integ)
//    thresh : LUT_th[t]    = V0 * exp(-(t - td_fire)/tc_fire)
//    fire   : v_mem + bias >= LUT_th[t], for t in [ceil(td_fire), T).
//
//  Build (in this directory):
//    nvcc snn_mnist_stepbystep.cu -o snn_mnist_stepbystep -O3 -Wno-deprecated-gpu-targets
//  Run (from a directory containing exported_models/ and dataset_downloaded/):
//    ./snn_mnist_stepbystep
// ============================================================================
#include <iostream>
#include <vector>
#include <string>
#include <fstream>
#include <cmath>
#include <cuda_runtime.h>

#define TIME_WINDOW 80
#define VTH_INIT 1.0f
#define INF_TIME 9999.0f
#define TIME_FIRE_START 0.0f

#define CHECK_CUDA(call) \
    do { cudaError_t err = call; \
         if (err != cudaSuccess) { printf("CUDA Error: %s at line %d\n", cudaGetErrorString(err), __LINE__); exit(1);} \
    } while (0)

struct LayerWeights {
    float* d_kernel; float* d_bias;
    float tc_fire; float td;
    int k_h, k_w, c_in, c_out;
};

__global__ void k_encode_image(const float* img, float* out_spikes, int size, float tc, float td) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= size) return;
    float pixel = img[idx];
    if (pixel < 1e-5) { out_spikes[idx] = INF_TIME; return; }
    float t_float = td - tc * logf(pixel);
    if (t_float < 0.0f) t_float = 0.0f;
    float t_spike = ceilf(t_float);
    if (t_spike > (float)TIME_WINDOW) out_spikes[idx] = INF_TIME;
    else out_spikes[idx] = t_spike;
}

// ---- naive time-driven conv/fc: full-window integration, NO early exit ----
__global__ void k_conv_stepbystep(
    const float* __restrict__ in, float* __restrict__ output_spikes,
    const float* __restrict__ kernel, const float* __restrict__ bias,
    const float* __restrict__ LUT_decay, const float* __restrict__ LUT_th,
    int t_min, int B,
    int H_in, int W_in, int C_in,
    int H_out, int W_out, int C_out,
    int K_h, int K_w, bool is_fc
) {
    const int T = TIME_WINDOW;
    int nid = blockIdx.x * blockDim.x + threadIdx.x;
    int total = B * H_out * W_out * C_out;
    if (nid >= total) return;

    int c_out, image_id, h_out = 0, w_out = 0;
    if (is_fc) {
        c_out = nid % C_out;
        image_id = nid / C_out;
    } else {
        c_out = nid % C_out;
        int t1 = nid / C_out;
        w_out = t1 % W_out;
        int t2 = t1 / W_out;
        h_out = t2 % H_out;
        image_id = t2 / H_out;
    }

    // full per-timestep increment array (in local memory) -- the bottleneck the
    // chunked VD-TTFS integrator avoids.
    float delta[TIME_WINDOW];
    #pragma unroll
    for (int i = 0; i < TIME_WINDOW; ++i) delta[i] = 0.0f;

    // one pass over the afferents: bin each spike's LUT-modulated contribution
    // by its (integer) arrival timestep.
    if (is_fc) {
        const float* src = in + (size_t)image_id * C_in;
        for (int cin = 0; cin < C_in; ++cin) {
            float t_in = src[cin];
            if (t_in < INF_TIME) {
                int tb = (int)floorf(t_in - TIME_FIRE_START); if (tb < 0) tb = 0;
                if (tb < T) delta[tb] += kernel[(size_t)cin * C_out + c_out] * LUT_decay[tb];
            }
        }
    } else {
        for (int kh = 0; kh < K_h; ++kh) {
            int h_in = h_out + kh;
            for (int kw = 0; kw < K_w; ++kw) {
                int w_in = w_out + kw;
                int col = (image_id * H_in + h_in) * W_in + w_in;
                const float* src = in + (size_t)col * C_in;
                int wbase = (kh * K_w + kw) * C_in;
                for (int cin = 0; cin < C_in; ++cin) {
                    float t_in = src[cin];
                    if (t_in < INF_TIME) {
                        int tb = (int)floorf(t_in - TIME_FIRE_START); if (tb < 0) tb = 0;
                        if (tb < T) delta[tb] += kernel[(size_t)(wbase + cin) * C_out + c_out] * LUT_decay[tb];
                    }
                }
            }
        }
    }

    // integrate over the WHOLE window, step by step; no early exit.
    float carry = bias[c_out];
    float result = INF_TIME;
    for (int t = 0; t < T; ++t) {
        carry += delta[t];
        if (result >= INF_TIME) {
            float th = LUT_th[t];
            if (t >= t_min && carry >= th && th >= 1e-5f) result = (float)t;
        }
    }
    output_spikes[nid] = result;
}

__global__ void k_pooling(const float* in_spikes, float* out_spikes, int B, int H_in, int W_in, int C) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int H_out = H_in / 2, W_out = W_in / 2;
    if (idx >= B * H_out * W_out * C) return;
    int b = idx / (H_out * W_out * C);
    int local_idx = idx % (H_out * W_out * C);
    int c = local_idx % C, w = (local_idx / C) % W_out, h = local_idx / C / W_out;
    float min_t = INF_TIME;
    for (int i = 0; i < 2; ++i)
        for (int j = 0; j < 2; ++j) {
            int ch = h*2+i, cw = w*2+j;
            if (ch < H_in && cw < W_in) {
                float t = in_spikes[b*(H_in*W_in*C) + (ch*W_in+cw)*C + c];
                if (t < min_t) min_t = t;
            }
        }
    out_spikes[idx] = min_t;
}

// ============================== host helpers ================================

void load_weights(const std::string& filename, std::vector<LayerWeights>& layers) {
    std::ifstream f(filename, std::ios::binary);
    if (!f.is_open()) { std::cerr << "File not found: " << filename << "\n"; exit(1); }
    int num_layers; f.read((char*)&num_layers, 4);
    for (int i = 0; i < num_layers; ++i) {
        LayerWeights l; int ndim;
        f.read((char*)&ndim, 4); std::vector<int> k_shape(ndim); int k_size = 1;
        for (int j = 0; j < ndim; ++j) { f.read((char*)&k_shape[j], 4); k_size *= k_shape[j]; }
        if (ndim == 4) { l.k_h = k_shape[0]; l.k_w = k_shape[1]; l.c_in = k_shape[2]; l.c_out = k_shape[3]; }
        else          { l.k_h = 1; l.k_w = 1; l.c_in = k_shape[0]; l.c_out = k_shape[1]; }
        std::vector<float> h_kernel(k_size); f.read((char*)h_kernel.data(), k_size * 4);
        CHECK_CUDA(cudaMalloc(&l.d_kernel, k_size * 4));
        CHECK_CUDA(cudaMemcpy(l.d_kernel, h_kernel.data(), k_size * 4, cudaMemcpyHostToDevice));
        f.read((char*)&ndim, 4); int b_size; f.read((char*)&b_size, 4);
        std::vector<float> h_bias(b_size); f.read((char*)h_bias.data(), b_size * 4);
        CHECK_CUDA(cudaMalloc(&l.d_bias, b_size * 4));
        CHECK_CUDA(cudaMemcpy(l.d_bias, h_bias.data(), b_size * 4, cudaMemcpyHostToDevice));
        f.read((char*)&ndim, 4); int tc_size; f.read((char*)&tc_size, 4); f.read((char*)&l.tc_fire, 4);
        f.read((char*)&ndim, 4); int td_size; f.read((char*)&td_size, 4); f.read((char*)&l.td, 4);
        layers.push_back(l);
    }
}

void update_LUTs(float* d_LUT_decay, float* d_LUT_th, float tc_integ, float td_integ, float tc_fire, float td_fire) {
    std::vector<float> h_LUT_decay(TIME_WINDOW), h_LUT_th(TIME_WINDOW);
    for (int t = 0; t < TIME_WINDOW; ++t) {
        h_LUT_decay[t] = expf(-((float)t - td_integ) / tc_integ);
        h_LUT_th[t]    = VTH_INIT * expf(-((float)t - td_fire) / tc_fire);
    }
    cudaMemcpy(d_LUT_decay, h_LUT_decay.data(), TIME_WINDOW * 4, cudaMemcpyHostToDevice);
    cudaMemcpy(d_LUT_th, h_LUT_th.data(), TIME_WINDOW * 4, cudaMemcpyHostToDevice);
}

void launch_layer(float* d_in, float* d_out, int H_in, int W_in, int C_in, int H_out, int W_out, int C_out,
                  int layer_idx, float* d_LUT_decay, float* d_LUT_th, const std::vector<LayerWeights>& layers,
                  int B, bool is_fc) {
    int t_min = (int)ceilf(layers[layer_idx].td); if (t_min < 0) t_min = 0;
    int n = B * H_out * W_out * C_out, th = 256, blk = (n + th - 1) / th;
    k_conv_stepbystep<<<blk, th>>>(d_in, d_out,
        layers[layer_idx].d_kernel, layers[layer_idx].d_bias, d_LUT_decay, d_LUT_th, t_min,
        B, H_in, W_in, C_in, H_out, W_out, C_out, layers[layer_idx].k_h, layers[layer_idx].k_w, is_fc);
    CHECK_CUDA(cudaDeviceSynchronize());
}

void launch_pool(float* d_in, float* d_out, int H_in, int W_in, int C, int B) {
    int n = B * (H_in/2) * (W_in/2) * C, threads = 256, blocks = (n + threads - 1) / threads;
    k_pooling<<<blocks, threads>>>(d_in, d_out, B, H_in, W_in, C);
    CHECK_CUDA(cudaDeviceSynchronize());
}

int main() {
    const int B = 10000;                                   // full MNIST test set
    const std::string DATA_DIR = "dataset_downloaded/mnist_test10k/";

    std::cout << ">>> Loading LeNet-MNIST weights..." << std::endl;
    std::vector<LayerWeights> layers;
    load_weights("exported_models/snn_weights.bin", layers);

    std::cout << ">>> Loading " << B << " MNIST test images..." << std::endl;
    std::vector<float> all_imgs(B * 784);
    for (int i = 0; i < B; ++i) {
        std::ifstream img_f(DATA_DIR + std::to_string(i) + ".bin", std::ios::binary);
        if (!img_f.is_open()) { std::cerr << "Cannot open image " << i << ".bin" << std::endl; return 1; }
        img_f.read((char*)&all_imgs[i * 784], 784 * 4);
    }
    std::vector<float> all_labels(B * 10);
    std::ifstream lbl_f(DATA_DIR + "label_onehot", std::ios::binary);
    for (int i = 0; i < B; ++i) { lbl_f.seekg(i*10*4, std::ios::beg); lbl_f.read((char*)&all_labels[i*10], 10*4); }

    float *d_img, *d_s0, *d_s1, *d_s1_p, *d_s2, *d_s2_p, *d_out;
    CHECK_CUDA(cudaMalloc(&d_img, B*784*4));
    CHECK_CUDA(cudaMalloc(&d_s0, B*28*28*1*4));
    CHECK_CUDA(cudaMalloc(&d_s1, B*24*24*12*4));
    CHECK_CUDA(cudaMalloc(&d_s1_p, B*12*12*12*4));
    CHECK_CUDA(cudaMalloc(&d_s2, B*8*8*64*4));
    CHECK_CUDA(cudaMalloc(&d_s2_p, B*4*4*64*4));
    CHECK_CUDA(cudaMalloc(&d_out, B*10*4));

    float *d_LUT_decay, *d_LUT_th;
    CHECK_CUDA(cudaMalloc(&d_LUT_decay, TIME_WINDOW*4));
    CHECK_CUDA(cudaMalloc(&d_LUT_th, TIME_WINDOW*4));

    CHECK_CUDA(cudaMemcpy(d_img, all_imgs.data(), B*784*4, cudaMemcpyHostToDevice));

    float tc_in = 17.452274f, td_in = 0.0f;
    std::cout << ">>> Running step-by-step (time-driven) inference..." << std::endl;

    cudaEvent_t start, stop; cudaEventCreate(&start); cudaEventCreate(&stop);
    cudaEventRecord(start);

    int et = 256;
    k_encode_image<<<(B*784+et-1)/et, et>>>(d_img, d_s0, B*784, tc_in, td_in);
    CHECK_CUDA(cudaDeviceSynchronize());

    update_LUTs(d_LUT_decay, d_LUT_th, tc_in, td_in, layers[0].tc_fire, layers[0].td);
    launch_layer(d_s0, d_s1, 28,28,1, 24,24,12, 0, d_LUT_decay, d_LUT_th, layers, B, false);
    launch_pool(d_s1, d_s1_p, 24,24,12, B);

    update_LUTs(d_LUT_decay, d_LUT_th, layers[0].tc_fire, layers[0].td, layers[1].tc_fire, layers[1].td);
    launch_layer(d_s1_p, d_s2, 12,12,12, 8,8,64, 1, d_LUT_decay, d_LUT_th, layers, B, false);
    launch_pool(d_s2, d_s2_p, 8,8,64, B);

    update_LUTs(d_LUT_decay, d_LUT_th, layers[1].tc_fire, layers[1].td, layers[2].tc_fire, layers[2].td);
    launch_layer(d_s2_p, d_out, 1,1,1024, 1,1,10, 2, d_LUT_decay, d_LUT_th, layers, B, true);

    cudaEventRecord(stop); cudaEventSynchronize(stop);
    float ms = 0; cudaEventElapsedTime(&ms, start, stop);

    std::vector<float> h_out(B * 10);
    CHECK_CUDA(cudaMemcpy(h_out.data(), d_out, B*10*4, cudaMemcpyDeviceToHost));
    int correct = 0;
    for (int bi = 0; bi < B; ++bi) {
        float min_t = INF_TIME; int pred = -1;
        for (int i = 0; i < 10; ++i) { float t = h_out[bi*10+i]; if (t < min_t) { min_t = t; pred = i; } }
        int truth = -1;
        for (int i = 0; i < 10; ++i) if (all_labels[bi*10+i] > 0.5f) truth = i;
        if (pred == truth) ++correct;
    }
    float acc = (float)correct / B * 100.0f;

    std::cout << "\n=======================================" << std::endl;
    std::cout << "STEP-BY-STEP (LeNet-MNIST) COMPLETED" << std::endl;
    std::cout << "Images Processed:     " << B << std::endl;
    std::cout << "Final Accuracy:       " << acc << "%" << std::endl;
    std::cout << "Total Inference Time: " << ms/1000.0f << " seconds" << std::endl;
    std::cout << "Throughput:           " << (B/(ms/1000.0f)) << " images/sec" << std::endl;
    std::cout << "=======================================" << std::endl;
    return 0;
}
