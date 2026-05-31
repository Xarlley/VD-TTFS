// Step-by-step (time-driven) TTFS-SNN baseline. Task 1: LeNet on MNIST. A single
// loop over T timesteps integrates and fires every neuron in the whole network at
// each step; transcendental dynamics are evaluated inline (no look-up table) and
// no algorithmic optimization (no early exit) is applied. Same network/weights/
// data/dynamics as VD-TTFS.
// Build: nvcc snn_mnist_stepbystep.cu -o snn_mnist_stepbystep -O3 -Wno-deprecated-gpu-targets
// Run:   ./snn_mnist_stepbystep
#include <iostream>
#include <vector>
#include <string>
#include <fstream>
#include <cmath>
#include <cuda_runtime.h>

#define TIME_WINDOW 80
#define VTH_INIT 1.0f
#define INF_TIME 9999.0f

#define CHECK_CUDA(call) \
    do { cudaError_t err = call; \
         if (err != cudaSuccess) { printf("CUDA Error: %s at line %d\n", cudaGetErrorString(err), __LINE__); exit(1);} \
    } while (0)

struct LayerWeights {
    float* d_kernel; float* d_bias;
    float tc_fire; float td;
    int k_h, k_w, c_in, c_out;
};

// Emit a binary spike at timestep t for every input pixel whose first-spike time
// (t = ceil(td - tc*log(pixel))) equals t.
__global__ void k_encode_step(const float* img, char* spikes_out, int size, int t, float tc, float td) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= size) return;
    spikes_out[idx] = 0;
    float pixel = img[idx];
    if (pixel < 1e-5f) return;
    float t_float = td - tc * logf(pixel);
    if (t_float < 0.0f) t_float = 0.0f;
    int t_spike = (int)ceilf(t_float);
    if (t == t_spike && t_spike <= TIME_WINDOW) spikes_out[idx] = 1;
}

// One time-driven conv/fc step: integrate the synapses that spike at this timestep
// (each scaled by the afferent weight-decay kernel at t), then fire on the decaying
// threshold. Valid convolution, NHWC layout. The fire-once latch is the TTFS model.
__global__ void k_conv_step(
    const char* __restrict__ spikes_in, char* __restrict__ spikes_out,
    float* __restrict__ vmem, bool* __restrict__ fired,
    const float* __restrict__ kernel, const float* __restrict__ bias,
    int B, int H_in, int W_in, int C_in, int H_out, int W_out, int C_out,
    int K_h, int K_w, bool is_fc, int t, int t_min,
    float tc_integ, float td_integ, float tc_fire, float td_fire)
{
    int nid = blockIdx.x * blockDim.x + threadIdx.x;
    int total = B * H_out * W_out * C_out;
    if (nid >= total) return;
    if (fired[nid]) { spikes_out[nid] = 0; return; }

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

    float kernel_val = expf(-((float)t - td_integ) / tc_integ);
    float psp = 0.0f;
    if (is_fc) {
        const char* src = spikes_in + (size_t)image_id * C_in;
        for (int cin = 0; cin < C_in; ++cin)
            if (src[cin]) psp += kernel[(size_t)cin * C_out + c_out];
    } else {
        for (int kh = 0; kh < K_h; ++kh) {
            int h_in = h_out + kh;
            for (int kw = 0; kw < K_w; ++kw) {
                int w_in = w_out + kw;
                int col = (image_id * H_in + h_in) * W_in + w_in;
                const char* src = spikes_in + (size_t)col * C_in;
                int wbase = (kh * K_w + kw) * C_in;
                for (int cin = 0; cin < C_in; ++cin)
                    if (src[cin]) psp += kernel[(size_t)(wbase + cin) * C_out + c_out];
            }
        }
    }
    vmem[nid] += psp * kernel_val;

    spikes_out[nid] = 0;
    if (t >= t_min) {
        float th = VTH_INIT * expf(-((float)t - td_fire) / tc_fire);
        if (vmem[nid] + bias[c_out] >= th && th >= 1e-5f) {
            spikes_out[nid] = 1;
            fired[nid] = true;
        }
    }
}

// Spiking max-pool: the pooled neuron fires the first timestep any of its 2x2
// inputs spikes (equivalent to the minimum of the input first-spike times).
__global__ void k_pool_step(const char* spikes_in, char* spikes_out, bool* fired,
                            int B, int H_in, int W_in, int C) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int H_out = H_in / 2, W_out = W_in / 2;
    if (idx >= B * H_out * W_out * C) return;
    if (fired[idx]) { spikes_out[idx] = 0; return; }

    int b = idx / (H_out * W_out * C);
    int local_idx = idx % (H_out * W_out * C);
    int c = local_idx % C, w = (local_idx / C) % W_out, h = local_idx / C / W_out;

    spikes_out[idx] = 0;
    for (int i = 0; i < 2; ++i)
        for (int j = 0; j < 2; ++j) {
            int ch = h * 2 + i, cw = w * 2 + j;
            if (ch < H_in && cw < W_in && spikes_in[b * (H_in * W_in * C) + (ch * W_in + cw) * C + c]) {
                spikes_out[idx] = 1;
                fired[idx] = true;
                return;
            }
        }
}

// Record the first-spike timestep of an output neuron (argmin over classes = prediction).
__global__ void k_record_first_spike(const char* spikes, float* first_time, int size, int t) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= size) return;
    if (spikes[idx] && first_time[idx] >= INF_TIME) first_time[idx] = (float)t;
}

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

int main() {
    const int B = 10000;                                   // full MNIST test set (matches the paper's Task 1)
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
    for (int i = 0; i < B; ++i) { lbl_f.seekg(i * 10 * 4, std::ios::beg); lbl_f.read((char*)&all_labels[i * 10], 10 * 4); }

    // network state: binary spike maps, membrane potentials, and fire-once latches
    char *s0, *s1, *s1_p, *s2, *s2_p, *s_out;
    float *v1, *v2, *v_out, *d_img, *d_out_time;
    bool *f1, *fp1, *f2, *fp2, *f_out;
    CHECK_CUDA(cudaMalloc(&d_img, B * 784 * 4));
    CHECK_CUDA(cudaMalloc(&s0,   (size_t)B * 28 * 28 * 1));
    CHECK_CUDA(cudaMalloc(&s1,   (size_t)B * 24 * 24 * 12));
    CHECK_CUDA(cudaMalloc(&s1_p, (size_t)B * 12 * 12 * 12));
    CHECK_CUDA(cudaMalloc(&s2,   (size_t)B * 8 * 8 * 64));
    CHECK_CUDA(cudaMalloc(&s2_p, (size_t)B * 4 * 4 * 64));
    CHECK_CUDA(cudaMalloc(&s_out,(size_t)B * 10));
    CHECK_CUDA(cudaMalloc(&v1,   (size_t)B * 24 * 24 * 12 * 4));
    CHECK_CUDA(cudaMalloc(&v2,   (size_t)B * 8 * 8 * 64 * 4));
    CHECK_CUDA(cudaMalloc(&v_out,(size_t)B * 10 * 4));
    CHECK_CUDA(cudaMalloc(&f1,   (size_t)B * 24 * 24 * 12));
    CHECK_CUDA(cudaMalloc(&fp1,  (size_t)B * 12 * 12 * 12));
    CHECK_CUDA(cudaMalloc(&f2,   (size_t)B * 8 * 8 * 64));
    CHECK_CUDA(cudaMalloc(&fp2,  (size_t)B * 4 * 4 * 64));
    CHECK_CUDA(cudaMalloc(&f_out,(size_t)B * 10));
    CHECK_CUDA(cudaMalloc(&d_out_time, (size_t)B * 10 * 4));

    CHECK_CUDA(cudaMemset(v1, 0, (size_t)B * 24 * 24 * 12 * 4));
    CHECK_CUDA(cudaMemset(v2, 0, (size_t)B * 8 * 8 * 64 * 4));
    CHECK_CUDA(cudaMemset(v_out, 0, (size_t)B * 10 * 4));
    CHECK_CUDA(cudaMemset(f1, 0, (size_t)B * 24 * 24 * 12));
    CHECK_CUDA(cudaMemset(fp1, 0, (size_t)B * 12 * 12 * 12));
    CHECK_CUDA(cudaMemset(f2, 0, (size_t)B * 8 * 8 * 64));
    CHECK_CUDA(cudaMemset(fp2, 0, (size_t)B * 4 * 4 * 64));
    CHECK_CUDA(cudaMemset(f_out, 0, (size_t)B * 10));
    {
        std::vector<float> init(B * 10, INF_TIME);
        CHECK_CUDA(cudaMemcpy(d_out_time, init.data(), (size_t)B * 10 * 4, cudaMemcpyHostToDevice));
    }
    CHECK_CUDA(cudaMemcpy(d_img, all_imgs.data(), B * 784 * 4, cudaMemcpyHostToDevice));

    const float tc_in = 17.452274f, td_in = 0.0f;
    // per-layer fire constants drive both this layer's threshold and the next layer's decay kernel
    float tc0 = layers[0].tc_fire, td0 = layers[0].td;
    float tc1 = layers[1].tc_fire, td1 = layers[1].td;
    float tc2 = layers[2].tc_fire, td2 = layers[2].td;
    int t_min0 = (int)ceilf(td0); if (t_min0 < 0) t_min0 = 0;
    int t_min1 = (int)ceilf(td1); if (t_min1 < 0) t_min1 = 0;
    int t_min2 = (int)ceilf(td2); if (t_min2 < 0) t_min2 = 0;

    auto blk = [](int n) { return (n + 255) / 256; };
    int n1 = B * 24 * 24 * 12, np1 = B * 12 * 12 * 12, n2 = B * 8 * 8 * 64, np2 = B * 4 * 4 * 64, no = B * 10;

    std::cout << ">>> Running step-by-step (time-driven) inference..." << std::endl;
    cudaEvent_t start, stop; cudaEventCreate(&start); cudaEventCreate(&stop);
    cudaEventRecord(start);

    for (int t = 0; t < TIME_WINDOW; ++t) {
        k_encode_step<<<blk(B * 784), 256>>>(d_img, s0, B * 784, t, tc_in, td_in);

        k_conv_step<<<blk(n1), 256>>>(s0, s1, v1, f1, layers[0].d_kernel, layers[0].d_bias,
            B, 28, 28, 1, 24, 24, 12, 5, 5, false, t, t_min0, tc_in, td_in, tc0, td0);
        k_pool_step<<<blk(np1), 256>>>(s1, s1_p, fp1, B, 24, 24, 12);

        k_conv_step<<<blk(n2), 256>>>(s1_p, s2, v2, f2, layers[1].d_kernel, layers[1].d_bias,
            B, 12, 12, 12, 8, 8, 64, 5, 5, false, t, t_min1, tc0, td0, tc1, td1);
        k_pool_step<<<blk(np2), 256>>>(s2, s2_p, fp2, B, 8, 8, 64);

        k_conv_step<<<blk(no), 256>>>(s2_p, s_out, v_out, f_out, layers[2].d_kernel, layers[2].d_bias,
            B, 1, 1, 1024, 1, 1, 10, 1, 1, true, t, t_min2, tc1, td1, tc2, td2);
        k_record_first_spike<<<blk(no), 256>>>(s_out, d_out_time, no, t);
    }

    cudaEventRecord(stop); cudaEventSynchronize(stop);
    float ms = 0; cudaEventElapsedTime(&ms, start, stop);

    std::vector<float> h_out(B * 10);
    CHECK_CUDA(cudaMemcpy(h_out.data(), d_out_time, B * 10 * 4, cudaMemcpyDeviceToHost));
    int correct = 0;
    for (int bi = 0; bi < B; ++bi) {
        float min_t = INF_TIME; int pred = -1;
        for (int i = 0; i < 10; ++i) { float tt = h_out[bi * 10 + i]; if (tt < min_t) { min_t = tt; pred = i; } }
        int truth = -1;
        for (int i = 0; i < 10; ++i) if (all_labels[bi * 10 + i] > 0.5f) truth = i;
        if (pred == truth) ++correct;
    }
    float acc = (float)correct / B * 100.0f;

    std::cout << "\n=======================================" << std::endl;
    std::cout << "STEP-BY-STEP (LeNet-MNIST) COMPLETED" << std::endl;
    std::cout << "Images Processed:     " << B << std::endl;
    std::cout << "Final Accuracy:       " << acc << "%" << std::endl;
    std::cout << "Total Inference Time: " << ms / 1000.0f << " seconds" << std::endl;
    std::cout << "Throughput:           " << (B / (ms / 1000.0f)) << " images/sec" << std::endl;
    std::cout << "=======================================" << std::endl;
    return 0;
}
