// Event-driven warp-cooperative TTFS-SNN baseline. Task 1: LeNet on MNIST.
// One warp per output neuron; per-timestep increments in shared memory with
// cooperative prefix-sum early exit.
// Build: nvcc snn_eventdriven_baseline.cu -o snn_eventdriven_baseline -O3
#include <iostream>
#include <vector>
#include <string>
#include <fstream>
#include <algorithm>
#include <cmath>
#include <cuda_runtime.h>
#include <cfloat>

// Global configuration constants
#define TIME_WINDOW 80.0f // total time window
#define VTH_INIT 1.0f // initial threshold
#define INF_TIME 9999.0f // marker for "never fired"

// Macro to check whether a CUDA call succeeded
#define CHECK_CUDA(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            printf("CUDA Error: %s at line %d\n", cudaGetErrorString(err), __LINE__); \
            exit(1); \
        } \
    } while (0)

// Weight structure
struct LayerWeights {
    float* d_kernel; // device-side convolution kernel weights
    float* d_bias; // device-side bias
    float tc_fire; // firing time constant
    float td; // delay time
    int k_h, k_w; // kernel height and width
    int c_in, c_out; // input and output channel counts
};

// CUDA kernel: image encoding, converts pixel values to spike times
__global__ void k_encode_image(const float* img, float* out_spikes, int size, float tc, float td) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x; // thread index
    if (idx >= size) return; // bounds check
    float pixel = img[idx]; // current pixel value
    if (pixel < 1e-5) { // pixel too dark or zero
        out_spikes[idx] = INF_TIME; // mark as never fired
        return;
    }
    float t_float = td - tc * logf(pixel); // TTFS encoding
    if (t_float < 0.0f) t_float = 0.0f; // clamp time to non-negative
    float t_spike = ceilf(t_float); // round up to discrete time step
    if (t_spike > TIME_WINDOW) out_spikes[idx] = INF_TIME; // outside time window
    else out_spikes[idx] = t_spike; // set spike time
}

// CUDA kernel: pooling layer, min-time pooling
__global__ void k_pooling(const float* in_spikes, float* out_spikes, int B, int H_in, int W_in, int C) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x; // thread index
    int H_out = H_in / 2; // output height
    int W_out = W_in / 2; // output width
    int total = B * H_out * W_out * C; // total output size
    if (idx >= total) return; // bounds check
    int b = idx / (H_out * W_out * C);
    int local_idx = idx % (H_out * W_out * C);
    int c = local_idx % C; // current channel
    int w = (local_idx / C) % W_out; // current width
    int h = local_idx / C / W_out; // current height
    int h_start = h * 2; // input start height
    int w_start = w * 2; // input start width
    float min_t = INF_TIME; // initialize minimum time
    // 2x2 max pooling, equivalent to minimum time under TTFS
    for (int i = 0; i < 2; ++i) {
        for (int j = 0; j < 2; ++j) {
            int cur_h = h_start + i;
            int cur_w = w_start + j;
            if (cur_h < H_in && cur_w < W_in) {
                int in_idx = b * (H_in * W_in * C) + (cur_h * W_in + cur_w) * C + c;
                float t = in_spikes[in_idx];
                if (t < min_t) min_t = t; // update minimum time
            }
        }
    }
    out_spikes[idx] = min_t; // set output spike time
}

// CUDA kernel: warp-cooperative layer inference
__global__ void k_layer_inference_optimized(
    const float* input_spikes, // previous layer output spike times
    float* output_spikes, // current layer output spike times
    const float* kernel, // current layer weights
    const float* bias, // current layer bias
    const float* LUT_decay, // weight-decay LUT
    const float* LUT_th, // threshold LUT
    int t_min, // earliest check time
    int B, // batch size
    int H_in, int W_in, int C_in, // input feature map dimensions
    int H_out, int W_out, int C_out, // output feature map dimensions
    int K_h, int K_w, // kernel dimensions
    bool is_fc // whether this is a fully-connected layer
) {
    const int T = 80;
    const int WARP_SIZE = 32;
    int warps_per_block = blockDim.x / WARP_SIZE;
    int neuron_id = blockIdx.x * warps_per_block + (threadIdx.x / WARP_SIZE);
    int num_neurons_per_image = H_out * W_out * C_out;
    int total_neurons = B * num_neurons_per_image;
    if (neuron_id >= total_neurons) return;
    int lane = threadIdx.x % WARP_SIZE;
    int image_id = neuron_id / num_neurons_per_image;
    int local_id = neuron_id % num_neurons_per_image;
    int c_out, w_out = 0, h_out = 0;
    if (is_fc) {
        c_out = local_id;
    } else {
        c_out = local_id % C_out;
        w_out = (local_id / C_out) % W_out;
        h_out = local_id / C_out / W_out;
    }
    float b_val = bias[c_out];
    extern __shared__ float shared_delta[];
    float* my_delta = shared_delta + (threadIdx.x / WARP_SIZE) * T;
    // initialize delta
    for (int i = lane; i < T; i += WARP_SIZE) {
        my_delta[i] = 0.0f;
    }
    __syncwarp();
    // fill delta
    if (lane == 0) {
        if (is_fc) {
            int num_inputs = C_in;
            for (int i = 0; i < num_inputs; ++i) {
                int in_idx = image_id * num_inputs + i;
                float t_in = input_spikes[in_idx];
                if (t_in < TIME_WINDOW) {
                    int t = (int) t_in;
                    float decay = LUT_decay[t];
                    float w = kernel[i * C_out + c_out];
                    my_delta[t] += w * decay;
                }
            }
        } else {
            // convolution layer
            int h_in_start = h_out;
            int w_in_start = w_out;
            for (int kh = 0; kh < K_h; ++kh) {
                for (int kw = 0; kw < K_w; ++kw) {
                    for (int cin = 0; cin < C_in; ++cin) {
                        int h_in = h_in_start + kh;
                        int w_in = w_in_start + kw;
                        if (h_in >= 0 && h_in < H_in && w_in >= 0 && w_in < W_in) {
                            int in_idx = image_id * (H_in * W_in * C_in) + (h_in * W_in + w_in) * C_in + cin;
                            float t_in = input_spikes[in_idx];
                            if (t_in < TIME_WINDOW) {
                                int t = (int) t_in;
                                float decay = LUT_decay[t];
                                int w_idx = ((kh * K_w + kw) * C_in + cin) * C_out + c_out;
                                float w = kernel[w_idx];
                                my_delta[t] += w * decay;
                            }
                        }
                    }
                }
            }
        }
    }
    __syncwarp();
    // compute total_sum
    float total_sum = 0.0f;
    for (int base = 0; base < T; base += WARP_SIZE) {
        float val = (base + lane < T) ? my_delta[base + lane] : 0.0f;
        for (int offset = 16; offset > 0; offset /= 2) {
            val += __shfl_down_sync(0xffffffffu, val, offset);
        }
        if (lane == 0) total_sum += val;
    }
    __syncwarp();
    total_sum = __shfl_sync(0xffffffffu, total_sum, 0); // broadcast to all lanes
    float min_th = LUT_th[T - 1];
    if (b_val + total_sum < min_th) {
        if (lane == 0) output_spikes[neuron_id] = INF_TIME;
        return;
    }
    // cooperative early-exit prefix sum
    float current_base = 0.0f;
    const int chunk_size = WARP_SIZE;
    int num_chunks = (T + chunk_size - 1) / chunk_size;
    for (int ch = 0; ch < num_chunks; ++ch) {
        int t_start = ch * chunk_size;
        float val = (t_start + lane < T) ? my_delta[t_start + lane] : 0.0f;
        float prefix = val;
        for (int d = 1; d < chunk_size; d *= 2) {
            float up = __shfl_up_sync(0xffffffffu, prefix, d);
            if (lane >= d) prefix += up;
        }
        float full_prefix = current_base + prefix;
        float th = (t_start + lane < T) ? LUT_th[t_start + lane] : 0.0f;
        bool pred = (t_start + lane < T) && (t_start + lane >= t_min) && (full_prefix + b_val >= th) && (th >= 1e-5f);
        unsigned int mask = __ballot_sync(0xffffffffu, pred);
        if (mask != 0) {
            int first = __ffs(mask) - 1;
            if (lane == 0) output_spikes[neuron_id] = (float)(t_start + first);
            return;
        }
        float block_sum = __shfl_sync(0xffffffffu, prefix, chunk_size - 1);
        if (lane == 0) current_base += block_sum;
        __syncwarp();
        // check whether firing is impossible
        float remaining_sum = total_sum - current_base;
        if (current_base + b_val + remaining_sum < min_th) {
            if (lane == 0) output_spikes[neuron_id] = INF_TIME;
            return;
        }
    }
    // no spike emitted
    if (lane == 0) output_spikes[neuron_id] = INF_TIME;
}

// Host helper: load weights
void load_weights(const std::string& filename, std::vector<LayerWeights>& layers) {
    std::ifstream f(filename, std::ios::binary);
    if (!f.is_open()) { std::cerr << "File not found: " << filename << "\n"; exit(1); }
    int num_layers;
    f.read((char*)&num_layers, 4); // read layer count
    for(int i=0; i<num_layers; ++i) {
        LayerWeights l;
        int ndim;
        // Kernel
        f.read((char*)&ndim, 4); // read ndim
        std::vector<int> k_shape(ndim);
        int k_size = 1;
        for(int j=0; j<ndim; ++j) {
            f.read((char*)&k_shape[j], 4); // read each dimension size
            k_size *= k_shape[j]; // accumulate total size
        }
        if (ndim == 4) { l.k_h = k_shape[0]; l.k_w = k_shape[1]; l.c_in = k_shape[2]; l.c_out = k_shape[3]; }
        else { l.k_h = 1; l.k_w = 1; l.c_in = k_shape[0]; l.c_out = k_shape[1]; }

        std::vector<float> h_kernel(k_size);
        f.read((char*)h_kernel.data(), k_size * 4); // read weight data
        CHECK_CUDA(cudaMalloc(&l.d_kernel, k_size * 4));
        CHECK_CUDA(cudaMemcpy(l.d_kernel, h_kernel.data(), k_size * 4, cudaMemcpyHostToDevice));
        // Bias
        f.read((char*)&ndim, 4);
        int b_size;
        f.read((char*)&b_size, 4); // read bias size
        std::vector<float> h_bias(b_size);
        f.read((char*)h_bias.data(), b_size * 4); // read bias data
        CHECK_CUDA(cudaMalloc(&l.d_bias, b_size * 4));
        CHECK_CUDA(cudaMemcpy(l.d_bias, h_bias.data(), b_size * 4, cudaMemcpyHostToDevice));
        // TC / TD
        f.read((char*)&ndim, 4); int tc_size; f.read((char*)&tc_size, 4);
        f.read((char*)&l.tc_fire, 4); // read firing time constant
        f.read((char*)&ndim, 4); int td_size; f.read((char*)&td_size, 4);
        f.read((char*)&l.td, 4); // read delay time
        layers.push_back(l); // append layer to vector
    }
}

int main() {
    const int B = 10000;   // full MNIST test set (matches the paper's Task 1)
    const int T = 80;
    const int BLOCK_SIZE = 128;
    const int WARP_SIZE = 32;
    // load network weights
    std::vector<LayerWeights> layers;
    load_weights("exported_models/snn_weights.bin", layers);
    // load all image data
    std::vector<float> all_imgs(B * 784);
    for (int i = 0; i < B; ++i) {
        std::string img_path = "dataset_downloaded/mnist_test10k/" + std::to_string(i) + ".bin";
        std::ifstream img_f(img_path, std::ios::binary);
        if (!img_f.is_open()) {
            std::cerr << "Image file not found: " << img_path << "\n";
            return 1;
        }
        img_f.read((char*)&all_imgs[i * 784], 784 * 4);
    }
    // load all labels
    std::vector<float> all_labels(B * 10);
    std::ifstream lbl_f("dataset_downloaded/mnist_test10k/label_onehot", std::ios::binary);
    if (!lbl_f.is_open()) {
        std::cerr << "Label file not found\n";
        return 1;
    }
    for (int i = 0; i < B; ++i) {
        lbl_f.seekg(i * 10 * 4, std::ios::beg);
        lbl_f.read((char*)&all_labels[i * 10], 10 * 4);
    }
    // allocate device memory and copy image data to device
    float *d_img, *d_s0, *d_s1, *d_s1_p, *d_s2, *d_s2_p, *d_out;
    CHECK_CUDA(cudaMalloc(&d_img, B * 784 * 4));
    CHECK_CUDA(cudaMemcpy(d_img, all_imgs.data(), B * 784 * 4, cudaMemcpyHostToDevice));
    // allocate feature-map device memory
    CHECK_CUDA(cudaMalloc(&d_s0, B * 28 * 28 * 1 * 4));
    CHECK_CUDA(cudaMalloc(&d_s1, B * 24 * 24 * 12 * 4));
    CHECK_CUDA(cudaMalloc(&d_s1_p, B * 12 * 12 * 12 * 4));
    CHECK_CUDA(cudaMalloc(&d_s2, B * 8 * 8 * 64 * 4));
    CHECK_CUDA(cudaMalloc(&d_s2_p, B * 4 * 4 * 64 * 4));
    CHECK_CUDA(cudaMalloc(&d_out, B * 10 * 4));
    // allocate LUT device memory
    float *d_LUT_decay, *d_LUT_th;
    CHECK_CUDA(cudaMalloc(&d_LUT_decay, T * 4));
    CHECK_CUDA(cudaMalloc(&d_LUT_th, T * 4));
    std::cout << ">>> Starting SNN Inference (Optimized)..." << std::endl;
    float tc_in = 17.452274f; // input layer time constant
    float td_in = 0.0f; // input layer delay time
    // image encoding: convert pixel values to spike times
    int encode_threads = 256;
    int encode_blocks = (B * 784 + encode_threads - 1) / encode_threads;
    k_encode_image<<<encode_blocks, encode_threads>>>(d_img, d_s0, B * 784, tc_in, td_in);
    CHECK_CUDA(cudaDeviceSynchronize());
    // prepare LUT for layer 0
    std::vector<float> h_LUT_decay(T);
    std::vector<float> h_LUT_th(T);
    float tc_integ = tc_in;
    float td_integ = td_in;
    float tc_fire = layers[0].tc_fire;
    float td_fire = layers[0].td;
    int t_min = (int)ceilf(td_fire);
    for (int t = 0; t < T; ++t) {
        float rel = (float)t - td_integ;
        h_LUT_decay[t] = expf(-rel / tc_integ);
        float rel_th = (float)t - td_fire;
        h_LUT_th[t] = VTH_INIT * expf(-rel_th / tc_fire);
    }
    CHECK_CUDA(cudaMemcpy(d_LUT_decay, h_LUT_decay.data(), T * 4, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_LUT_th, h_LUT_th.data(), T * 4, cudaMemcpyHostToDevice));
    // convolution layer 1 inference
    int num_neurons_c1 = B * 24 * 24 * 12;
    int warps_per_block_c1 = BLOCK_SIZE / WARP_SIZE;
    int num_blocks_c1 = (num_neurons_c1 + warps_per_block_c1 - 1) / warps_per_block_c1;
    k_layer_inference_optimized<<<num_blocks_c1, BLOCK_SIZE, warps_per_block_c1 * T * sizeof(float)>>>(
        d_s0, d_s1, layers[0].d_kernel, layers[0].d_bias, d_LUT_decay, d_LUT_th, t_min,
        B, 28, 28, 1, 24, 24, 12, 5, 5, false
    );
    CHECK_CUDA(cudaDeviceSynchronize());
    // pooling layer 1 inference
    int threads_p1 = 256;
    int num_neurons_p1 = B * 12 * 12 * 12;
    int blocks_p1 = (num_neurons_p1 + threads_p1 - 1) / threads_p1;
    k_pooling<<<blocks_p1, threads_p1>>>(d_s1, d_s1_p, B, 24, 24, 12);
    CHECK_CUDA(cudaDeviceSynchronize());
    // prepare LUT for layer 1
    tc_integ = layers[0].tc_fire;
    td_integ = layers[0].td;
    tc_fire = layers[1].tc_fire;
    td_fire = layers[1].td;
    t_min = (int)ceilf(td_fire);
    for (int t = 0; t < T; ++t) {
        float rel = (float)t - td_integ;
        h_LUT_decay[t] = expf(-rel / tc_integ);
        float rel_th = (float)t - td_fire;
        h_LUT_th[t] = VTH_INIT * expf(-rel_th / tc_fire);
    }
    CHECK_CUDA(cudaMemcpy(d_LUT_decay, h_LUT_decay.data(), T * 4, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_LUT_th, h_LUT_th.data(), T * 4, cudaMemcpyHostToDevice));
    // convolution layer 2 inference
    int num_neurons_c2 = B * 8 * 8 * 64;
    int warps_per_block_c2 = BLOCK_SIZE / WARP_SIZE;
    int num_blocks_c2 = (num_neurons_c2 + warps_per_block_c2 - 1) / warps_per_block_c2;
    k_layer_inference_optimized<<<num_blocks_c2, BLOCK_SIZE, warps_per_block_c2 * T * sizeof(float)>>>(
        d_s1_p, d_s2, layers[1].d_kernel, layers[1].d_bias, d_LUT_decay, d_LUT_th, t_min,
        B, 12, 12, 12, 8, 8, 64, 5, 5, false
    );
    CHECK_CUDA(cudaDeviceSynchronize());
    // pooling layer 2 inference
    int threads_p2 = 256;
    int num_neurons_p2 = B * 4 * 4 * 64;
    int blocks_p2 = (num_neurons_p2 + threads_p2 - 1) / threads_p2;
    k_pooling<<<blocks_p2, threads_p2>>>(d_s2, d_s2_p, B, 8, 8, 64);
    CHECK_CUDA(cudaDeviceSynchronize());
    // prepare LUT for layer 2 (FC)
    tc_integ = layers[1].tc_fire;
    td_integ = layers[1].td;
    tc_fire = layers[2].tc_fire;
    td_fire = layers[2].td;
    t_min = (int)ceilf(td_fire);
    for (int t = 0; t < T; ++t) {
        float rel = (float)t - td_integ;
        h_LUT_decay[t] = expf(-rel / tc_integ);
        float rel_th = (float)t - td_fire;
        h_LUT_th[t] = VTH_INIT * expf(-rel_th / tc_fire);
    }
    CHECK_CUDA(cudaMemcpy(d_LUT_decay, h_LUT_decay.data(), T * 4, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_LUT_th, h_LUT_th.data(), T * 4, cudaMemcpyHostToDevice));
    // fully-connected layer inference
    int num_neurons_fc = B * 10;
    int warps_per_block_fc = BLOCK_SIZE / WARP_SIZE;
    int num_blocks_fc = (num_neurons_fc + warps_per_block_fc - 1) / warps_per_block_fc;
    k_layer_inference_optimized<<<num_blocks_fc, BLOCK_SIZE, warps_per_block_fc * T * sizeof(float)>>>(
        d_s2_p, d_out, layers[2].d_kernel, layers[2].d_bias, d_LUT_decay, d_LUT_th, t_min,
        B, 1, 1, 1024, 1, 1, 10, 1, 1, true
    );
    CHECK_CUDA(cudaDeviceSynchronize());
    // copy inference results back to host
    std::vector<float> h_out(B * 10);
    CHECK_CUDA(cudaMemcpy(h_out.data(), d_out, B * 10 * 4, cudaMemcpyDeviceToHost));
    // compute accuracy
    int correct = 0;
    for (int bi = 0; bi < B; ++bi) {
        float min_time = INF_TIME;
        int pred_cls = -1;
        for (int i = 0; i < 10; ++i) {
            float t = h_out[bi * 10 + i];
            if (t < min_time) {
                min_time = t;
                pred_cls = i;
            }
        }
        int true_cls = -1;
        for (int i = 0; i < 10; ++i) {
            if (all_labels[bi * 10 + i] > 0.5f) true_cls = i;
        }
        if (pred_cls == true_cls) ++correct;
    }
    float accuracy = static_cast<float>(correct) / B;
    std::cout << "Accuracy on " << B << " images: " << accuracy << std::endl;
    // free memory
    CHECK_CUDA(cudaFree(d_img));
    CHECK_CUDA(cudaFree(d_s0));
    CHECK_CUDA(cudaFree(d_s1));
    CHECK_CUDA(cudaFree(d_s1_p));
    CHECK_CUDA(cudaFree(d_s2));
    CHECK_CUDA(cudaFree(d_s2_p));
    CHECK_CUDA(cudaFree(d_out));
    CHECK_CUDA(cudaFree(d_LUT_decay));
    CHECK_CUDA(cudaFree(d_LUT_th));
    for (auto& l : layers) {
        CHECK_CUDA(cudaFree(l.d_kernel));
        CHECK_CUDA(cudaFree(l.d_bias));
    }
    return 0;
}
