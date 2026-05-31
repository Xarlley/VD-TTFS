// Step-by-step (time-driven) SNN baseline. Task 3: 7-layer recurrent CuLIF SNN
// on DVSGesture. A single loop over T timesteps integrates and fires every neuron
// in the whole network each step; no algorithmic optimization (no early exit).
// Build: nvcc snn_dvs_stepbystep.cu -o snn_dvs_stepbystep -O3 -Wno-deprecated-gpu-targets
#include <iostream>
#include <fstream>
#include <vector>
#include <cmath>
#include <chrono>
#include <iomanip>
#include <cuda_runtime.h>

#define CHECK_CUDA(call) { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        std::cerr << "CUDA Error: " << cudaGetErrorString(err) << " at line " << __LINE__ << std::endl; \
        exit(EXIT_FAILURE); \
    } \
}

#define GET_BLOCKS(total) ((total + 255) / 256)

const int NUM_SAMPLES = 1078;
const int T_TOTAL = 160;
const int NUM_CLASSES = 10;

const float decay_syn_conv = exp(-1.0f / 5.0f);
const float decay_syn_fc = exp(-1.0f / 60.0f);

const float v_th_conv = 5.0f;
const float v_th_fc = 10.0f;

__global__ void extract_input_step(const float* all_data, float* step_input, int N, int C, int H, int W, int T_total, int t) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_pixels = N * C * H * W;
    if (idx < total_pixels) {
        step_input[idx] = all_data[idx * T_total + t];
    }
}

__global__ void conv2d_kernel(const float* input, const float* weight, float* output,
                              int N, int C_in, int C_out, int H, int W, int K) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = N * C_out * H * W;
    if (idx < total) {
        int n = idx / (C_out * H * W);
        int x = idx % W;
        int y = (idx / W) % H;
        int c_out = (idx / (W * H)) % C_out;
        int pad = K / 2;
        float sum = 0.0f;

        for (int c_in = 0; c_in < C_in; ++c_in) {
            for (int ky = 0; ky < K; ++ky) {
                for (int kx = 0; kx < K; ++kx) {
                    int in_y = y + ky - pad;
                    int in_x = x + kx - pad;
                    if (in_y >= 0 && in_y < H && in_x >= 0 && in_x < W) {
                        int in_idx = ((n * C_in + c_in) * H + in_y) * W + in_x;
                        int w_idx = ((c_out * C_in + c_in) * K + ky) * K + kx;
                        sum += input[in_idx] * weight[w_idx];
                    }
                }
            }
        }
        output[idx] = sum;
    }
}

__global__ void maxpool2d_kernel(const float* input, float* output,
                                 int N, int C, int H_in, int W_in) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int H_out = H_in / 2;
    int W_out = W_in / 2;
    int total = N * C * H_out * W_out;

    if (idx < total) {
        int x_out = idx % W_out;
        int y_out = (idx / W_out) % H_out;
        int c = (idx / (W_out * H_out)) % C;
        int n = idx / (C * H_out * W_out);

        int x_in = x_out * 2;
        int y_in = y_out * 2;
        float max_val = -1e9f;
        for (int dy = 0; dy < 2; ++dy) {
            for (int dx = 0; dx < 2; ++dx) {
                int in_idx = ((n * C + c) * H_in + (y_in + dy)) * W_in + (x_in + dx);
                max_val = fmaxf(max_val, input[in_idx]);
            }
        }
        output[idx] = max_val;
    }
}

__global__ void fc_kernel(const float* input, const float* weight, float* output,
                          int N, int in_features, int out_features) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = N * out_features;
    if (idx < total) {
        int row = idx / out_features;
        int col = idx % out_features;
        float sum = 0.0f;
        for (int i = 0; i < in_features; ++i) {
            sum += input[row * in_features + i] * weight[col * in_features + i];
        }
        output[idx] = sum;
    }
}

__global__ void add_tensors_kernel(float* a, const float* b, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) a[idx] += b[idx];
}

// CuIF dynamics integrated and fired for every neuron at every timestep.
// The fire-once latch (has_fired) is the TTFS neuron model, not an optimization:
// a neuron emits a single spike and is silent thereafter.
__global__ void cuif_step_kernel(
    const float* input, float* v, float* I, float* spikes, bool* has_fired,
    float decay_syn, float v_th, int total_neurons, int t, float* first_times)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < total_neurons) {
        float curr_I = I[idx] * decay_syn + input[idx];
        float curr_v = v[idx] + curr_I;
        I[idx] = curr_I;
        v[idx] = curr_v;

        if (!has_fired[idx] && curr_v >= v_th) {
            spikes[idx] = 1.0f;
            has_fired[idx] = true;
            if (first_times != nullptr && first_times[idx] > 9000.0f) {
                first_times[idx] = (float)t;
            }
        } else {
            spikes[idx] = 0.0f;
        }
    }
}

void load_binary(const char* filepath, void* host_ptr, size_t bytes) {
    std::ifstream file(filepath, std::ios::binary);
    if (!file) { std::cerr << "Failed to open " << filepath << std::endl; exit(1); }
    file.read(reinterpret_cast<char*>(host_ptr), bytes);
    file.close();
}

int main() {
    std::cout << ">>> Step-by-step SNN inference (DVSGesture, full-window integration)..." << std::endl;
    int batch_size = NUM_SAMPLES;

    size_t w_bytes = 1292032 * sizeof(float);
    float* h_weights = new float[1292032];
    load_binary("cuda_assets/dvsgesture_weights.bin", h_weights, w_bytes);
    float* d_weights;
    CHECK_CUDA(cudaMalloc(&d_weights, w_bytes));
    CHECK_CUDA(cudaMemcpy(d_weights, h_weights, w_bytes, cudaMemcpyHostToDevice));

    float *w_conv1 = d_weights + 0, *w_conv2 = d_weights + 1152, *w_conv3 = d_weights + 74880;
    float *w_fc1_rec = d_weights + 222336, *w_fc1 = d_weights + 238720, *w_fc2 = d_weights + 1287296;

    size_t data_bytes = batch_size * 2 * 32 * 32 * T_TOTAL * sizeof(float);
    float* h_data = new float[batch_size * 2 * 32 * 32 * T_TOTAL];
    load_binary("cuda_assets/train_data.bin", h_data, data_bytes);
    float* d_data;
    CHECK_CUDA(cudaMalloc(&d_data, data_bytes));
    CHECK_CUDA(cudaMemcpy(d_data, h_data, data_bytes, cudaMemcpyHostToDevice));

    int* h_labels = new int[batch_size * NUM_CLASSES];
    load_binary("cuda_assets/train_labels.bin", h_labels, batch_size * NUM_CLASSES * sizeof(int));

    int n_c1 = batch_size * 64 * 32 * 32, n_p1 = batch_size * 128 * 16 * 16;
    int n_c2 = batch_size * 128 * 32 * 32, n_p2 = batch_size * 128 * 8 * 8;
    int n_c3 = batch_size * 128 * 16 * 16, n_fc1 = batch_size * 128, n_fc2 = batch_size * 10;

    float *d_step_in;
    CHECK_CUDA(cudaMalloc(&d_step_in, batch_size * 2 * 32 * 32 * sizeof(float)));

    auto allocate_layer = [&](int num_neurons, float*& out, float*& v, float*& i, float*& s, bool*& hf) {
        CHECK_CUDA(cudaMalloc(&out, num_neurons * sizeof(float)));
        CHECK_CUDA(cudaMalloc(&v, num_neurons * sizeof(float)));
        CHECK_CUDA(cudaMalloc(&i, num_neurons * sizeof(float)));
        CHECK_CUDA(cudaMalloc(&s, num_neurons * sizeof(float)));
        CHECK_CUDA(cudaMalloc(&hf, num_neurons * sizeof(bool)));
        CHECK_CUDA(cudaMemset(v, 0, num_neurons * sizeof(float)));
        CHECK_CUDA(cudaMemset(i, 0, num_neurons * sizeof(float)));
        CHECK_CUDA(cudaMemset(s, 0, num_neurons * sizeof(float)));
        CHECK_CUDA(cudaMemset(hf, 0, num_neurons * sizeof(bool)));
    };

    float *d_c1_out, *d_c1_v, *d_c1_i, *d_c1_s; bool *hf_c1;
    float *d_c2_out, *d_c2_v, *d_c2_i, *d_c2_s; bool *hf_c2;
    float *d_p1_out, *d_p1_v, *d_p1_i, *d_p1_s; bool *hf_p1;
    float *d_c3_out, *d_c3_v, *d_c3_i, *d_c3_s; bool *hf_c3;
    float *d_p2_out, *d_p2_v, *d_p2_i, *d_p2_s; bool *hf_p2;
    float *d_fc1_out, *d_fc1_rec_out, *d_fc1_v, *d_fc1_i, *d_fc1_s; bool *hf_fc1;
    float *d_fc2_out, *d_fc2_v, *d_fc2_i, *d_fc2_s; bool *hf_fc2;

    allocate_layer(n_c1, d_c1_out, d_c1_v, d_c1_i, d_c1_s, hf_c1);
    allocate_layer(n_c2, d_c2_out, d_c2_v, d_c2_i, d_c2_s, hf_c2);
    allocate_layer(n_p1, d_p1_out, d_p1_v, d_p1_i, d_p1_s, hf_p1);
    allocate_layer(n_c3, d_c3_out, d_c3_v, d_c3_i, d_c3_s, hf_c3);
    allocate_layer(n_p2, d_p2_out, d_p2_v, d_p2_i, d_p2_s, hf_p2);
    allocate_layer(n_fc1, d_fc1_out, d_fc1_v, d_fc1_i, d_fc1_s, hf_fc1);
    CHECK_CUDA(cudaMalloc(&d_fc1_rec_out, n_fc1 * sizeof(float)));
    allocate_layer(n_fc2, d_fc2_out, d_fc2_v, d_fc2_i, d_fc2_s, hf_fc2);

    float *d_first_times;
    CHECK_CUDA(cudaMalloc(&d_first_times, n_fc2 * sizeof(float)));
    float* h_first_times_init = new float[n_fc2];
    std::fill_n(h_first_times_init, n_fc2, 9999.0f);
    CHECK_CUDA(cudaMemcpy(d_first_times, h_first_times_init, n_fc2 * sizeof(float), cudaMemcpyHostToDevice));
    delete[] h_first_times_init;

    cudaEvent_t start_evt, stop_evt;
    cudaEventCreate(&start_evt); cudaEventCreate(&stop_evt);
    cudaEventRecord(start_evt);

    for (int t = 0; t < T_TOTAL; ++t) {
        extract_input_step<<<GET_BLOCKS(batch_size * 2 * 32 * 32), 256>>>(d_data, d_step_in, batch_size, 2, 32, 32, T_TOTAL, t);

        // --- Layer 1: Conv ---
        conv2d_kernel<<<GET_BLOCKS(n_c1), 256>>>(d_step_in, w_conv1, d_c1_out, batch_size, 2, 64, 32, 32, 3);
        cuif_step_kernel<<<GET_BLOCKS(n_c1), 256>>>(d_c1_out, d_c1_v, d_c1_i, d_c1_s, hf_c1, decay_syn_conv, v_th_conv, n_c1, t, nullptr);

        // --- Layer 2: Conv ---
        conv2d_kernel<<<GET_BLOCKS(n_c2), 256>>>(d_c1_s, w_conv2, d_c2_out, batch_size, 64, 128, 32, 32, 3);
        cuif_step_kernel<<<GET_BLOCKS(n_c2), 256>>>(d_c2_out, d_c2_v, d_c2_i, d_c2_s, hf_c2, decay_syn_conv, v_th_conv, n_c2, t, nullptr);

        // --- Layer 3: Pool ---
        maxpool2d_kernel<<<GET_BLOCKS(n_p1), 256>>>(d_c2_s, d_p1_out, batch_size, 128, 32, 32);
        cuif_step_kernel<<<GET_BLOCKS(n_p1), 256>>>(d_p1_out, d_p1_v, d_p1_i, d_p1_s, hf_p1, decay_syn_conv, v_th_conv, n_p1, t, nullptr);

        // --- Layer 4: Conv ---
        conv2d_kernel<<<GET_BLOCKS(n_c3), 256>>>(d_p1_s, w_conv3, d_c3_out, batch_size, 128, 128, 16, 16, 3);
        cuif_step_kernel<<<GET_BLOCKS(n_c3), 256>>>(d_c3_out, d_c3_v, d_c3_i, d_c3_s, hf_c3, decay_syn_conv, v_th_conv, n_c3, t, nullptr);

        // --- Layer 5: Pool ---
        maxpool2d_kernel<<<GET_BLOCKS(n_p2), 256>>>(d_c3_s, d_p2_out, batch_size, 128, 16, 16);
        cuif_step_kernel<<<GET_BLOCKS(n_p2), 256>>>(d_p2_out, d_p2_v, d_p2_i, d_p2_s, hf_p2, decay_syn_conv, v_th_conv, n_p2, t, nullptr);

        // --- Layer 6: FC1 (Recurrent) ---
        fc_kernel<<<GET_BLOCKS(n_fc1), 256>>>(d_p2_s, w_fc1, d_fc1_out, batch_size, 8192, 128);
        fc_kernel<<<GET_BLOCKS(n_fc1), 256>>>(d_fc1_s, w_fc1_rec, d_fc1_rec_out, batch_size, 128, 128); // d_fc1_s holds t-1 state
        add_tensors_kernel<<<GET_BLOCKS(n_fc1), 256>>>(d_fc1_out, d_fc1_rec_out, n_fc1);
        cuif_step_kernel<<<GET_BLOCKS(n_fc1), 256>>>(d_fc1_out, d_fc1_v, d_fc1_i, d_fc1_s, hf_fc1, decay_syn_fc, v_th_fc, n_fc1, t, nullptr);

        // --- Layer 7: FC2 (Output) ---
        fc_kernel<<<GET_BLOCKS(n_fc2), 256>>>(d_fc1_s, w_fc2, d_fc2_out, batch_size, 128, 10);
        cuif_step_kernel<<<GET_BLOCKS(n_fc2), 256>>>(d_fc2_out, d_fc2_v, d_fc2_i, d_fc2_s, hf_fc2, decay_syn_fc, v_th_fc, n_fc2, t, d_first_times);
    }

    cudaEventRecord(stop_evt);
    cudaEventSynchronize(stop_evt);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start_evt, stop_evt);

    std::cout << "=================================================" << std::endl;
    std::cout << "Inference time: " << milliseconds << " ms" << std::endl;
    std::cout << "Throughput:     " << (batch_size * 1000.0f) / milliseconds << " samples/sec" << std::endl;
    std::cout << "=================================================" << std::endl;

    float* h_first_times = new float[n_fc2];
    CHECK_CUDA(cudaMemcpy(h_first_times, d_first_times, n_fc2 * sizeof(float), cudaMemcpyDeviceToHost));

    int correct_fs = 0;
    for (int i = 0; i < batch_size; ++i) {
        int true_label = 0;
        int max_gt = h_labels[i * 10];
        for (int c = 1; c < 10; ++c) {
            if (h_labels[i * 10 + c] > max_gt) { max_gt = h_labels[i * 10 + c]; true_label = c; }
        }
        int pred_fs = 0;
        float min_time = h_first_times[i * 10];
        for (int c = 1; c < 10; ++c) {
            if (h_first_times[i * 10 + c] < min_time) { min_time = h_first_times[i * 10 + c]; pred_fs = c; }
        }
        if (pred_fs == true_label) correct_fs++;
    }

    std::cout << "Accuracy (first-spike): " << std::fixed << std::setprecision(2) << ((float)correct_fs / batch_size) * 100.0f << " %" << std::endl;
    std::cout << "=================================================" << std::endl;

    return 0;
}
