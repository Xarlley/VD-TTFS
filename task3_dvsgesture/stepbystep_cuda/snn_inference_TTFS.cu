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

#define LAUNCH_CHECK() { CHECK_CUDA(cudaGetLastError()); }
#define GET_BLOCKS(total) ((total + 255) / 256)

// ==========================================================
// TTFS-SNN 超参数与配置 (严格对齐你的最新修改)
// ==========================================================
const int NUM_SAMPLES = 1078;
const int T_TOTAL = 160; 
const int NUM_CLASSES = 10;

// [调整] 突触电流衰减 (保留)
const float decay_syn_conv = exp(-1.0f / 5.0f);
const float decay_syn_fc = exp(-1.0f / 60.0f);

// [调整] 严格 TTFS 模式下的大阈值
const float v_th_conv = 5.0f;
const float v_th_fc = 10.0f;

// ==========================================================
// 核心 CUDA 算子 (全部植入 sample_active 早退机制)
// ==========================================================

__global__ void extract_input_step(const float* all_data, float* step_input, int N, int C, int H, int W, int T_total, int t, const bool* sample_active) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_pixels = N * C * H * W;
    if (idx < total_pixels) {
        int n = idx / (C * H * W);
        if (!sample_active[n]) return; // 样本早退，免提取！
        step_input[idx] = all_data[idx * T_total + t];
    }
}

__global__ void conv2d_kernel(const float* input, const float* weight, float* output, 
                              int N, int C_in, int C_out, int H, int W, int K, const bool* sample_active) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = N * C_out * H * W;
    if (idx < total) {
        int n = idx / (C_out * H * W);
        if (!sample_active[n]) return; // 样本早退，免计算！

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
                                 int N, int C, int H_in, int W_in, const bool* sample_active) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int H_out = H_in / 2;
    int W_out = W_in / 2;
    int total = N * C * H_out * W_out;
    
    if (idx < total) {
        int n = idx / (C * H_out * W_out);
        if (!sample_active[n]) return;

        int x_out = idx % W_out;
        int y_out = (idx / W_out) % H_out;
        int c = (idx / (W_out * H_out)) % C;

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
                          int N, int in_features, int out_features, const bool* sample_active) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = N * out_features;
    if (idx < total) {
        int row = idx / out_features; // 即 N
        if (!sample_active[row]) return;

        int col = idx % out_features;
        float sum = 0.0f;
        for (int i = 0; i < in_features; ++i) {
            sum += input[row * in_features + i] * weight[col * in_features + i];
        }
        output[idx] = sum;
    }
}

__global__ void add_tensors_kernel(float* a, const float* b, int size, int features_per_sample, const bool* sample_active) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        if (!sample_active[idx / features_per_sample]) return;
        a[idx] += b[idx];
    }
}

// ----------------------------------------------------------
// 严格 CuIF 动力学 Kernel
// ----------------------------------------------------------
__global__ void cuif_ttfs_kernel(
    const float* input, float* v, float* I, float* spikes, bool* has_fired,
    float decay_syn, float v_th, int total_neurons, int neurons_per_sample, 
    const bool* sample_active, int t, float* first_times) 
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < total_neurons) {
        int n = idx / neurons_per_sample;
        
        // 1. 样本早退，切断所有脉冲发送，防止递归层接收到幽灵信号
        if (!sample_active[n]) {
            spikes[idx] = 0.0f; 
            return;
        }

        // 2. 神经元已发放过脉冲 (严格 TTFS 拦截)，输出归零并跳过更新
        if (has_fired[idx]) {
            spikes[idx] = 0.0f;
            return;
        }

        float curr_I = I[idx];
        float curr_v = v[idx];

        // 3. CuIF 动力学：无电位泄露积分
        curr_I = curr_I * decay_syn + input[idx];
        curr_v = curr_v + curr_I; 

        // 4. 脉冲触发与状态钳制
        if (curr_v >= v_th) {
            spikes[idx] = 1.0f;
            has_fired[idx] = true;
            v[idx] = -10000.0f; // 钳制电位
            I[idx] = curr_I;
            
            // 如果是 FC2 输出层，记录首次发放时间
            if (first_times != nullptr && first_times[idx] > 9000.0f) {
                first_times[idx] = (float)t;
            }
        } else {
            spikes[idx] = 0.0f;
            v[idx] = curr_v;
            I[idx] = curr_I;
        }
    }
}

// ----------------------------------------------------------
// 核心：监控输出层，触发样本早退
// ----------------------------------------------------------
__global__ void check_sample_early_exit(const float* fc2_spikes, bool* sample_active, int N, int num_classes) {
    int n = blockIdx.x * blockDim.x + threadIdx.x;
    if (n < N && sample_active[n]) {
        for(int c = 0; c < num_classes; ++c) {
            if (fc2_spikes[n * num_classes + c] > 0.5f) {
                sample_active[n] = false; // 触发全网络熔断！
                break;
            }
        }
    }
}

// ==========================================================
// 辅助函数
// ==========================================================
void load_binary(const char* filepath, void* host_ptr, size_t bytes) {
    std::ifstream file(filepath, std::ios::binary);
    if (!file) { std::cerr << "Failed to open " << filepath << std::endl; exit(1); }
    file.read(reinterpret_cast<char*>(host_ptr), bytes);
    file.close();
}

int main() {
    std::cout << "🚀 Starting Strict TTFS-SNN with Sample-Level Early Exit..." << std::endl;
    int batch_size = NUM_SAMPLES;
    
    // 1. 加载最新权重
    size_t w_bytes = 1292032 * sizeof(float);
    float* h_weights = new float[1292032];
    load_binary("cuda_assets/dvsgesture_weights.bin", h_weights, w_bytes);
    float* d_weights;
    CHECK_CUDA(cudaMalloc(&d_weights, w_bytes));
    CHECK_CUDA(cudaMemcpy(d_weights, h_weights, w_bytes, cudaMemcpyHostToDevice));

    float *w_conv1 = d_weights + 0, *w_conv2 = d_weights + 1152, *w_conv3 = d_weights + 74880;
    float *w_fc1_rec = d_weights + 222336, *w_fc1 = d_weights + 238720, *w_fc2 = d_weights + 1287296;

    // 2. 加载数据
    size_t data_bytes = batch_size * 2 * 32 * 32 * T_TOTAL * sizeof(float);
    float* h_data = new float[batch_size * 2 * 32 * 32 * T_TOTAL];
    load_binary("cuda_assets/train_data.bin", h_data, data_bytes);
    float* d_data;
    CHECK_CUDA(cudaMalloc(&d_data, data_bytes));
    CHECK_CUDA(cudaMemcpy(d_data, h_data, data_bytes, cudaMemcpyHostToDevice));

    int* h_labels = new int[batch_size * NUM_CLASSES];
    load_binary("cuda_assets/train_labels.bin", h_labels, batch_size * NUM_CLASSES * sizeof(int));

    // 3. 显存分配
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

    // 【新增】样本级早退数组
    bool* d_sample_active;
    CHECK_CUDA(cudaMalloc(&d_sample_active, batch_size * sizeof(bool)));
    bool* h_sample_active = new bool[batch_size];
    std::fill_n(h_sample_active, batch_size, true);
    CHECK_CUDA(cudaMemcpy(d_sample_active, h_sample_active, batch_size * sizeof(bool), cudaMemcpyHostToDevice));

    float *d_first_times;
    CHECK_CUDA(cudaMalloc(&d_first_times, n_fc2 * sizeof(float)));
    float* h_first_times_init = new float[n_fc2];
    std::fill_n(h_first_times_init, n_fc2, 9999.0f);
    CHECK_CUDA(cudaMemcpy(d_first_times, h_first_times_init, n_fc2 * sizeof(float), cudaMemcpyHostToDevice));
    delete[] h_first_times_init;

    // ==========================================================
    // 推理计时开始
    // ==========================================================
    cudaEvent_t start_evt, stop_evt;
    cudaEventCreate(&start_evt); cudaEventCreate(&stop_evt);
    cudaEventRecord(start_evt);

    for (int t = 0; t < T_TOTAL; ++t) {
        extract_input_step<<<GET_BLOCKS(batch_size * 2 * 32 * 32), 256>>>(d_data, d_step_in, batch_size, 2, 32, 32, T_TOTAL, t, d_sample_active);
        
        // --- Layer 1: Conv ---
        conv2d_kernel<<<GET_BLOCKS(n_c1), 256>>>(d_step_in, w_conv1, d_c1_out, batch_size, 2, 64, 32, 32, 3, d_sample_active);
        cuif_ttfs_kernel<<<GET_BLOCKS(n_c1), 256>>>(d_c1_out, d_c1_v, d_c1_i, d_c1_s, hf_c1, decay_syn_conv, v_th_conv, n_c1, 64*32*32, d_sample_active, t, nullptr);

        // --- Layer 2: Conv ---
        conv2d_kernel<<<GET_BLOCKS(n_c2), 256>>>(d_c1_s, w_conv2, d_c2_out, batch_size, 64, 128, 32, 32, 3, d_sample_active);
        cuif_ttfs_kernel<<<GET_BLOCKS(n_c2), 256>>>(d_c2_out, d_c2_v, d_c2_i, d_c2_s, hf_c2, decay_syn_conv, v_th_conv, n_c2, 128*32*32, d_sample_active, t, nullptr);

        // --- Layer 3: Pool ---
        maxpool2d_kernel<<<GET_BLOCKS(n_p1), 256>>>(d_c2_s, d_p1_out, batch_size, 128, 32, 32, d_sample_active);
        cuif_ttfs_kernel<<<GET_BLOCKS(n_p1), 256>>>(d_p1_out, d_p1_v, d_p1_i, d_p1_s, hf_p1, decay_syn_conv, v_th_conv, n_p1, 128*16*16, d_sample_active, t, nullptr);

        // --- Layer 4: Conv ---
        conv2d_kernel<<<GET_BLOCKS(n_c3), 256>>>(d_p1_s, w_conv3, d_c3_out, batch_size, 128, 128, 16, 16, 3, d_sample_active);
        cuif_ttfs_kernel<<<GET_BLOCKS(n_c3), 256>>>(d_c3_out, d_c3_v, d_c3_i, d_c3_s, hf_c3, decay_syn_conv, v_th_conv, n_c3, 128*16*16, d_sample_active, t, nullptr);

        // --- Layer 5: Pool ---
        maxpool2d_kernel<<<GET_BLOCKS(n_p2), 256>>>(d_c3_s, d_p2_out, batch_size, 128, 16, 16, d_sample_active);
        cuif_ttfs_kernel<<<GET_BLOCKS(n_p2), 256>>>(d_p2_out, d_p2_v, d_p2_i, d_p2_s, hf_p2, decay_syn_conv, v_th_conv, n_p2, 128*8*8, d_sample_active, t, nullptr);

        // --- Layer 6: FC1 (Recurrent) ---
        fc_kernel<<<GET_BLOCKS(n_fc1), 256>>>(d_p2_s, w_fc1, d_fc1_out, batch_size, 8192, 128, d_sample_active);
        fc_kernel<<<GET_BLOCKS(n_fc1), 256>>>(d_fc1_s, w_fc1_rec, d_fc1_rec_out, batch_size, 128, 128, d_sample_active); // d_fc1_s 是 t-1 遗留状态
        add_tensors_kernel<<<GET_BLOCKS(n_fc1), 256>>>(d_fc1_out, d_fc1_rec_out, n_fc1, 128, d_sample_active);
        cuif_ttfs_kernel<<<GET_BLOCKS(n_fc1), 256>>>(d_fc1_out, d_fc1_v, d_fc1_i, d_fc1_s, hf_fc1, decay_syn_fc, v_th_fc, n_fc1, 128, d_sample_active, t, nullptr);

        // --- Layer 7: FC2 (Output) ---
        fc_kernel<<<GET_BLOCKS(n_fc2), 256>>>(d_fc1_s, w_fc2, d_fc2_out, batch_size, 128, 10, d_sample_active);
        cuif_ttfs_kernel<<<GET_BLOCKS(n_fc2), 256>>>(d_fc2_out, d_fc2_v, d_fc2_i, d_fc2_s, hf_fc2, decay_syn_fc, v_th_fc, n_fc2, 10, d_sample_active, t, d_first_times);

        // 【极其关键】：扫描 FC2 脉冲，更新早退矩阵
        check_sample_early_exit<<<GET_BLOCKS(batch_size), 256>>>(d_fc2_s, d_sample_active, batch_size, 10);
    }

    cudaEventRecord(stop_evt);
    cudaEventSynchronize(stop_evt);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start_evt, stop_evt);
    
    // 统计有多少样本最终被熔断早退
    CHECK_CUDA(cudaMemcpy(h_sample_active, d_sample_active, batch_size * sizeof(bool), cudaMemcpyDeviceToHost));
    int active_samples = 0;
    for(int i=0; i<batch_size; i++) if(h_sample_active[i]) active_samples++;

    std::cout << "=================================================" << std::endl;
    std::cout << "⏱️ TTFS 动态熔断版耗时: " << milliseconds << " ms" << std::endl;
    std::cout << "🔥 最终吞吐量: " << (batch_size * 1000.0f) / milliseconds << " samples/sec" << std::endl;
    std::cout << "🎯 成功熔断的样本数量: " << (batch_size - active_samples) << " / " << batch_size << std::endl;
    std::cout << "=================================================" << std::endl;

    // ==========================================================
    // 验证准确率 (Argmin over first_times)
    // ==========================================================
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

    std::cout << "✅ Strict TTFS Accuracy: " << std::fixed << std::setprecision(2) << ((float)correct_fs / batch_size) * 100.0f << " %" << std::endl;
    std::cout << "=================================================" << std::endl;

    return 0;
}