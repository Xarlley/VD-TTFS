#include <iostream>
#include <vector>
#include <string>
#include <fstream>
#include <algorithm>
#include <cmath>
#include <cuda_runtime.h>
#include <cfloat>

#define TIME_WINDOW 80.0f
#define TIME_FIRE_START 40.0f  // 核心修复：层与层之间的启动时间偏移
#define VTH_INIT 1.0f
#define INF_TIME 9999.0f
#define MAX_CONN_LIMIT 5000

#define CHECK_CUDA(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            printf("CUDA Error: %s at line %d\n", cudaGetErrorString(err), __LINE__); \
            exit(1); \
        } \
    } while (0)

struct LayerWeights {
    float* d_kernel; float* d_bias;
    float tc_fire; float td;
    int k_h, k_w, c_in, c_out;
};

struct InputEvent {
    float time; float weight;
};

__global__ void k_encode_image(const float* img, float* out_spikes, int size, float tc, float td) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= size) return;
    float pixel = img[idx];
    if (pixel < 1e-5) { out_spikes[idx] = INF_TIME; return; }
    float t_float = td - tc * logf(pixel);
    if (t_float < 0.0f) t_float = 0.0f;
    float t_spike = ceilf(t_float);
    if (t_spike > TIME_WINDOW) out_spikes[idx] = INF_TIME;
    else out_spikes[idx] = t_spike;
}

__global__ void k_layer_inference(
    const float* input_spikes, float* output_spikes,
    const float* kernel, const float* bias, InputEvent* global_event_buffer,
    int H_in, int W_in, int C_in, int H_out, int W_out, int C_out, int K_h, int K_w,
    float tc_fire, float tc_integ, float td_fire, float td_integ, bool is_fc
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= H_out * W_out * C_out) return;

    InputEvent* my_events = global_event_buffer + idx * MAX_CONN_LIMIT;
    int event_count = 0;
    int c_out, w_out, h_out;
    if (is_fc) { c_out = idx; w_out = 0; h_out = 0; } 
    else { c_out = idx % C_out; w_out = (idx / C_out) % W_out; h_out = (idx / C_out) / W_out; }
    float b_val = bias[c_out];

    if (is_fc) {
        for (int i = 0; i < C_in; ++i) {
            float t_in = input_spikes[i];
            if (t_in < INF_TIME) {
                float w = kernel[i * C_out + c_out];
                float w_decayed = w * expf(-(t_in - td_integ) / tc_integ);
                float arrival_time = t_in - TIME_FIRE_START; // 核心修复：对齐到本地时间窗口
                if (event_count < MAX_CONN_LIMIT) my_events[event_count++] = {arrival_time, w_decayed};
            }
        }
    } else {
        int pad_h = K_h / 2, pad_w = K_w / 2;
        for (int kh = 0; kh < K_h; ++kh) {
            for (int kw = 0; kw < K_w; ++kw) {
                int h_in = h_out - pad_h + kh, w_in = w_out - pad_w + kw;
                if (h_in >= 0 && h_in < H_in && w_in >= 0 && w_in < W_in) {
                    for (int cin = 0; cin < C_in; ++cin) {
                        float t_in = input_spikes[(h_in * W_in + w_in) * C_in + cin];
                        if (t_in < INF_TIME) {
                            float w = kernel[((kh * K_w + kw) * C_in + cin) * C_out + c_out];
                            float w_decayed = w * expf(-(t_in - td_integ) / tc_integ);
                            float arrival_time = t_in - TIME_FIRE_START; // 核心修复：对齐到本地时间窗口
                            if (event_count < MAX_CONN_LIMIT) my_events[event_count++] = {arrival_time, w_decayed};
                        }
                    }
                }
            }
        }
    }

    for (int i = 0; i < event_count - 1; ++i)
        for (int j = 0; j < event_count - i - 1; ++j)
            if (my_events[j].time > my_events[j+1].time) {
                InputEvent temp = my_events[j]; my_events[j] = my_events[j+1]; my_events[j+1] = temp;
            }

    float v_mem_acc = 0.0f, final_spike_time = INF_TIME;
    int current_event_idx = 0;
    for (int t = (td_fire > 0 ? (int)ceilf(td_fire) : 0); t < (int)TIME_WINDOW; ++t) {
        while (current_event_idx < event_count && my_events[current_event_idx].time <= (float)t)
            v_mem_acc += my_events[current_event_idx++].weight;
        float v_th = VTH_INIT * expf(-((float)t - td_fire) / tc_fire);
        if (v_mem_acc + b_val >= v_th && v_th >= 1e-5) { final_spike_time = (float)t; break; }
    }
    output_spikes[idx] = final_spike_time;
}

__global__ void k_layer_inference_vmem(
    const float* input_spikes, float* output_vmem,
    const float* kernel, const float* bias, InputEvent* global_event_buffer,
    int C_in, int C_out, float tc_integ, float td_integ
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= C_out) return;

    InputEvent* my_events = global_event_buffer + idx * MAX_CONN_LIMIT;
    int event_count = 0;
    float b_val = bias[idx];

    for (int i = 0; i < C_in; ++i) {
        float t_in = input_spikes[i];
        if (t_in < INF_TIME) {
            float w = kernel[i * C_out + idx];
            float w_decayed = w * expf(-(t_in - td_integ) / tc_integ);
            float arrival_time = t_in - TIME_FIRE_START; // 核心修复：对齐到本地时间窗口
            if (event_count < MAX_CONN_LIMIT) my_events[event_count++] = {arrival_time, w_decayed};
        }
    }

    for (int i = 0; i < event_count - 1; ++i)
        for (int j = 0; j < event_count - i - 1; ++j)
            if (my_events[j].time > my_events[j+1].time) {
                InputEvent temp = my_events[j]; my_events[j] = my_events[j+1]; my_events[j+1] = temp;
            }

    float v_mem_acc = 0.0f;
    float max_v_mem = -1e9f;  
    int current_event_idx = 0;

    for (int t = 0; t < (int)TIME_WINDOW; ++t) {
        while (current_event_idx < event_count && my_events[current_event_idx].time <= (float)t)
            v_mem_acc += my_events[current_event_idx++].weight;
        float v_mem = v_mem_acc + b_val;
        if (v_mem > max_v_mem) max_v_mem = v_mem;
    }
    output_vmem[idx] = max_v_mem;
}

__global__ void k_pooling(const float* in_spikes, float* out_spikes, int H_in, int W_in, int C) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= (H_in / 2) * (W_in / 2) * C) return;
    int c = idx % C, w = (idx / C) % (W_in / 2), h = (idx / C) / (W_in / 2);
    float min_t = INF_TIME;
    for (int i = 0; i < 2; ++i)
        for (int j = 0; j < 2; ++j) {
            int cur_h = h * 2 + i, cur_w = w * 2 + j;
            if (cur_h < H_in && cur_w < W_in) {
                float t = in_spikes[(cur_h * W_in + cur_w) * C + c];
                if (t < min_t) min_t = t;
            }
        }
    out_spikes[idx] = min_t;
}

void load_weights(const std::string& filename, std::vector<LayerWeights>& layers) {
    std::ifstream f(filename, std::ios::binary);
    if (!f.is_open()) { std::cerr << "File not found: " << filename << "\n"; exit(1); }
    int num_layers; f.read((char*)&num_layers, 4);
    for(int i=0; i<num_layers; ++i) {
        LayerWeights l; int ndim;
        f.read((char*)&ndim, 4); std::vector<int> k_shape(ndim); int k_size = 1;
        for(int j=0; j<ndim; ++j) { f.read((char*)&k_shape[j], 4); k_size *= k_shape[j]; }
        if (ndim == 4) { l.k_h = k_shape[0]; l.k_w = k_shape[1]; l.c_in = k_shape[2]; l.c_out = k_shape[3]; }
        else { l.k_h = 1; l.k_w = 1; l.c_in = k_shape[0]; l.c_out = k_shape[1]; }
        std::vector<float> h_kernel(k_size); f.read((char*)h_kernel.data(), k_size * 4);
        CHECK_CUDA(cudaMalloc(&l.d_kernel, k_size * 4)); CHECK_CUDA(cudaMemcpy(l.d_kernel, h_kernel.data(), k_size * 4, cudaMemcpyHostToDevice));
        f.read((char*)&ndim, 4); int b_size; f.read((char*)&b_size, 4);
        std::vector<float> h_bias(b_size); f.read((char*)h_bias.data(), b_size * 4);
        CHECK_CUDA(cudaMalloc(&l.d_bias, b_size * 4)); CHECK_CUDA(cudaMemcpy(l.d_bias, h_bias.data(), b_size * 4, cudaMemcpyHostToDevice));
        f.read((char*)&ndim, 4); int tc_size; f.read((char*)&tc_size, 4); f.read((char*)&l.tc_fire, 4);
        f.read((char*)&ndim, 4); int td_size; f.read((char*)&td_size, 4); f.read((char*)&l.td, 4);
        layers.push_back(l);
    }
}

void launch_conv(float* d_in, float* d_out, int H_in, int W_in, int C_in, int H_out, int W_out, int C_out, int layer_idx, float prev_tc, float prev_td, InputEvent* d_event_buffer, const std::vector<LayerWeights>& layers) {
    k_layer_inference<<< (H_out * W_out * C_out + 255) / 256, 256 >>>(d_in, d_out, layers[layer_idx].d_kernel, layers[layer_idx].d_bias, d_event_buffer, H_in, W_in, C_in, H_out, W_out, C_out, layers[layer_idx].k_h, layers[layer_idx].k_w, layers[layer_idx].tc_fire, prev_tc, layers[layer_idx].td, prev_td, false);
    CHECK_CUDA(cudaDeviceSynchronize());
}
void launch_pool(float* d_in, float* d_out, int H_in, int W_in, int C) {
    k_pooling<<< ((H_in / 2 * W_in / 2 * C) + 255) / 256, 256 >>>(d_in, d_out, H_in, W_in, C);
    CHECK_CUDA(cudaDeviceSynchronize());
}
void launch_fc(float* d_in, float* d_out, int C_in, int C_out, int layer_idx, float prev_tc, float prev_td, InputEvent* d_event_buffer, const std::vector<LayerWeights>& layers) {
    k_layer_inference<<< (C_out + 255) / 256, 256 >>>(d_in, d_out, layers[layer_idx].d_kernel, layers[layer_idx].d_bias, d_event_buffer, 1, 1, C_in, 1, 1, C_out, 1, 1, layers[layer_idx].tc_fire, prev_tc, layers[layer_idx].td, prev_td, true);
    CHECK_CUDA(cudaDeviceSynchronize());
}
void launch_fc_vmem(float* d_in, float* d_out, int C_in, int C_out, int layer_idx, float prev_tc, float prev_td, InputEvent* d_event_buffer, const std::vector<LayerWeights>& layers) {
    k_layer_inference_vmem<<< (C_out + 255) / 256, 256 >>>(d_in, d_out, layers[layer_idx].d_kernel, layers[layer_idx].d_bias, d_event_buffer, C_in, C_out, prev_tc, prev_td);
    CHECK_CUDA(cudaDeviceSynchronize());
}

int main(int argc, char* argv[]) {
    int target_idx = 0;
    if (argc > 1) target_idx = std::atoi(argv[1]);

    std::vector<LayerWeights> layers;
    load_weights("exported_models/snn_weights_vgg.bin", layers);

    std::string img_path = "dataset_downloaded/cifar10_float/" + std::to_string(target_idx) + ".bin";
    std::ifstream img_f(img_path, std::ios::binary);
    std::vector<float> h_img(3072);
    img_f.read((char*)h_img.data(), 3072 * 4);

    float *d_img, *d_s0, *d_c1, *d_c1_1, *d_p1, *d_c2, *d_c2_1, *d_p2, *d_c3, *d_c3_1, *d_c3_2, *d_p3, *d_c4, *d_c4_1, *d_c4_2, *d_p4, *d_c5, *d_c5_1, *d_c5_2, *d_p5, *d_fc1, *d_fc2, *d_fc3;
    CHECK_CUDA(cudaMalloc(&d_img, 3072 * 4)); CHECK_CUDA(cudaMemcpy(d_img, h_img.data(), 3072 * 4, cudaMemcpyHostToDevice)); CHECK_CUDA(cudaMalloc(&d_s0, 3072 * 4)); 
    CHECK_CUDA(cudaMalloc(&d_c1, 32*32*64*4)); CHECK_CUDA(cudaMalloc(&d_c1_1, 32*32*64*4)); CHECK_CUDA(cudaMalloc(&d_p1, 16*16*64*4));
    CHECK_CUDA(cudaMalloc(&d_c2, 16*16*128*4)); CHECK_CUDA(cudaMalloc(&d_c2_1, 16*16*128*4)); CHECK_CUDA(cudaMalloc(&d_p2, 8*8*128*4));
    CHECK_CUDA(cudaMalloc(&d_c3, 8*8*256*4)); CHECK_CUDA(cudaMalloc(&d_c3_1, 8*8*256*4)); CHECK_CUDA(cudaMalloc(&d_c3_2, 8*8*256*4)); CHECK_CUDA(cudaMalloc(&d_p3, 4*4*256*4));
    CHECK_CUDA(cudaMalloc(&d_c4, 4*4*512*4)); CHECK_CUDA(cudaMalloc(&d_c4_1, 4*4*512*4)); CHECK_CUDA(cudaMalloc(&d_c4_2, 4*4*512*4)); CHECK_CUDA(cudaMalloc(&d_p4, 2*2*512*4));
    CHECK_CUDA(cudaMalloc(&d_c5, 2*2*512*4)); CHECK_CUDA(cudaMalloc(&d_c5_1, 2*2*512*4)); CHECK_CUDA(cudaMalloc(&d_c5_2, 2*2*512*4)); CHECK_CUDA(cudaMalloc(&d_p5, 1*1*512*4));
    CHECK_CUDA(cudaMalloc(&d_fc1, 512*4)); CHECK_CUDA(cudaMalloc(&d_fc2, 512*4)); CHECK_CUDA(cudaMalloc(&d_fc3, 10*4));

    InputEvent* d_event_buffer;
    CHECK_CUDA(cudaMalloc(&d_event_buffer, 65536 * MAX_CONN_LIMIT * sizeof(InputEvent)));

    float tc_in = 34.75016403198242f, td_in = 0.0f;       
    k_encode_image<<<(3072+255)/256, 256>>>(d_img, d_s0, 3072, tc_in, td_in); CHECK_CUDA(cudaDeviceSynchronize());

    launch_conv(d_s0,   d_c1,   32, 32, 3,  32, 32, 64, 0, tc_in, td_in, d_event_buffer, layers);
    launch_conv(d_c1,   d_c1_1, 32, 32, 64, 32, 32, 64, 1, layers[0].tc_fire, layers[0].td, d_event_buffer, layers);
    launch_pool(d_c1_1, d_p1,   32, 32, 64);

    launch_conv(d_p1,   d_c2,   16, 16, 64,  16, 16, 128, 2, layers[1].tc_fire, layers[1].td, d_event_buffer, layers);
    launch_conv(d_c2,   d_c2_1, 16, 16, 128, 16, 16, 128, 3, layers[2].tc_fire, layers[2].td, d_event_buffer, layers);
    launch_pool(d_c2_1, d_p2,   16, 16, 128);

    launch_conv(d_p2,   d_c3,   8, 8, 128, 8, 8, 256, 4, layers[3].tc_fire, layers[3].td, d_event_buffer, layers);
    launch_conv(d_c3,   d_c3_1, 8, 8, 256, 8, 8, 256, 5, layers[4].tc_fire, layers[4].td, d_event_buffer, layers);
    launch_conv(d_c3_1, d_c3_2, 8, 8, 256, 8, 8, 256, 6, layers[5].tc_fire, layers[5].td, d_event_buffer, layers);
    launch_pool(d_c3_2, d_p3,   8, 8, 256);

    launch_conv(d_p3,   d_c4,   4, 4, 256, 4, 4, 512, 7, layers[6].tc_fire, layers[6].td, d_event_buffer, layers);
    launch_conv(d_c4,   d_c4_1, 4, 4, 512, 4, 4, 512, 8, layers[7].tc_fire, layers[7].td, d_event_buffer, layers);
    launch_conv(d_c4_1, d_c4_2, 4, 4, 512, 4, 4, 512, 9, layers[8].tc_fire, layers[8].td, d_event_buffer, layers);
    launch_pool(d_c4_2, d_p4,   4, 4, 512);

    launch_conv(d_p4,   d_c5,   2, 2, 512, 2, 2, 512, 10, layers[9].tc_fire,  layers[9].td,  d_event_buffer, layers);
    launch_conv(d_c5,   d_c5_1, 2, 2, 512, 2, 2, 512, 11, layers[10].tc_fire, layers[10].td, d_event_buffer, layers);
    launch_conv(d_c5_1, d_c5_2, 2, 2, 512, 2, 2, 512, 12, layers[11].tc_fire, layers[11].td, d_event_buffer, layers);
    launch_pool(d_c5_2, d_p5,   2, 2, 512);

    launch_fc(d_p5,  d_fc1, 512, 512, 13, layers[12].tc_fire, layers[12].td, d_event_buffer, layers);
    launch_fc(d_fc1, d_fc2, 512, 512, 14, layers[13].tc_fire, layers[13].td, d_event_buffer, layers);
    launch_fc_vmem(d_fc2, d_fc3, 512, 10,  15, layers[14].tc_fire, layers[14].td, d_event_buffer, layers);

    std::vector<float> h_out(10);
    CHECK_CUDA(cudaMemcpy(h_out.data(), d_fc3, 10 * 4, cudaMemcpyDeviceToHost));

    float max_v = -1e9f;
    int pred_cls = -1;

    for (int i = 0; i < 10; ++i) {
        if (h_out[i] > max_v) { 
            max_v = h_out[i]; 
            pred_cls = i; 
        }
    }

    std::cout << "Prediction: " << pred_cls << "\n";

    std::ifstream lbl_f("dataset_downloaded/cifar10_float/label_onehot", std::ios::binary);
    std::vector<float> lbl(10); lbl_f.seekg(target_idx * 10 * 4, std::ios::beg); lbl_f.read((char*)lbl.data(), 10 * 4);
    int true_cls = -1; for(int i = 0; i < 10; ++i) if(lbl[i] > 0.5f) true_cls = i;
    std::cout << "Ground Truth: " << true_cls << std::endl;

    if (pred_cls == true_cls && pred_cls != -1) std::cout << "SUCCESS!\n"; else std::cout << "FAILED.\n";

    return 0;
}
