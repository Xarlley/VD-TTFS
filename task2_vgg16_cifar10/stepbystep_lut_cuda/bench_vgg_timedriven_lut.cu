// Time-driven SNN baseline with on-GPU LUTs (VGG16 / CIFAR-10).
#include <iostream>
#include <vector>
#include <string>
#include <fstream>
#include <cmath>
#include <cuda_runtime.h>
#include <chrono>

#define BATCH_SIZE 1000
#define TOTAL_IMAGES 10000
// advance to 720 so the last layer (15 * 40 + 80 = 680, plus margin) fully unrolls
#define TIME_STEPS 720
#define TIME_WINDOW 80
#define TIME_FIRE_START 40
#define VTH_INIT 1.0f

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

// 1. on-GPU LUT generator (avoids CPU/GPU floating-point differences)
__global__ void k_init_luts(float* lut_integ, float* lut_fire, float prev_tc, float prev_td, float tc_fire, float td_fire) {
    int t = threadIdx.x; 
    if (t <= 80) {
        lut_integ[t] = expf(-((float)t - prev_td) / prev_tc);
        lut_fire[t]  = 1.0f * expf(-((float)t - td_fire) / tc_fire);
    }
}

// 2. input-layer encoder
__global__ void k_precompute_input_spikes(const float* __restrict__ img, int* __restrict__ spike_times, float tc, float td) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= BATCH_SIZE * 3072) return;
    
    float pixel = img[idx];
    if (pixel < 1e-5f) { spike_times[idx] = 9999; return; }
    
    float t_float = td - tc * logf(pixel);
    int t_spike = (int)ceilf(t_float < 0.0f ? 0.0f : t_float);
    spike_times[idx] = (t_spike <= 80) ? t_spike : 9999; // allow t=80 limit for dark pixels
}

__global__ void k_encode_input(const int* __restrict__ spike_times, char* __restrict__ spikes_out, int t) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= BATCH_SIZE * 3072) return;
    spikes_out[idx] = (spike_times[idx] == t) ? 1 : 0;
}

// 3. core compute unit (strictly separates the t_in and t_fire phases)
__global__ void k_conv_step(
    const char* __restrict__ spikes_in, char* __restrict__ spikes_out, 
    float* __restrict__ vmem, bool* __restrict__ fired,
    const float* __restrict__ weight, const float* __restrict__ bias,
    const float* __restrict__ lut_integ, const float* __restrict__ lut_fire,
    int t_in, int t_fire, bool do_integ, bool do_fire,
    int C_in, int H_in, int W_in, int C_out, int H_out, int W_out, int K_h, int K_w) 
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= BATCH_SIZE * C_out * H_out * W_out) return;

    if (fired[idx]) { spikes_out[idx] = 0; return; }

    int c_out = idx % C_out, w_out = (idx / C_out) % W_out, h_out = (idx / (C_out * W_out)) % H_out, b = idx / (C_out * W_out * H_out);
    float b_val = bias[c_out];

    if (do_integ) {
        float kernel_val = lut_integ[t_in];
        float psp = 0.0f;
        int pad_h = K_h / 2, pad_w = K_w / 2;
        for (int kh = 0; kh < K_h; ++kh) {
            for (int kw = 0; kw < K_w; ++kw) {
                int h_in = h_out - pad_h + kh, w_in = w_out - pad_w + kw;
                if (h_in >= 0 && h_in < H_in && w_in >= 0 && w_in < W_in) {
                    for (int cin = 0; cin < C_in; ++cin) {
                        if (spikes_in[((b * H_in + h_in) * W_in + w_in) * C_in + cin]) {
                            psp += weight[((kh * K_w + kw) * C_in + cin) * C_out + c_out] * kernel_val;
                        }
                    }
                }
            }
        }
        vmem[idx] += psp;
    }

    spikes_out[idx] = 0;
    if (do_fire) {
        float vth = lut_fire[t_fire];
        if (vmem[idx] + b_val >= vth && vth >= 1e-5f) {
            spikes_out[idx] = 1; fired[idx] = true;
        }
    }
}

__global__ void k_pool_step(
    const char* __restrict__ spikes_in, char* __restrict__ spikes_out, bool* __restrict__ fired,
    int C, int H_in, int W_in) 
{
    int H_out = H_in / 2, W_out = W_in / 2;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= BATCH_SIZE * C * H_out * W_out) return;

    if (fired[idx]) { spikes_out[idx] = 0; return; }

    int c = idx % C, w_out = (idx / C) % W_out, h_out = (idx / (C * W_out)) % H_out, b = idx / (C * W_out * H_out);

    spikes_out[idx] = 0;
    for (int i = 0; i < 2; ++i) {
        for (int j = 0; j < 2; ++j) {
            if (spikes_in[((b * H_in + h_out * 2 + i) * W_in + w_out * 2 + j) * C + c]) {
                spikes_out[idx] = 1; fired[idx] = true; return; 
            }
        }
    }
}

__global__ void k_fc_step(
    const char* __restrict__ spikes_in, char* __restrict__ spikes_out, 
    float* __restrict__ vmem, bool* __restrict__ fired,
    const float* __restrict__ weight, const float* __restrict__ bias,
    const float* __restrict__ lut_integ, const float* __restrict__ lut_fire,
    int t_in, int t_fire, bool do_integ, bool do_fire, int C_in, int C_out) 
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= BATCH_SIZE * C_out) return;

    if (fired[idx]) { spikes_out[idx] = 0; return; }

    int c_out = idx % C_out, b = idx / C_out;
    float b_val = bias[c_out];

    if (do_integ) {
        float kernel_val = lut_integ[t_in];
        float psp = 0.0f;
        for (int cin = 0; cin < C_in; ++cin) {
            if (spikes_in[b * C_in + cin]) psp += weight[cin * C_out + c_out] * kernel_val;
        }
        vmem[idx] += psp;
    }

    spikes_out[idx] = 0;
    if (do_fire) {
        float vth = lut_fire[t_fire];
        if (vmem[idx] + b_val >= vth && vth >= 1e-5f) {
            spikes_out[idx] = 1; fired[idx] = true;
        }
    }
}

__global__ void k_fc_out_step(
    const char* __restrict__ spikes_in, float* __restrict__ vmem,
    const float* __restrict__ weight, const float* __restrict__ bias,
    const float* __restrict__ lut_integ, int t_in, bool do_integ, bool do_eval,
    int C_in, int C_out, float* __restrict__ max_vmem) 
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= BATCH_SIZE * C_out) return;

    int c_out = idx % C_out, b = idx / C_out;
    float b_val = bias[c_out];

    if (do_integ) {
        float kernel_val = lut_integ[t_in];
        float psp = 0.0f;
        for (int cin = 0; cin < C_in; ++cin) {
            if (spikes_in[b * C_in + cin]) psp += weight[cin * C_out + c_out] * kernel_val;
        }
        vmem[idx] += psp;
    }

    if (do_eval) {
        float current_vmem = vmem[idx] + b_val;
        if (current_vmem > max_vmem[idx]) max_vmem[idx] = current_vmem;
    }
}

// host helpers and phase control
inline void launch_conv_host(int t, int lay_idx, char* spk_in, char* spk_out, float* vm, bool* frd, 
                             int H_in, int W_in, int C_in, int H_out, int W_out, int C_out,
                             float* d_lut_integ, float* d_lut_fire, const std::vector<LayerWeights>& layers) 
{
    int t_in   = t - lay_idx * TIME_FIRE_START;
    int t_fire = t - (lay_idx + 1) * TIME_FIRE_START;

    bool do_integ = (t_in >= 0 && t_in <= 80); // capture boundary peaks at negative time
    int td_ceil = (layers[lay_idx].td > 0) ? (int)ceilf(layers[lay_idx].td) : 0;
    bool do_fire  = (t_fire >= td_ceil && t_fire < 80); // strictly truncate at <80

    if (!do_integ && !do_fire) return;

    k_conv_step<<<(BATCH_SIZE * C_out * H_out * W_out + 255)/256, 256>>>(
        spk_in, spk_out, vm, frd, layers[lay_idx].d_kernel, layers[lay_idx].d_bias,
        d_lut_integ + lay_idx * 81, d_lut_fire + lay_idx * 81, t_in, t_fire, do_integ, do_fire,
        C_in, H_in, W_in, C_out, H_out, W_out, layers[lay_idx].k_h, layers[lay_idx].k_w);
}

inline void launch_fc_host(int t, int lay_idx, char* spk_in, char* spk_out, float* vm, bool* frd, 
                           int C_in, int C_out, float* d_lut_integ, float* d_lut_fire, const std::vector<LayerWeights>& layers) 
{
    int t_in   = t - lay_idx * TIME_FIRE_START;
    int t_fire = t - (lay_idx + 1) * TIME_FIRE_START;

    bool do_integ = (t_in >= 0 && t_in <= 80);
    int td_ceil = (layers[lay_idx].td > 0) ? (int)ceilf(layers[lay_idx].td) : 0;
    bool do_fire  = (t_fire >= td_ceil && t_fire < 80); 

    if (!do_integ && !do_fire) return;

    k_fc_step<<<(BATCH_SIZE * C_out + 255)/256, 256>>>(
        spk_in, spk_out, vm, frd, layers[lay_idx].d_kernel, layers[lay_idx].d_bias,
        d_lut_integ + lay_idx * 81, d_lut_fire + lay_idx * 81, t_in, t_fire, do_integ, do_fire, C_in, C_out);
}

inline void launch_fc_out_host(int t, int lay_idx, char* spk_in, float* vm, int C_in, int C_out, 
                               float* max_vmem, float* d_lut_integ, const std::vector<LayerWeights>& layers) 
{
    int t_in   = t - lay_idx * TIME_FIRE_START;
    int t_fire = t - (lay_idx + 1) * TIME_FIRE_START;

    bool do_integ = (t_in >= 0 && t_in <= 80);
    bool do_eval  = (t_fire >= 0 && t_fire < 80); // fully prevents max_vmem drift

    if (!do_integ && !do_eval) return;

    k_fc_out_step<<<(BATCH_SIZE * C_out + 255)/256, 256>>>(
        spk_in, vm, layers[lay_idx].d_kernel, layers[lay_idx].d_bias,
        d_lut_integ + lay_idx * 81, t_in, do_integ, do_eval, C_in, C_out, max_vmem);
}

#define LAUNCH_POOL(spk_in, spk_out, frd, C, H_in, W_in) \
    k_pool_step<<<(BATCH_SIZE * C * (H_in/2) * (W_in/2) + 255)/256, 256>>>(spk_in, spk_out, frd, C, H_in, W_in)

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

void reset_layer(char* spk, float* vm, bool* frd, int size) {
    CHECK_CUDA(cudaMemset(spk, 0, BATCH_SIZE * size * sizeof(char)));
    if (vm) CHECK_CUDA(cudaMemset(vm, 0, BATCH_SIZE * size * sizeof(float)));
    if (frd) CHECK_CUDA(cudaMemset(frd, 0, BATCH_SIZE * size * sizeof(bool)));
}

int main() {
    std::cout << "Starting 100% Accuracy-Aligned Ultimate Baseline SNN..." << std::endl;
    
    std::vector<LayerWeights> layers;
    load_weights("exported_models/snn_weights_vgg.bin", layers);

    // initialize LUTs entirely on the GPU for exact ULP (Unit in the Last Place) consistency
    float tc_in = 34.750164f, td_in = 0.0f;
    float *d_lut_integ, *d_lut_fire;
    CHECK_CUDA(cudaMalloc(&d_lut_integ, 16 * 81 * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_lut_fire,  16 * 81 * sizeof(float)));
    
    for (int i = 0; i < 16; ++i) {
        float prev_tc = (i==0) ? tc_in : layers[i-1].tc_fire;
        float prev_td = (i==0) ? td_in : layers[i-1].td;
        k_init_luts<<<1, 81>>>(d_lut_integ + i * 81, d_lut_fire + i * 81, prev_tc, prev_td, layers[i].tc_fire, layers[i].td);
    }

    std::vector<float> h_all_imgs(TOTAL_IMAGES * 3072);
    std::vector<int> h_all_labels(TOTAL_IMAGES);
    for (int img_idx = 0; img_idx < TOTAL_IMAGES; ++img_idx) {
        std::ifstream img_f("dataset_downloaded/cifar10_float/" + std::to_string(img_idx) + ".bin", std::ios::binary);
        if (img_f.is_open()) img_f.read((char*)&h_all_imgs[img_idx * 3072], 3072 * 4);
        std::ifstream lbl_f("dataset_downloaded/cifar10_float/label_onehot", std::ios::binary);
        if (lbl_f.is_open()) {
            std::vector<float> lbl(10); lbl_f.seekg(img_idx * 10 * 4, std::ios::beg); lbl_f.read((char*)lbl.data(), 10 * 4);
            for(int i = 0; i < 10; ++i) if(lbl[i] > 0.5f) h_all_labels[img_idx] = i;
        }
    }

    float *d_img; CHECK_CUDA(cudaMalloc(&d_img, BATCH_SIZE * 3072 * sizeof(float))); 
    int *d_spike_times; CHECK_CUDA(cudaMalloc(&d_spike_times, BATCH_SIZE * 3072 * sizeof(int)));
    auto alloc_layer = [](char** spk, float** vm, bool** frd, int size) {
        CHECK_CUDA(cudaMalloc(spk, BATCH_SIZE * size * sizeof(char)));
        if (vm) CHECK_CUDA(cudaMalloc(vm, BATCH_SIZE * size * sizeof(float)));
        if (frd) CHECK_CUDA(cudaMalloc(frd, BATCH_SIZE * size * sizeof(bool)));
    };
    char *s0, *s1, *s1_1, *p1, *s2, *s2_1, *p2, *s3, *s3_1, *s3_2, *p3;
    char *s4, *s4_1, *s4_2, *p4, *s5, *s5_1, *s5_2, *p5, *s_fc1, *s_fc2, *s_fc3;
    float *v1, *v1_1, *v2, *v2_1, *v3, *v3_1, *v3_2, *v4, *v4_1, *v4_2, *v5, *v5_1, *v5_2, *v_fc1, *v_fc2, *v_fc3;
    bool *f1, *f1_1, *fp1, *f2, *f2_1, *fp2, *f3, *f3_1, *f3_2, *fp3, *f4, *f4_1, *f4_2, *fp4, *f5, *f5_1, *f5_2, *fp5, *f_fc1, *f_fc2;

    alloc_layer(&s0, nullptr, nullptr, 3072);
    alloc_layer(&s1, &v1, &f1, 32*32*64); alloc_layer(&s1_1, &v1_1, &f1_1, 32*32*64); alloc_layer(&p1, nullptr, &fp1, 16*16*64);
    alloc_layer(&s2, &v2, &f2, 16*16*128); alloc_layer(&s2_1, &v2_1, &f2_1, 16*16*128); alloc_layer(&p2, nullptr, &fp2, 8*8*128);
    alloc_layer(&s3, &v3, &f3, 8*8*256); alloc_layer(&s3_1, &v3_1, &f3_1, 8*8*256); alloc_layer(&s3_2, &v3_2, &f3_2, 8*8*256); alloc_layer(&p3, nullptr, &fp3, 4*4*256);
    alloc_layer(&s4, &v4, &f4, 4*4*512); alloc_layer(&s4_1, &v4_1, &f4_1, 4*4*512); alloc_layer(&s4_2, &v4_2, &f4_2, 4*4*512); alloc_layer(&p4, nullptr, &fp4, 2*2*512);
    alloc_layer(&s5, &v5, &f5, 2*2*512); alloc_layer(&s5_1, &v5_1, &f5_1, 2*2*512); alloc_layer(&s5_2, &v5_2, &f5_2, 2*2*512); alloc_layer(&p5, nullptr, &fp5, 1*1*512);
    alloc_layer(&s_fc1, &v_fc1, &f_fc1, 512); alloc_layer(&s_fc2, &v_fc2, &f_fc2, 512); alloc_layer(&s_fc3, &v_fc3, nullptr, 10);
    float *d_max_vmem_fc3; CHECK_CUDA(cudaMalloc(&d_max_vmem_fc3, BATCH_SIZE * 10 * sizeof(float)));

    int total_correct = 0; float total_gpu_time_ms = 0;
    std::vector<float> h_out_max(BATCH_SIZE * 10);
    std::vector<float> h_init_max_vmem(BATCH_SIZE * 10, -1e9f);
    cudaEvent_t start, stop; cudaEventCreate(&start); cudaEventCreate(&stop);

    int num_batches = TOTAL_IMAGES / BATCH_SIZE;

    for (int batch = 0; batch < num_batches; ++batch) {
        CHECK_CUDA(cudaMemcpy(d_img, &h_all_imgs[batch * BATCH_SIZE * 3072], BATCH_SIZE * 3072 * 4, cudaMemcpyHostToDevice));

        reset_layer(s0, nullptr, nullptr, 3072);
        reset_layer(s1, v1, f1, 32*32*64); reset_layer(s1_1, v1_1, f1_1, 32*32*64); reset_layer(p1, nullptr, fp1, 16*16*64);
        reset_layer(s2, v2, f2, 16*16*128); reset_layer(s2_1, v2_1, f2_1, 16*16*128); reset_layer(p2, nullptr, fp2, 8*8*128);
        reset_layer(s3, v3, f3, 8*8*256); reset_layer(s3_1, v3_1, f3_1, 8*8*256); reset_layer(s3_2, v3_2, f3_2, 8*8*256); reset_layer(p3, nullptr, fp3, 4*4*256);
        reset_layer(s4, v4, f4, 4*4*512); reset_layer(s4_1, v4_1, f4_1, 4*4*512); reset_layer(s4_2, v4_2, f4_2, 4*4*512); reset_layer(p4, nullptr, fp4, 2*2*512);
        reset_layer(s5, v5, f5, 2*2*512); reset_layer(s5_1, v5_1, f5_1, 2*2*512); reset_layer(s5_2, v5_2, f5_2, 2*2*512); reset_layer(p5, nullptr, fp5, 1*1*512);
        reset_layer(s_fc1, v_fc1, f_fc1, 512); reset_layer(s_fc2, v_fc2, f_fc2, 512); reset_layer(s_fc3, v_fc3, nullptr, 10);
        CHECK_CUDA(cudaMemcpy(d_max_vmem_fc3, h_init_max_vmem.data(), BATCH_SIZE * 10 * sizeof(float), cudaMemcpyHostToDevice));

        cudaEventRecord(start);
        k_precompute_input_spikes<<<(BATCH_SIZE * 3072 + 255)/256, 256>>>(d_img, d_spike_times, tc_in, td_in);

        for (int t = 0; t < TIME_STEPS; ++t) {
            k_encode_input<<<(BATCH_SIZE * 3072 + 255)/256, 256>>>(d_spike_times, s0, t);
            
            launch_conv_host(t, 0,  s0,   s1,   v1,   f1,   32, 32, 3,  32, 32, 64, d_lut_integ, d_lut_fire, layers);
            launch_conv_host(t, 1,  s1,   s1_1, v1_1, f1_1, 32, 32, 64, 32, 32, 64, d_lut_integ, d_lut_fire, layers);
            LAUNCH_POOL(s1_1, p1, fp1, 64, 32, 32);

            launch_conv_host(t, 2,  p1,   s2,   v2,   f2,   16, 16, 64,  16, 16, 128, d_lut_integ, d_lut_fire, layers);
            launch_conv_host(t, 3,  s2,   s2_1, v2_1, f2_1, 16, 16, 128, 16, 16, 128, d_lut_integ, d_lut_fire, layers);
            LAUNCH_POOL(s2_1, p2, fp2, 128, 16, 16);

            launch_conv_host(t, 4,  p2,   s3,   v3,   f3,   8, 8, 128, 8, 8, 256, d_lut_integ, d_lut_fire, layers);
            launch_conv_host(t, 5,  s3,   s3_1, v3_1, f3_1, 8, 8, 256, 8, 8, 256, d_lut_integ, d_lut_fire, layers);
            launch_conv_host(t, 6,  s3_1, s3_2, v3_2, f3_2, 8, 8, 256, 8, 8, 256, d_lut_integ, d_lut_fire, layers);
            LAUNCH_POOL(s3_2, p3, fp3, 256, 8, 8);

            launch_conv_host(t, 7,  p3,   s4,   v4,   f4,   4, 4, 256, 4, 4, 512, d_lut_integ, d_lut_fire, layers);
            launch_conv_host(t, 8,  s4,   s4_1, v4_1, f4_1, 4, 4, 512, 4, 4, 512, d_lut_integ, d_lut_fire, layers);
            launch_conv_host(t, 9,  s4_1, s4_2, v4_2, f4_2, 4, 4, 512, 4, 4, 512, d_lut_integ, d_lut_fire, layers);
            LAUNCH_POOL(s4_2, p4, fp4, 512, 4, 4);

            launch_conv_host(t, 10, p4,   s5,   v5,   f5,   2, 2, 512, 2, 2, 512, d_lut_integ, d_lut_fire, layers);
            launch_conv_host(t, 11, s5,   s5_1, v5_1, f5_1, 2, 2, 512, 2, 2, 512, d_lut_integ, d_lut_fire, layers);
            launch_conv_host(t, 12, s5_1, s5_2, v5_2, f5_2, 2, 2, 512, 2, 2, 512, d_lut_integ, d_lut_fire, layers);
            LAUNCH_POOL(s5_2, p5, fp5, 512, 2, 2);

            launch_fc_host(t, 13, p5,    s_fc1, v_fc1, f_fc1, 512, 512, d_lut_integ, d_lut_fire, layers);
            launch_fc_host(t, 14, s_fc1, s_fc2, v_fc2, f_fc2, 512, 512, d_lut_integ, d_lut_fire, layers);
            launch_fc_out_host(t, 15, s_fc2, v_fc3, 512, 10, d_max_vmem_fc3, d_lut_integ, layers);
        }

        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        float batch_time_ms = 0; cudaEventElapsedTime(&batch_time_ms, start, stop);
        total_gpu_time_ms += batch_time_ms;

        CHECK_CUDA(cudaMemcpy(h_out_max.data(), d_max_vmem_fc3, BATCH_SIZE * 10 * sizeof(float), cudaMemcpyDeviceToHost));
        int batch_correct = 0;
        for (int b = 0; b < BATCH_SIZE; ++b) {
            float max_v = -1e9f; int pred = -1;
            for (int i = 0; i < 10; ++i) if (h_out_max[b * 10 + i] > max_v) { max_v = h_out_max[b * 10 + i]; pred = i; }
            if (pred == h_all_labels[batch * BATCH_SIZE + b]) batch_correct++;
        }
        total_correct += batch_correct;
        std::cout << " Batch [" << batch + 1 << "/" << num_batches << "] Acc: " << (float)batch_correct / BATCH_SIZE * 100.0f << "% | Time: " << batch_time_ms << " ms" << std::endl;
    }

    float total_s = total_gpu_time_ms / 1000.0f;
    std::cout << "\n========================================" << std::endl;
    std::cout << "[Final Perfect Baseline SNN (Time-Driven + Strict TTFS Logic)]" << std::endl;
    std::cout << "Total Images  : " << TOTAL_IMAGES << std::endl;
    std::cout << "Accuracy      : " << (float)total_correct / TOTAL_IMAGES * 100.0f << " %" << std::endl;
    std::cout << "Total GPU Time: " << total_s << " Seconds" << std::endl;
    std::cout << "Throughput    : " << TOTAL_IMAGES / total_s << " Imgs/Sec" << std::endl;
    std::cout << "========================================" << std::endl;

    return 0;
}
