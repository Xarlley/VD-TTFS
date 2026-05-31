#include <iostream>
#include <vector>
#include <string>
#include <fstream>
#include <cmath>
#include <cuda_runtime.h>
#include <chrono>

// Time-driven SNN baseline (VGG16 / CIFAR-10), large-batch variant for 16GB V100.
// 5000 images per batch consume roughly 12-14 GB of VRAM.
#define BATCH_SIZE 5000
#define TOTAL_IMAGES 10000
#define TIME_STEPS 680
#define TIME_WINDOW 80.0f
#define TIME_FIRE_START 40.0f
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

// Pure time-driven CUDA kernels

__global__ void k_encode_input(const float* __restrict__ img, char* __restrict__ spikes_out, int t, float tc, float td) {
    // use long long to avoid index overflow at very large thread counts
    long long idx = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= (long long)BATCH_SIZE * 3072) return;
    
    spikes_out[idx] = 0;
    float pixel = img[idx];
    if (pixel < 1e-5f) return;
    
    float t_float = td - tc * logf(pixel);
    if (t_float < 0.0f) t_float = 0.0f;
    
    int t_spike = (int)ceilf(t_float);
    if (t == t_spike && t_spike <= (int)TIME_WINDOW) {
        spikes_out[idx] = 1;
    }
}

__global__ void k_conv_step(
    const char* __restrict__ spikes_in, char* __restrict__ spikes_out, 
    float* __restrict__ vmem, bool* __restrict__ fired,
    const float* __restrict__ weight, const float* __restrict__ bias,
    int C_in, int H_in, int W_in, int C_out, int H_out, int W_out,
    int K_h, int K_w, int t, int depth,
    float tc_integ, float td_integ, float tc_fire, float td_fire) 
{
    long long idx = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    long long total_threads = (long long)BATCH_SIZE * C_out * H_out * W_out;
    if (idx >= total_threads) return;

    if (fired[idx]) { spikes_out[idx] = 0; return; }

    int c_out = idx % C_out;
    int w_out = (idx / C_out) % W_out;
    int h_out = (idx / (C_out * W_out)) % H_out;
    int b     = idx / ((long long)C_out * W_out * H_out);

    float t_integ = (float)t - (depth - 1) * TIME_FIRE_START;
    float t_fire  = (float)t - depth * TIME_FIRE_START;
    float b_val   = bias[c_out];

    if (t_integ >= 0.0f && t_integ < TIME_WINDOW) {
        float kernel_val = expf(-(t_integ - td_integ) / tc_integ);
        float psp = 0.0f;
        int pad_h = K_h / 2, pad_w = K_w / 2;
        
        for (int kh = 0; kh < K_h; ++kh) {
            for (int kw = 0; kw < K_w; ++kw) {
                int h_in = h_out - pad_h + kh;
                int w_in = w_out - pad_w + kw;
                if (h_in >= 0 && h_in < H_in && w_in >= 0 && w_in < W_in) {
                    for (int cin = 0; cin < C_in; ++cin) {
                        long long in_idx = (((long long)b * H_in + h_in) * W_in + w_in) * C_in + cin;
                        if (spikes_in[in_idx]) {
                            int w_idx = ((kh * K_w + kw) * C_in + cin) * C_out + c_out;
                            psp += weight[w_idx] * kernel_val;
                        }
                    }
                }
            }
        }
        vmem[idx] += psp;
    }

    spikes_out[idx] = 0;
    float start_fire = td_fire > 0.0f ? ceilf(td_fire) : 0.0f;
    if (t_fire >= start_fire && t_fire < TIME_WINDOW) {
        float vth = VTH_INIT * expf(-(t_fire - td_fire) / tc_fire);
        if (vmem[idx] + b_val >= vth && vth >= 1e-5f) {
            spikes_out[idx] = 1;
            fired[idx] = true;
        }
    }
}

__global__ void k_pool_step(
    const char* __restrict__ spikes_in, char* __restrict__ spikes_out, bool* __restrict__ fired,
    int C, int H_in, int W_in) 
{
    int H_out = H_in / 2;
    int W_out = W_in / 2;
    long long idx = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= (long long)BATCH_SIZE * C * H_out * W_out) return;

    if (fired[idx]) { spikes_out[idx] = 0; return; }

    int c     = idx % C;
    int w_out = (idx / C) % W_out;
    int h_out = (idx / (C * W_out)) % H_out;
    int b     = idx / ((long long)C * W_out * H_out);

    spikes_out[idx] = 0;
    for (int i = 0; i < 2; ++i) {
        for (int j = 0; j < 2; ++j) {
            int h_in = h_out * 2 + i;
            int w_in = w_out * 2 + j;
            long long in_idx = (((long long)b * H_in + h_in) * W_in + w_in) * C + c;
            if (spikes_in[in_idx]) {
                spikes_out[idx] = 1;
                fired[idx] = true;
                return; 
            }
        }
    }
}

__global__ void k_fc_step(
    const char* __restrict__ spikes_in, char* __restrict__ spikes_out, 
    float* __restrict__ vmem, bool* __restrict__ fired,
    const float* __restrict__ weight, const float* __restrict__ bias,
    int C_in, int C_out, int t, int depth,
    float tc_integ, float td_integ, float tc_fire, float td_fire,
    bool is_output_layer, float* __restrict__ max_vmem) 
{
    long long idx = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= (long long)BATCH_SIZE * C_out) return;

    int c_out = idx % C_out;
    int b     = idx / C_out;

    if (!is_output_layer && fired[idx]) { spikes_out[idx] = 0; return; }

    float t_integ = (float)t - (depth - 1) * TIME_FIRE_START;
    float t_fire  = (float)t - depth * TIME_FIRE_START;
    float b_val   = bias[c_out];

    if (t_integ >= 0.0f && t_integ < TIME_WINDOW) {
        float kernel_val = expf(-(t_integ - td_integ) / tc_integ);
        float psp = 0.0f;
        for (int cin = 0; cin < C_in; ++cin) {
            if (spikes_in[(long long)b * C_in + cin]) {
                psp += weight[cin * C_out + c_out] * kernel_val;
            }
        }
        vmem[idx] += psp;
    }

    if (is_output_layer) {
        if (t_integ >= 0.0f && t_integ < TIME_WINDOW) {
            float current_vmem = vmem[idx] + b_val;
            if (current_vmem > max_vmem[idx]) {
                max_vmem[idx] = current_vmem;
            }
        }
    } else {
        spikes_out[idx] = 0;
        float start_fire = td_fire > 0.0f ? ceilf(td_fire) : 0.0f;
        if (t_fire >= start_fire && t_fire < TIME_WINDOW) {
            float vth = VTH_INIT * expf(-(t_fire - td_fire) / tc_fire);
            if (vmem[idx] + b_val >= vth && vth >= 1e-5f) {
                spikes_out[idx] = 1;
                fired[idx] = true;
            }
        }
    }
}

// weight loader
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

// reset state buffers for each new batch (size_t to avoid overflow)
void reset_layer(char* spk, float* vm, bool* frd, size_t size) {
    CHECK_CUDA(cudaMemset(spk, 0, (size_t)BATCH_SIZE * size * sizeof(char)));
    if (vm) CHECK_CUDA(cudaMemset(vm, 0, (size_t)BATCH_SIZE * size * sizeof(float)));
    if (frd) CHECK_CUDA(cudaMemset(frd, 0, (size_t)BATCH_SIZE * size * sizeof(bool)));
}

int main() {
    std::cout << "Starting EXTREME Time-Driven Baseline SNN Inference..." << std::endl;
    std::cout << "Total Images: " << TOTAL_IMAGES << " | Batch Size: " << BATCH_SIZE << std::endl;
    std::cout << "[INFO] This configuration will consume ~12-14 GB of VRAM." << std::endl;

    std::vector<LayerWeights> layers;
    load_weights("exported_models/snn_weights_vgg.bin", layers);

    std::cout << "Loading 10000 images from disk..." << std::flush;
    std::vector<float> h_all_imgs(TOTAL_IMAGES * 3072);
    std::vector<int> h_all_labels(TOTAL_IMAGES);
    
    for (int img_idx = 0; img_idx < TOTAL_IMAGES; ++img_idx) {
        std::string img_path = "dataset_downloaded/cifar10_float/" + std::to_string(img_idx) + ".bin";
        std::ifstream img_f(img_path, std::ios::binary);
        if (img_f.is_open()) {
            img_f.read((char*)&h_all_imgs[img_idx * 3072], 3072 * 4);
        }

        std::ifstream lbl_f("dataset_downloaded/cifar10_float/label_onehot", std::ios::binary);
        if (lbl_f.is_open()) {
            std::vector<float> lbl(10); 
            lbl_f.seekg(img_idx * 10 * 4, std::ios::beg); 
            lbl_f.read((char*)lbl.data(), 10 * 4);
            int true_cls = -1; 
            for(int i = 0; i < 10; ++i) if(lbl[i] > 0.5f) true_cls = i;
            h_all_labels[img_idx] = true_cls;
        }
    }
    std::cout << " Done." << std::endl;

    // allocate device memory (use size_t to avoid integer overflow on multi-GB sizes)
    float *d_img;
    CHECK_CUDA(cudaMalloc(&d_img, (size_t)BATCH_SIZE * 3072 * 4)); 

    auto alloc_layer = [](char** spk, float** vm, bool** frd, size_t size) {
        CHECK_CUDA(cudaMalloc(spk, (size_t)BATCH_SIZE * size * sizeof(char)));
        if (vm) CHECK_CUDA(cudaMalloc(vm, (size_t)BATCH_SIZE * size * sizeof(float)));
        if (frd) CHECK_CUDA(cudaMalloc(frd, (size_t)BATCH_SIZE * size * sizeof(bool)));
    };

    char *s0, *s1, *s1_1, *p1, *s2, *s2_1, *p2, *s3, *s3_1, *s3_2, *p3;
    char *s4, *s4_1, *s4_2, *p4, *s5, *s5_1, *s5_2, *p5, *s_fc1, *s_fc2, *s_fc3;
    float *v1, *v1_1, *v2, *v2_1, *v3, *v3_1, *v3_2;
    float *v4, *v4_1, *v4_2, *v5, *v5_1, *v5_2, *v_fc1, *v_fc2, *v_fc3;
    bool *f1, *f1_1, *fp1, *f2, *f2_1, *fp2, *f3, *f3_1, *f3_2, *fp3;
    bool *f4, *f4_1, *f4_2, *fp4, *f5, *f5_1, *f5_2, *fp5, *f_fc1, *f_fc2;

    alloc_layer(&s0, nullptr, nullptr, 3072);
    alloc_layer(&s1, &v1, &f1, 32*32*64); alloc_layer(&s1_1, &v1_1, &f1_1, 32*32*64); alloc_layer(&p1, nullptr, &fp1, 16*16*64);
    alloc_layer(&s2, &v2, &f2, 16*16*128); alloc_layer(&s2_1, &v2_1, &f2_1, 16*16*128); alloc_layer(&p2, nullptr, &fp2, 8*8*128);
    alloc_layer(&s3, &v3, &f3, 8*8*256); alloc_layer(&s3_1, &v3_1, &f3_1, 8*8*256); alloc_layer(&s3_2, &v3_2, &f3_2, 8*8*256); alloc_layer(&p3, nullptr, &fp3, 4*4*256);
    alloc_layer(&s4, &v4, &f4, 4*4*512); alloc_layer(&s4_1, &v4_1, &f4_1, 4*4*512); alloc_layer(&s4_2, &v4_2, &f4_2, 4*4*512); alloc_layer(&p4, nullptr, &fp4, 2*2*512);
    alloc_layer(&s5, &v5, &f5, 2*2*512); alloc_layer(&s5_1, &v5_1, &f5_1, 2*2*512); alloc_layer(&s5_2, &v5_2, &f5_2, 2*2*512); alloc_layer(&p5, nullptr, &fp5, 1*1*512);
    alloc_layer(&s_fc1, &v_fc1, &f_fc1, 512); alloc_layer(&s_fc2, &v_fc2, &f_fc2, 512); alloc_layer(&s_fc3, &v_fc3, nullptr, 10);

    float *d_max_vmem_fc3;
    CHECK_CUDA(cudaMalloc(&d_max_vmem_fc3, (size_t)BATCH_SIZE * 10 * sizeof(float)));

    float tc_in = 34.750164f, td_in = 0.0f;       

    // grid sizing for very large thread counts
    #define LAUNCH_CONV(spk_in, spk_out, vm, frd, H_in, W_in, C_in, H_out, W_out, C_out, lay_idx, depth, prev_tc, prev_td) \
        k_conv_step<<<(((long long)BATCH_SIZE * C_out * H_out * W_out) + 255)/256, 256>>>( \
            spk_in, spk_out, vm, frd, layers[lay_idx].d_kernel, layers[lay_idx].d_bias, \
            C_in, H_in, W_in, C_out, H_out, W_out, layers[lay_idx].k_h, layers[lay_idx].k_w, \
            t, depth, prev_tc, prev_td, layers[lay_idx].tc_fire, layers[lay_idx].td)

    #define LAUNCH_POOL(spk_in, spk_out, frd, C, H_in, W_in) \
        k_pool_step<<<(((long long)BATCH_SIZE * C * (H_in/2) * (W_in/2)) + 255)/256, 256>>>( \
            spk_in, spk_out, frd, C, H_in, W_in)

    #define LAUNCH_FC(spk_in, spk_out, vm, frd, C_in, C_out, lay_idx, depth, prev_tc, prev_td, is_out) \
        k_fc_step<<<(((long long)BATCH_SIZE * C_out) + 255)/256, 256>>>( \
            spk_in, spk_out, vm, frd, layers[lay_idx].d_kernel, layers[lay_idx].d_bias, \
            C_in, C_out, t, depth, prev_tc, prev_td, layers[lay_idx].tc_fire, layers[lay_idx].td, is_out, d_max_vmem_fc3)


    int total_correct = 0;
    float total_gpu_time_ms = 0;
    std::vector<float> h_out_max(BATCH_SIZE * 10);
    std::vector<float> h_init_max_vmem(BATCH_SIZE * 10, -1e9f);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    int num_batches = TOTAL_IMAGES / BATCH_SIZE;

    // batch scheduling loop (only a few very large batches)
    for (int batch = 0; batch < num_batches; ++batch) {
        std::cout << "Processing Extreme Batch [" << batch + 1 << "/" << num_batches << "] (5000 images concurrent)..." << std::flush;

        CHECK_CUDA(cudaMemcpy(d_img, &h_all_imgs[batch * BATCH_SIZE * 3072], (size_t)BATCH_SIZE * 3072 * 4, cudaMemcpyHostToDevice));

        // clear state
        reset_layer(s0, nullptr, nullptr, 3072);
        reset_layer(s1, v1, f1, 32*32*64); reset_layer(s1_1, v1_1, f1_1, 32*32*64); reset_layer(p1, nullptr, fp1, 16*16*64);
        reset_layer(s2, v2, f2, 16*16*128); reset_layer(s2_1, v2_1, f2_1, 16*16*128); reset_layer(p2, nullptr, fp2, 8*8*128);
        reset_layer(s3, v3, f3, 8*8*256); reset_layer(s3_1, v3_1, f3_1, 8*8*256); reset_layer(s3_2, v3_2, f3_2, 8*8*256); reset_layer(p3, nullptr, fp3, 4*4*256);
        reset_layer(s4, v4, f4, 4*4*512); reset_layer(s4_1, v4_1, f4_1, 4*4*512); reset_layer(s4_2, v4_2, f4_2, 4*4*512); reset_layer(p4, nullptr, fp4, 2*2*512);
        reset_layer(s5, v5, f5, 2*2*512); reset_layer(s5_1, v5_1, f5_1, 2*2*512); reset_layer(s5_2, v5_2, f5_2, 2*2*512); reset_layer(p5, nullptr, fp5, 1*1*512);
        reset_layer(s_fc1, v_fc1, f_fc1, 512); reset_layer(s_fc2, v_fc2, f_fc2, 512); reset_layer(s_fc3, v_fc3, nullptr, 10);
        
        CHECK_CUDA(cudaMemcpy(d_max_vmem_fc3, h_init_max_vmem.data(), (size_t)BATCH_SIZE * 10 * sizeof(float), cudaMemcpyHostToDevice));

        cudaEventRecord(start);

        for (int t = 0; t < TIME_STEPS; ++t) {
            k_encode_input<<<(((long long)BATCH_SIZE * 3072) + 255)/256, 256>>>(d_img, s0, t, tc_in, td_in);
            
            LAUNCH_CONV(s0,   s1,   v1,   f1,   32, 32, 3,  32, 32, 64, 0, 1, tc_in, td_in);
            LAUNCH_CONV(s1,   s1_1, v1_1, f1_1, 32, 32, 64, 32, 32, 64, 1, 2, layers[0].tc_fire, layers[0].td);
            LAUNCH_POOL(s1_1, p1,   fp1,  64, 32, 32);

            LAUNCH_CONV(p1,   s2,   v2,   f2,   16, 16, 64,  16, 16, 128, 2, 3, layers[1].tc_fire, layers[1].td);
            LAUNCH_CONV(s2,   s2_1, v2_1, f2_1, 16, 16, 128, 16, 16, 128, 3, 4, layers[2].tc_fire, layers[2].td);
            LAUNCH_POOL(s2_1, p2,   fp2,  128, 16, 16);

            LAUNCH_CONV(p2,   s3,   v3,   f3,   8, 8, 128, 8, 8, 256, 4, 5, layers[3].tc_fire, layers[3].td);
            LAUNCH_CONV(s3,   s3_1, v3_1, f3_1, 8, 8, 256, 8, 8, 256, 5, 6, layers[4].tc_fire, layers[4].td);
            LAUNCH_CONV(s3_1, s3_2, v3_2, f3_2, 8, 8, 256, 8, 8, 256, 6, 7, layers[5].tc_fire, layers[5].td);
            LAUNCH_POOL(s3_2, p3,   fp3,  256, 8, 8);

            LAUNCH_CONV(p3,   s4,   v4,   f4,   4, 4, 256, 4, 4, 512, 7, 8,  layers[6].tc_fire, layers[6].td);
            LAUNCH_CONV(s4,   s4_1, v4_1, f4_1, 4, 4, 512, 4, 4, 512, 8, 9,  layers[7].tc_fire, layers[7].td);
            LAUNCH_CONV(s4_1, s4_2, v4_2, f4_2, 4, 4, 512, 4, 4, 512, 9, 10, layers[8].tc_fire, layers[8].td);
            LAUNCH_POOL(s4_2, p4,   fp4,  512, 4, 4);

            LAUNCH_CONV(p4,   s5,   v5,   f5,   2, 2, 512, 2, 2, 512, 10, 11, layers[9].tc_fire,  layers[9].td);
            LAUNCH_CONV(s5,   s5_1, v5_1, f5_1, 2, 2, 512, 2, 2, 512, 11, 12, layers[10].tc_fire, layers[10].td);
            LAUNCH_CONV(s5_1, s5_2, v5_2, f5_2, 2, 2, 512, 2, 2, 512, 12, 13, layers[11].tc_fire, layers[11].td);
            LAUNCH_POOL(s5_2, p5,   fp5,  512, 2, 2);

            LAUNCH_FC(p5,    s_fc1, v_fc1, f_fc1, 512, 512, 13, 14, layers[12].tc_fire, layers[12].td, false);
            LAUNCH_FC(s_fc1, s_fc2, v_fc2, f_fc2, 512, 512, 14, 15, layers[13].tc_fire, layers[13].td, false);
            LAUNCH_FC(s_fc2, s_fc3, v_fc3, nullptr, 512, 10, 15, 16, layers[14].tc_fire, layers[14].td, true);
        }

        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        float batch_time_ms = 0;
        cudaEventElapsedTime(&batch_time_ms, start, stop);
        total_gpu_time_ms += batch_time_ms;

        CHECK_CUDA(cudaMemcpy(h_out_max.data(), d_max_vmem_fc3, (size_t)BATCH_SIZE * 10 * sizeof(float), cudaMemcpyDeviceToHost));
        int batch_correct = 0;
        for (int b = 0; b < BATCH_SIZE; ++b) {
            float max_v = -1e9f;
            int pred_cls = -1;
            for (int i = 0; i < 10; ++i) {
                float val = h_out_max[b * 10 + i];
                if (val > max_v) { 
                    max_v = val; 
                    pred_cls = i; 
                }
            }
            if (pred_cls == h_all_labels[batch * BATCH_SIZE + b]) {
                batch_correct++;
            }
        }
        total_correct += batch_correct;
        std::cout << " Batch Acc: " << (float)batch_correct / BATCH_SIZE * 100.0f << "%" << std::endl;
    }

    float total_seconds = total_gpu_time_ms / 1000.0f;
    std::cout << "\n========================================" << std::endl;
    std::cout << "[EXTREME Baseline CUDA SNN (Time-Driven)]" << std::endl;
    std::cout << "Total Images  : " << TOTAL_IMAGES << std::endl;
    std::cout << "Final Accuracy: " << (float)total_correct / TOTAL_IMAGES * 100.0f << " %" << std::endl;
    std::cout << "Total GPU Time: " << total_seconds << " Seconds" << std::endl;
    std::cout << "Throughput    : " << TOTAL_IMAGES / total_seconds << " Imgs/Sec" << std::endl;
    std::cout << "========================================" << std::endl;

    return 0;
}
