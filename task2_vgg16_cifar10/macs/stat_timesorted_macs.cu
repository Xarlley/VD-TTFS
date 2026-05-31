// VD-TTFS time-sorted early-exit integrator, instrumented for MAC counting (VGG16 / CIFAR-10).
// Full 10000-image inference; reports SNN vs dense-ANN MAC ratios. See repo docs for the method.
// Define NOEARLYEXIT to measure the full sparse pass (Fr * dense) instead.
#include <iostream>
#include <vector>
#include <string>
#include <fstream>
#include <cmath>
#include <cuda_runtime.h>

#define TIME_WINDOW 80.0f
#define TIME_FIRE_START 40.0f
#define VTH_INIT 1.0f
#define INF_TIME 9999.0f
#define CHUNK 8
#define NUM_CHUNKS ((80 + CHUNK - 1) / CHUNK)   // ceil(T/CHUNK)
#define OFF_STRIDE (NUM_CHUNKS + 1)

#define CHECK_CUDA(call) \
    do { cudaError_t err = call; \
         if (err != cudaSuccess) { printf("CUDA Error: %s at line %d\n", cudaGetErrorString(err), __LINE__); exit(1);} \
    } while (0)

struct LayerWeights {
    float* d_kernel;          // ORIGINAL layout: conv [k_h,k_w,c_in,c_out], fc [c_in,c_out]
    float* d_bias;
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
    if (t_spike > TIME_WINDOW) out_spikes[idx] = INF_TIME;
    else out_spikes[idx] = t_spike;
}

// ---- counting-sort each input column's active spikes into 3 arrival chunks ----
__global__ void k_bucketize(const float* __restrict__ in, int N_cols, int C_in,
                            int* __restrict__ out_cin, float* __restrict__ out_tin,
                            int* __restrict__ out_off) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (col >= N_cols) return;
    const int T = (int)TIME_WINDOW;
    const float* src = in + (size_t)col * C_in;

    int cnt[NUM_CHUNKS];
    #pragma unroll
    for (int c = 0; c < NUM_CHUNKS; ++c) cnt[c] = 0;
    for (int i = 0; i < C_in; ++i) {
        float t_in = src[i];
        if (t_in < INF_TIME) {
            int tb = (int)floorf(t_in - TIME_FIRE_START); if (tb < 0) tb = 0;
            if (tb < T) cnt[tb / CHUNK]++;
        }
    }
    int cursor[NUM_CHUNKS];
    int run = 0;
    #pragma unroll
    for (int c = 0; c < NUM_CHUNKS; ++c) { out_off[col * OFF_STRIDE + c] = run; cursor[c] = run; run += cnt[c]; }
    out_off[col * OFF_STRIDE + NUM_CHUNKS] = run;

    int*   dc = out_cin + (size_t)col * C_in;
    float* dt = out_tin + (size_t)col * C_in;
    for (int i = 0; i < C_in; ++i) {
        float t_in = src[i];
        if (t_in < INF_TIME) {
            int tb = (int)floorf(t_in - TIME_FIRE_START); if (tb < 0) tb = 0;
            if (tb < T) {
                int pos = cursor[tb / CHUNK]++;
                dc[pos] = i; dt[pos] = t_in;
            }
        }
    }
}

__device__ unsigned long long g_mac_processed = 0;   // MACs actually executed (input gen)

// ---- chunked conv/fc with input-generation early-exit (INSTRUMENTED) ----
__global__ void k_conv_bucketed(
    float* __restrict__ output_spikes,
    const int* __restrict__ bkt_cin, const float* __restrict__ bkt_tin, const int* __restrict__ bkt_off,
    const float* __restrict__ kernel, const float* __restrict__ bias,
    const float* __restrict__ LUT_decay, const float* __restrict__ LUT_th,
    int t_min, int B,
    int H_in, int W_in, int C_in,
    int H_out, int W_out, int C_out,
    int K_h, int K_w, bool is_fc
) {
    const int T = (int)TIME_WINDOW;
    int nid = blockIdx.x * blockDim.x + threadIdx.x;
    int total = B * H_out * W_out * C_out;
    if (nid >= total) return;
    unsigned long long my_macs = 0;

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

    int pad_h = K_h / 2, pad_w = K_w / 2;
    int h_in_start = h_out - pad_h, w_in_start = w_out - pad_w;

    float carry = bias[c_out];
    float result = INF_TIME;
    float delta[CHUNK];

    for (int tau = 0; tau < NUM_CHUNKS; ++tau) {
        #pragma unroll
        for (int i = 0; i < CHUNK; ++i) delta[i] = 0.0f;
        int base_t = tau * CHUNK;

        if (is_fc) {
            int col = image_id;
            int s = bkt_off[col * OFF_STRIDE + tau], e = bkt_off[col * OFF_STRIDE + tau + 1];
            const int*   bc = bkt_cin + (size_t)col * C_in;
            const float* bt = bkt_tin + (size_t)col * C_in;
            for (int idx = s; idx < e; ++idx) {
                int cin = bc[idx]; float t_in = bt[idx];
                int tb = (int)floorf(t_in - TIME_FIRE_START); if (tb < 0) tb = 0;
                int ti = (int)t_in; if (ti < 0) ti = 0; else if (ti > T) ti = T;
                delta[tb - base_t] += kernel[(size_t)cin * C_out + c_out] * LUT_decay[ti];
                ++my_macs;
            }
        } else {
            for (int kh = 0; kh < K_h; ++kh) {
                int h_in = h_in_start + kh;
                if (h_in < 0 || h_in >= H_in) continue;
                for (int kw = 0; kw < K_w; ++kw) {
                    int w_in = w_in_start + kw;
                    if (w_in < 0 || w_in >= W_in) continue;
                    int col = (image_id * H_in + h_in) * W_in + w_in;
                    int s = bkt_off[col * OFF_STRIDE + tau], e = bkt_off[col * OFF_STRIDE + tau + 1];
                    const int*   bc = bkt_cin + (size_t)col * C_in;
                    const float* bt = bkt_tin + (size_t)col * C_in;
                    int wbase = (kh * K_w + kw) * C_in;
                    for (int idx = s; idx < e; ++idx) {
                        int cin = bc[idx]; float t_in = bt[idx];
                        int tb = (int)floorf(t_in - TIME_FIRE_START); if (tb < 0) tb = 0;
                        int ti = (int)t_in; if (ti < 0) ti = 0; else if (ti > T) ti = T;
                        delta[tb - base_t] += kernel[(size_t)(wbase + cin) * C_out + c_out] * LUT_decay[ti];
                        ++my_macs;
                    }
                }
            }
        }

        // integrate this chunk; break the instant we cross the threshold
        #pragma unroll
        for (int i = 0; i < CHUNK; ++i) {
            int t = base_t + i;
            if (t >= T) break;
            carry += delta[i];
            float th = LUT_th[t];
            if (t >= t_min && carry >= th && th >= 1e-5f) { result = (float)t; break; }
        }
#ifndef NOEARLYEXIT
        if (result < INF_TIME) break;   // fired -> skip remaining chunks (and their synapses)
#endif
    }

    output_spikes[nid] = result;
    atomicAdd(&g_mac_processed, my_macs);
}

__global__ void k_layer_vmem(
    const float* __restrict__ input_spikes, float* __restrict__ output_vmem,
    const float* __restrict__ kernel, const float* __restrict__ bias,
    const float* __restrict__ LUT_decay, int B, int C_in, int C_out
) {
    int local_id = blockIdx.x * blockDim.x + threadIdx.x;
    if (local_id >= B * C_out) return;
    int image_id = local_id / C_out, c_out = local_id % C_out;
    const int T = (int)TIME_WINDOW;
    float acc = 0.0f;
    for (int i = 0; i < C_in; ++i) {
        float t_in = input_spikes[image_id * C_in + i];
        if (t_in < INF_TIME && (t_in - TIME_FIRE_START) <= T) {
            int ti = (int)t_in; if (ti < 0) ti = 0; else if (ti > T) ti = T;
            acc += kernel[i * C_out + c_out] * LUT_decay[ti];
        }
    }
    output_vmem[local_id] = acc + bias[c_out];
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
    const int T = (int)TIME_WINDOW;
    std::vector<float> h_LUT_decay(T + 1), h_LUT_th(T);
    for (int t = 0; t <= T; ++t) h_LUT_decay[t] = expf(-((float)t - td_integ) / tc_integ);
    for (int t = 0; t < T;  ++t) h_LUT_th[t]    = VTH_INIT * expf(-((float)t - td_fire) / tc_fire);
    cudaMemcpy(d_LUT_decay, h_LUT_decay.data(), (T + 1) * 4, cudaMemcpyHostToDevice);
    cudaMemcpy(d_LUT_th, h_LUT_th.data(), T * 4, cudaMemcpyHostToDevice);
}

// bucket buffers (reused across layers, sized for the largest input tensor)
static int   *g_bkt_cin = nullptr;
static float *g_bkt_tin = nullptr;
static int   *g_bkt_off = nullptr;
static unsigned long long g_dense_macs = 0;   // host: dense ANN MAC count over all conv/fc launches

void launch_conv(float* d_in, float* d_out, int H_in, int W_in, int C_in, int H_out, int W_out, int C_out,
                 int layer_idx, float* d_LUT_decay, float* d_LUT_th, const std::vector<LayerWeights>& layers, int B) {
    g_dense_macs += (unsigned long long)B * H_out * W_out * C_out * (layers[layer_idx].k_h * layers[layer_idx].k_w * C_in);
    int t_min = (int)ceilf(layers[layer_idx].td); if (t_min < 0) t_min = 0;
    int N_cols = B * H_in * W_in;
    int bt = 256, bb = (N_cols + bt - 1) / bt;
    k_bucketize<<<bb, bt>>>(d_in, N_cols, C_in, g_bkt_cin, g_bkt_tin, g_bkt_off);
    int n = B * H_out * W_out * C_out, th = 256, blk = (n + th - 1) / th;
    k_conv_bucketed<<<blk, th>>>(d_out, g_bkt_cin, g_bkt_tin, g_bkt_off,
        layers[layer_idx].d_kernel, layers[layer_idx].d_bias, d_LUT_decay, d_LUT_th, t_min,
        B, H_in, W_in, C_in, H_out, W_out, C_out, layers[layer_idx].k_h, layers[layer_idx].k_w, false);
    CHECK_CUDA(cudaDeviceSynchronize());
}

void launch_fc(float* d_in, float* d_out, int C_in, int C_out, int layer_idx,
               float* d_LUT_decay, float* d_LUT_th, const std::vector<LayerWeights>& layers, int B) {
    g_dense_macs += (unsigned long long)B * C_out * C_in;
    int t_min = (int)ceilf(layers[layer_idx].td); if (t_min < 0) t_min = 0;
    int N_cols = B;
    int bt = 256, bb = (N_cols + bt - 1) / bt;
    k_bucketize<<<bb, bt>>>(d_in, N_cols, C_in, g_bkt_cin, g_bkt_tin, g_bkt_off);
    int n = B * C_out, th = 256, blk = (n + th - 1) / th;
    k_conv_bucketed<<<blk, th>>>(d_out, g_bkt_cin, g_bkt_tin, g_bkt_off,
        layers[layer_idx].d_kernel, layers[layer_idx].d_bias, d_LUT_decay, d_LUT_th, t_min,
        B, 1, 1, C_in, 1, 1, C_out, 1, 1, true);
    CHECK_CUDA(cudaDeviceSynchronize());
}

void launch_fc_vmem(float* d_in, float* d_out, int C_in, int C_out, int layer_idx,
                    float* d_LUT_decay, const std::vector<LayerWeights>& layers, int B) {
    int n = B * C_out, threads = 256, blocks = (n + threads - 1) / threads;
    k_layer_vmem<<<blocks, threads>>>(d_in, d_out, layers[layer_idx].d_kernel, layers[layer_idx].d_bias, d_LUT_decay, B, C_in, C_out);
    CHECK_CUDA(cudaDeviceSynchronize());
}

void launch_pool(float* d_in, float* d_out, int H_in, int W_in, int C, int B) {
    int n = B * (H_in/2) * (W_in/2) * C, threads = 256, blocks = (n + threads - 1) / threads;
    k_pooling<<<blocks, threads>>>(d_in, d_out, B, H_in, W_in, C);
    CHECK_CUDA(cudaDeviceSynchronize());
}

int main() {
    const int TOTAL_IMAGES = 10000;
    const int B = 1000;
    const int NUM_BATCHES = TOTAL_IMAGES / B;
    const int T = (int)TIME_WINDOW;

    std::cout << ">>> Loading VGG16 Weights..." << std::endl;
    std::vector<LayerWeights> layers;
    load_weights("exported_models/snn_weights_vgg.bin", layers);

    std::cout << ">>> Loading All " << TOTAL_IMAGES << " CIFAR-10 Test Images..." << std::endl;
    std::vector<float> all_imgs(TOTAL_IMAGES * 3072);
    for (int i = 0; i < TOTAL_IMAGES; ++i) {
        std::ifstream img_f("dataset_downloaded/cifar10_float/" + std::to_string(i) + ".bin", std::ios::binary);
        if (!img_f.is_open()) { std::cerr << "Cannot open image " << i << ".bin" << std::endl; return 1; }
        img_f.read((char*)&all_imgs[i * 3072], 3072 * 4);
    }
    std::vector<float> all_labels(TOTAL_IMAGES * 10);
    std::ifstream lbl_f("dataset_downloaded/cifar10_float/label_onehot", std::ios::binary);
    for (int i = 0; i < TOTAL_IMAGES; ++i) { lbl_f.seekg(i*10*4, std::ios::beg); lbl_f.read((char*)&all_labels[i*10], 10*4); }

    float *d_img, *d_s0, *d_c1, *d_c1_1, *d_p1, *d_c2, *d_c2_1, *d_p2;
    float *d_c3, *d_c3_1, *d_c3_2, *d_p3, *d_c4, *d_c4_1, *d_c4_2, *d_p4;
    float *d_c5, *d_c5_1, *d_c5_2, *d_p5, *d_fc1, *d_fc2, *d_fc3;
    CHECK_CUDA(cudaMalloc(&d_img, B*3072*4)); CHECK_CUDA(cudaMalloc(&d_s0, B*3072*4));
    CHECK_CUDA(cudaMalloc(&d_c1, B*32*32*64*4)); CHECK_CUDA(cudaMalloc(&d_c1_1, B*32*32*64*4)); CHECK_CUDA(cudaMalloc(&d_p1, B*16*16*64*4));
    CHECK_CUDA(cudaMalloc(&d_c2, B*16*16*128*4)); CHECK_CUDA(cudaMalloc(&d_c2_1, B*16*16*128*4)); CHECK_CUDA(cudaMalloc(&d_p2, B*8*8*128*4));
    CHECK_CUDA(cudaMalloc(&d_c3, B*8*8*256*4)); CHECK_CUDA(cudaMalloc(&d_c3_1, B*8*8*256*4)); CHECK_CUDA(cudaMalloc(&d_c3_2, B*8*8*256*4)); CHECK_CUDA(cudaMalloc(&d_p3, B*4*4*256*4));
    CHECK_CUDA(cudaMalloc(&d_c4, B*4*4*512*4)); CHECK_CUDA(cudaMalloc(&d_c4_1, B*4*4*512*4)); CHECK_CUDA(cudaMalloc(&d_c4_2, B*4*4*512*4)); CHECK_CUDA(cudaMalloc(&d_p4, B*2*2*512*4));
    CHECK_CUDA(cudaMalloc(&d_c5, B*2*2*512*4)); CHECK_CUDA(cudaMalloc(&d_c5_1, B*2*2*512*4)); CHECK_CUDA(cudaMalloc(&d_c5_2, B*2*2*512*4)); CHECK_CUDA(cudaMalloc(&d_p5, B*1*1*512*4));
    CHECK_CUDA(cudaMalloc(&d_fc1, B*512*4)); CHECK_CUDA(cudaMalloc(&d_fc2, B*512*4)); CHECK_CUDA(cudaMalloc(&d_fc3, B*10*4));

    // bucket buffers sized for the largest layer input (32*32*64 per image)
    size_t MAX_ELEMS = (size_t)B * 32 * 32 * 64;
    size_t MAX_COLS  = (size_t)B * 32 * 32;
    CHECK_CUDA(cudaMalloc(&g_bkt_cin, MAX_ELEMS * sizeof(int)));
    CHECK_CUDA(cudaMalloc(&g_bkt_tin, MAX_ELEMS * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&g_bkt_off, MAX_COLS * OFF_STRIDE * sizeof(int)));

    float *d_LUT_decay, *d_LUT_th;
    CHECK_CUDA(cudaMalloc(&d_LUT_decay, (T+1)*4)); CHECK_CUDA(cudaMalloc(&d_LUT_th, T*4));

    std::cout << ">>> Starting Full 10K Inference (time-sorted input-gen early-exit, "
              << NUM_BATCHES << " batches of " << B << ")..." << std::endl;

    int total_correct = 0;
    float tc_in = 34.75016403198242f, td_in = 0.0f;
    int et = 256;
    cudaEvent_t start, stop; cudaEventCreate(&start); cudaEventCreate(&stop); cudaEventRecord(start);

    for (int batch = 0; batch < NUM_BATCHES; ++batch) {
        CHECK_CUDA(cudaMemcpy(d_img, &all_imgs[batch*B*3072], B*3072*4, cudaMemcpyHostToDevice));
        k_encode_image<<<(B*3072+et-1)/et, et>>>(d_img, d_s0, B*3072, tc_in, td_in); CHECK_CUDA(cudaDeviceSynchronize());

        update_LUTs(d_LUT_decay, d_LUT_th, tc_in, td_in, layers[0].tc_fire, layers[0].td);
        launch_conv(d_s0, d_c1, 32,32,3, 32,32,64, 0, d_LUT_decay, d_LUT_th, layers, B);
        update_LUTs(d_LUT_decay, d_LUT_th, layers[0].tc_fire, layers[0].td, layers[1].tc_fire, layers[1].td);
        launch_conv(d_c1, d_c1_1, 32,32,64, 32,32,64, 1, d_LUT_decay, d_LUT_th, layers, B);
        launch_pool(d_c1_1, d_p1, 32,32,64, B);

        update_LUTs(d_LUT_decay, d_LUT_th, layers[1].tc_fire, layers[1].td, layers[2].tc_fire, layers[2].td);
        launch_conv(d_p1, d_c2, 16,16,64, 16,16,128, 2, d_LUT_decay, d_LUT_th, layers, B);
        update_LUTs(d_LUT_decay, d_LUT_th, layers[2].tc_fire, layers[2].td, layers[3].tc_fire, layers[3].td);
        launch_conv(d_c2, d_c2_1, 16,16,128, 16,16,128, 3, d_LUT_decay, d_LUT_th, layers, B);
        launch_pool(d_c2_1, d_p2, 16,16,128, B);

        update_LUTs(d_LUT_decay, d_LUT_th, layers[3].tc_fire, layers[3].td, layers[4].tc_fire, layers[4].td);
        launch_conv(d_p2, d_c3, 8,8,128, 8,8,256, 4, d_LUT_decay, d_LUT_th, layers, B);
        update_LUTs(d_LUT_decay, d_LUT_th, layers[4].tc_fire, layers[4].td, layers[5].tc_fire, layers[5].td);
        launch_conv(d_c3, d_c3_1, 8,8,256, 8,8,256, 5, d_LUT_decay, d_LUT_th, layers, B);
        update_LUTs(d_LUT_decay, d_LUT_th, layers[5].tc_fire, layers[5].td, layers[6].tc_fire, layers[6].td);
        launch_conv(d_c3_1, d_c3_2, 8,8,256, 8,8,256, 6, d_LUT_decay, d_LUT_th, layers, B);
        launch_pool(d_c3_2, d_p3, 8,8,256, B);

        update_LUTs(d_LUT_decay, d_LUT_th, layers[6].tc_fire, layers[6].td, layers[7].tc_fire, layers[7].td);
        launch_conv(d_p3, d_c4, 4,4,256, 4,4,512, 7, d_LUT_decay, d_LUT_th, layers, B);
        update_LUTs(d_LUT_decay, d_LUT_th, layers[7].tc_fire, layers[7].td, layers[8].tc_fire, layers[8].td);
        launch_conv(d_c4, d_c4_1, 4,4,512, 4,4,512, 8, d_LUT_decay, d_LUT_th, layers, B);
        update_LUTs(d_LUT_decay, d_LUT_th, layers[8].tc_fire, layers[8].td, layers[9].tc_fire, layers[9].td);
        launch_conv(d_c4_1, d_c4_2, 4,4,512, 4,4,512, 9, d_LUT_decay, d_LUT_th, layers, B);
        launch_pool(d_c4_2, d_p4, 4,4,512, B);

        update_LUTs(d_LUT_decay, d_LUT_th, layers[9].tc_fire, layers[9].td, layers[10].tc_fire, layers[10].td);
        launch_conv(d_p4, d_c5, 2,2,512, 2,2,512, 10, d_LUT_decay, d_LUT_th, layers, B);
        update_LUTs(d_LUT_decay, d_LUT_th, layers[10].tc_fire, layers[10].td, layers[11].tc_fire, layers[11].td);
        launch_conv(d_c5, d_c5_1, 2,2,512, 2,2,512, 11, d_LUT_decay, d_LUT_th, layers, B);
        update_LUTs(d_LUT_decay, d_LUT_th, layers[11].tc_fire, layers[11].td, layers[12].tc_fire, layers[12].td);
        launch_conv(d_c5_1, d_c5_2, 2,2,512, 2,2,512, 12, d_LUT_decay, d_LUT_th, layers, B);
        launch_pool(d_c5_2, d_p5, 2,2,512, B);

        update_LUTs(d_LUT_decay, d_LUT_th, layers[12].tc_fire, layers[12].td, layers[13].tc_fire, layers[13].td);
        launch_fc(d_p5, d_fc1, 512, 512, 13, d_LUT_decay, d_LUT_th, layers, B);
        update_LUTs(d_LUT_decay, d_LUT_th, layers[13].tc_fire, layers[13].td, layers[14].tc_fire, layers[14].td);
        launch_fc(d_fc1, d_fc2, 512, 512, 14, d_LUT_decay, d_LUT_th, layers, B);

        update_LUTs(d_LUT_decay, d_LUT_th, layers[14].tc_fire, layers[14].td, layers[15].tc_fire, layers[15].td);
        launch_fc_vmem(d_fc2, d_fc3, 512, 10, 15, d_LUT_decay, layers, B);

        std::vector<float> h_out(B * 10);
        CHECK_CUDA(cudaMemcpy(h_out.data(), d_fc3, B*10*4, cudaMemcpyDeviceToHost));
        int bc = 0;
        for (int bi = 0; bi < B; ++bi) {
            float mv = -1e9f; int pc = -1;
            for (int i = 0; i < 10; ++i) { float v = h_out[bi*10+i]; if (v > mv) { mv = v; pc = i; } }
            int tc_cls = -1, gi = batch*B + bi;
            for (int i = 0; i < 10; ++i) if (all_labels[gi*10+i] > 0.5f) tc_cls = i;
            if (pc == tc_cls) ++bc;
        }
        total_correct += bc;
        std::cout << "  -> Batch [" << batch+1 << "/" << NUM_BATCHES << "] Correct: " << bc << " / " << B << std::endl;
    }

    cudaEventRecord(stop); cudaEventSynchronize(stop);
    float ms = 0; cudaEventElapsedTime(&ms, start, stop);
    float acc = (float)total_correct / TOTAL_IMAGES * 100.0f;
    std::cout << "\n=======================================" << std::endl;
    std::cout << "TIME-SORTED INPUT-GEN EARLY-EXIT COMPLETED" << std::endl;
    std::cout << "Total Images Processed: " << TOTAL_IMAGES << std::endl;
    std::cout << "Final Accuracy:         " << acc << "%" << std::endl;
    std::cout << "Total Inference Time:   " << ms/1000.0f << " seconds" << std::endl;
    std::cout << "Throughput:             " << (TOTAL_IMAGES/(ms/1000.0f)) << " images/sec" << std::endl;
    std::cout << "=======================================" << std::endl;

    unsigned long long mac_processed = 0;
    CHECK_CUDA(cudaMemcpyFromSymbol(&mac_processed, g_mac_processed, sizeof(mac_processed)));
    double dense = (double)g_dense_macs;
    std::cout << "\n--------- OPERATION COUNT (conv+fc layers) ---------" << std::endl;
#ifdef NOEARLYEXIT
    std::cout << "MODE: NO early-exit (full sparse pass) -> this is Fr * dense" << std::endl;
#else
    std::cout << "MODE: time-sorted early-exit ON (CHUNK=" << CHUNK << ") -> this is Fr * gamma * dense" << std::endl;
#endif
    std::cout << "Dense ANN MACs:        " << (double)dense << std::endl;
    std::cout << "SNN sparse MACs:       " << (double)mac_processed << std::endl;
    std::cout << "SNN/ANN MAC ratio:     " << (mac_processed / dense) << std::endl;
    std::cout << "Ops saved vs ANN:      " << (100.0 * (1.0 - mac_processed / dense)) << " %" << std::endl;
    std::cout << "----------------------------------------------------" << std::endl;
    return 0;
}
