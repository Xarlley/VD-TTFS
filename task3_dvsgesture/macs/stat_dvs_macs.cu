// MAC counter for the VD-TTFS time-sorted integrator (conv2+conv3+fc2).
// Network: 7-layer recurrent CuLIF SNN, Task 3 (DVSGesture).
// Build: nvcc stat_dvs_macs.cu -o stat_dvs_macs -O3 -Wno-deprecated-gpu-targets
//   (define NOEARLYEXIT to count MACs without early exit)
#include <iostream>
#include <fstream>
#include <vector>
#include <cmath>
#include <iomanip>
#include <cstdint>
#include <cuda_runtime.h>

#define CHECK_CUDA(call) { cudaError_t err = call; \
    if (err != cudaSuccess) { std::cerr << "CUDA Error: " << cudaGetErrorString(err) << " at line " << __LINE__ << std::endl; exit(EXIT_FAILURE);} }
#define GET_BLOCKS(total) ((total + 255) / 256)

const int NUM_SAMPLES = 1078;
const int T_TOTAL = 160;
const int NUM_CLASSES = 10;
const int CHUNK_SIZE = 32;
const int NUM_CHUNKS = (T_TOTAL + CHUNK_SIZE - 1) / CHUNK_SIZE;   // 5
const int OFF_STRIDE = NUM_CHUNKS + 1;

const float decay_syn_conv = exp(-1.0f / 5.0f);
const float decay_syn_fc   = exp(-1.0f / 60.0f);
const float v_th_conv = 5.0f;
const float v_th_fc   = 10.0f;

__device__ unsigned long long g_mac=0;
// =================== layer 1: raw multi-spike, streaming, early-exit ==========
__global__ void conv1_raw_early(const float* all_data, float* out_ft, const float* weight,
                                int N, float decay, float th) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N * 64 * 32 * 32) return;
    int n = idx / (64*32*32), rem = idx % (64*32*32);
    int c_out = rem / (32*32), y = (rem/32)%32, x = rem%32;
    float I = 0.0f, V = 0.0f;
    int n_c_offset = n * 2*32*32*160, w_out_offset = c_out * 2*9;
    for (int chunk = 0; chunk < T_TOTAL; chunk += CHUNK_SIZE) {
        float in_val[CHUNK_SIZE] = {0.0f};
        for (int c_in = 0; c_in < 2; ++c_in) {
            int in_c_base = n_c_offset + c_in*32*32*160, w_in_base = w_out_offset + c_in*9;
            for (int ky = 0; ky < 3; ++ky) {
                int in_y = y+ky-1; if (in_y<0||in_y>=32) continue;
                int in_y_base = in_c_base + in_y*32*160, w_ky_base = w_in_base + ky*3;
                for (int kx = 0; kx < 3; ++kx) {
                    int in_x = x+kx-1; if (in_x<0||in_x>=32) continue;
                    int data_base = in_y_base + in_x*160 + chunk;
                    float w = weight[w_ky_base + kx];
                    const float4* dv = reinterpret_cast<const float4*>(&all_data[data_base]);
                    #pragma unroll
                    for (int t4 = 0; t4 < CHUNK_SIZE/4; ++t4) {
                        float4 v = dv[t4];
                        in_val[t4*4+0]+=w*v.x; in_val[t4*4+1]+=w*v.y; in_val[t4*4+2]+=w*v.z; in_val[t4*4+3]+=w*v.w;
                    }
                }
            }
        }
        #pragma unroll
        for (int t = 0; t < CHUNK_SIZE; ++t) {
            I = I*decay + in_val[t]; V += I;
            if (V >= th) { out_ft[idx] = (float)(chunk + t); return; }
        }
    }
    out_ft[idx] = 9999.0f;
}

// =================== time-sorted bucketize for a single-spike map =============
// Input fire-time map laid out [N, C, H, W] (channel-major). One "site" = (n,h,w);
// its C channels (stride H*W) are counting-sorted into NUM_CHUNKS arrival buckets.
__global__ void k_bucketize_conv(const float* in_ft, int N, int C, int H, int W,
                                 int* out_cin, float* out_tin, int* out_off) {
    int site = blockIdx.x * blockDim.x + threadIdx.x;
    if (site >= N * H * W) return;
    int n = site / (H*W), hw = site % (H*W);
    int base = n * C * H * W + hw;           // channel c is at base + c*(H*W)
    int cnt[NUM_CHUNKS]; for (int c=0;c<NUM_CHUNKS;++c) cnt[c]=0;
    for (int c = 0; c < C; ++c) {
        float tf = in_ft[base + c*H*W];
        if (tf < 9999.0f) { int tb=(int)tf; if (tb>=0 && tb<T_TOTAL) cnt[tb/CHUNK_SIZE]++; }
    }
    int cur[NUM_CHUNKS], run = 0;
    for (int c=0;c<NUM_CHUNKS;++c){ out_off[site*OFF_STRIDE+c]=run; cur[c]=run; run+=cnt[c]; }
    out_off[site*OFF_STRIDE+NUM_CHUNKS]=run;
    int*   dc = out_cin + (size_t)site*C;
    float* dt = out_tin + (size_t)site*C;
    for (int c = 0; c < C; ++c) {
        float tf = in_ft[base + c*H*W];
        if (tf < 9999.0f) { int tb=(int)tf; if (tb>=0 && tb<T_TOTAL){ int p=cur[tb/CHUNK_SIZE]++; dc[p]=c; dt[p]=tf; } }
    }
}

// one column per sample for an FC input vector [N, F]
__global__ void k_bucketize_fc(const float* in_ft, int N, int F,
                               int* out_cin, float* out_tin, int* out_off) {
    int n = blockIdx.x * blockDim.x + threadIdx.x;
    if (n >= N) return;
    const float* src = in_ft + (size_t)n * F;
    int cnt[NUM_CHUNKS]; for (int c=0;c<NUM_CHUNKS;++c) cnt[c]=0;
    for (int j = 0; j < F; ++j) { float tf=src[j]; if (tf<9999.0f){ int tb=(int)tf; if(tb>=0&&tb<T_TOTAL) cnt[tb/CHUNK_SIZE]++; } }
    int cur[NUM_CHUNKS], run = 0;
    for (int c=0;c<NUM_CHUNKS;++c){ out_off[n*OFF_STRIDE+c]=run; cur[c]=run; run+=cnt[c]; }
    out_off[n*OFF_STRIDE+NUM_CHUNKS]=run;
    int*   dc = out_cin + (size_t)n*F;
    float* dt = out_tin + (size_t)n*F;
    for (int j = 0; j < F; ++j) { float tf=src[j]; if (tf<9999.0f){ int tb=(int)tf; if(tb>=0&&tb<T_TOTAL){ int p=cur[tb/CHUNK_SIZE]++; dc[p]=j; dt[p]=tf; } } }
}

// =================== bucketized conv with chunked early-exit ==================
__global__ void conv_bucketed_early(const int* bcin, const float* btin, const int* boff,
                                    float* out_ft, const float* weight,
                                    int N, int C_in, int C_out, int H, int W, float decay, float th) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N * C_out * H * W) return;
    int n = idx / (C_out*H*W), rem = idx % (C_out*H*W);
    int c_out = rem/(H*W), y = (rem/W)%H, x = rem%W;
    float I = 0.0f, V = 0.0f;
    int w_out_base = c_out * C_in * 9;
    unsigned long long my_macs=0; float my_fire=9999.0f;
    for (int ch = 0; ch < NUM_CHUNKS; ++ch) {
        int base_t = ch * CHUNK_SIZE;
        float in_val[CHUNK_SIZE] = {0.0f};
        for (int ky = 0; ky < 3; ++ky) {
            int in_y = y+ky-1; if (in_y<0||in_y>=H) continue;
            for (int kx = 0; kx < 3; ++kx) {
                int in_x = x+kx-1; if (in_x<0||in_x>=W) continue;
                int site = (n*H + in_y)*W + in_x;
                int s = boff[site*OFF_STRIDE+ch], e = boff[site*OFF_STRIDE+ch+1];
                const int*   sc = bcin + (size_t)site*C_in;
                const float* st = btin + (size_t)site*C_in;
                int w_ky_base = w_out_base + ky*3 + kx;   // weight[((c_out*C_in+cin)*3+ky)*3+kx]
                for (int p = s; p < e; ++p) {
                    int cin = sc[p]; int tt = (int)st[p];
                    in_val[tt - base_t] += weight[w_ky_base + cin*9]; ++my_macs;
                }
            }
        }
        for (int t = 0; t < CHUNK_SIZE; ++t) {
            I = I*decay + in_val[t]; V += I;
            if (V >= th && my_fire==9999.0f) { my_fire=(float)(base_t+t);
#ifndef NOEARLYEXIT
                out_ft[idx]=my_fire; atomicAdd(&g_mac,my_macs); return;
#endif
            }
        }
    }
    out_ft[idx] = my_fire; atomicAdd(&g_mac, my_macs);
}

// =================== spiking max-pool (fan-in 4), chunked early-exit ==========
__global__ void pool_chunk_early(const float* in_ft, float* out_ft, int N, int C, int H_in, int W_in,
                                 float decay, float th) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int H_out = H_in/2, W_out = W_in/2;
    if (idx >= N*C*H_out*W_out) return;
    int n = idx/(C*H_out*W_out), rem = idx%(C*H_out*W_out);
    int c = rem/(H_out*W_out), yo = (rem/W_out)%H_out, xo = rem%W_out;
    float I = 0.0f, V = 0.0f;
    int base = (n*C + c)*H_in*W_in;
    for (int chunk = 0; chunk < T_TOTAL; chunk += CHUNK_SIZE) {
        float in_val[CHUNK_SIZE] = {0.0f};
        for (int dy=0; dy<2; ++dy) for (int dx=0; dx<2; ++dx) {
            float tf = in_ft[base + (yo*2+dy)*W_in + (xo*2+dx)];
            if (tf >= chunk && tf < chunk+CHUNK_SIZE) in_val[(int)tf - chunk] = 1.0f;
        }
        #pragma unroll
        for (int t = 0; t < CHUNK_SIZE; ++t) {
            I = I*decay + in_val[t]; V += I;
            if (V >= th) { out_ft[idx] = (float)(chunk + t); return; }
        }
    }
    out_ft[idx] = 9999.0f;
}

// =================== recurrent FC1: bucketized FF + sequential recurrence ======
__global__ void fc1_bucketed_recurrent(const int* bcin, const float* btin, const int* boff,
                                       float* out_ft, const float* w_fc1, const float* w_fc1_rec,
                                       int N, int in_features, int out_features, float decay, float th) {
    int n = blockIdx.x; if (n >= N) return;
    int i = threadIdx.x;                       // 128 neurons
    __shared__ bool fired_prev[128];
    fired_prev[i] = false;
    float I = 0.0f, V = 0.0f, my_fire = 9999.0f;
    bool has_fired = false;
    int w_offset = i * in_features, rec_offset = i * 128;
    const int*   sc = bcin + (size_t)n * in_features;
    const float* st = btin + (size_t)n * in_features;
    for (int ch = 0; ch < NUM_CHUNKS; ++ch) {
        int base_t = ch * CHUNK_SIZE;
        float in_ff[CHUNK_SIZE] = {0.0f};
        int s = boff[n*OFF_STRIDE+ch], e = boff[n*OFF_STRIDE+ch+1];
        for (int p = s; p < e; ++p) { int j = sc[p]; int tt = (int)st[p]; in_ff[tt - base_t] += w_fc1[w_offset + j]; }
        for (int t = 0; t < CHUNK_SIZE; ++t) {
            __syncthreads();
            float in_rec = 0.0f;
            for (int j = 0; j < 128; ++j) if (fired_prev[j]) in_rec += w_fc1_rec[rec_offset + j];
            I = I*decay + in_ff[t] + in_rec; V += I;
            bool fire_now = false;
            if (V >= th && !has_fired) { fire_now = true; has_fired = true; my_fire = (float)(base_t + t); V = -10000.0f; }
            __syncthreads();
            fired_prev[i] = fire_now;
        }
    }
    out_ft[n*out_features + i] = my_fire;
}

// =================== bucketized FC2 (output), chunked early-exit ===============
__global__ void fc2_bucketed_early(const int* bcin, const float* btin, const int* boff,
                                   float* out_ft, const float* w_fc2,
                                   int N, int in_features, int out_features, float decay, float th) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N * out_features) return;
    int n = idx / out_features, i = idx % out_features;
    float I = 0.0f, V = 0.0f;
    int w_offset = i * in_features;
    unsigned long long my_macs=0; float my_fire=9999.0f;
    const int*   sc = bcin + (size_t)n * in_features;
    const float* st = btin + (size_t)n * in_features;
    for (int ch = 0; ch < NUM_CHUNKS; ++ch) {
        int base_t = ch * CHUNK_SIZE;
        float in_val[CHUNK_SIZE] = {0.0f};
        int s = boff[n*OFF_STRIDE+ch], e = boff[n*OFF_STRIDE+ch+1];
        for (int p = s; p < e; ++p) { int j = sc[p]; int tt = (int)st[p]; in_val[tt - base_t] += w_fc2[w_offset + j]; }
        for (int t = 0; t < CHUNK_SIZE; ++t) {
            I = I*decay + in_val[t]; V += I;
            if (V >= th && my_fire==9999.0f) { my_fire=(float)(base_t+t);
#ifndef NOEARLYEXIT
                out_ft[idx]=my_fire; atomicAdd(&g_mac,my_macs); return;
#endif
            }
        }
    }
    out_ft[idx] = my_fire; atomicAdd(&g_mac, my_macs);
}

// ============================== host ========================================
void load_binary(const char* fp, void* host, size_t bytes) {
    std::ifstream f(fp, std::ios::binary);
    if (!f) { std::cerr << "Failed to open " << fp << std::endl; exit(1); }
    f.read(reinterpret_cast<char*>(host), bytes);
}

int main() {
    std::cout << ">>> VD-TTFS time-sorted (bucketized) chunked early-exit, DVSGesture..." << std::endl;
    int B = NUM_SAMPLES;
    size_t w_bytes = 1292032 * sizeof(float);
    float* h_w = new float[1292032];
    load_binary("cuda_assets/dvsgesture_weights.bin", h_w, w_bytes);
    float* d_w; CHECK_CUDA(cudaMalloc(&d_w, w_bytes)); CHECK_CUDA(cudaMemcpy(d_w, h_w, w_bytes, cudaMemcpyHostToDevice));
    float *w_conv1=d_w+0, *w_conv2=d_w+1152, *w_conv3=d_w+74880, *w_fc1_rec=d_w+222336, *w_fc1=d_w+238720, *w_fc2=d_w+1287296;

    size_t data_bytes = (size_t)B * 2*32*32*T_TOTAL * sizeof(float);
    float* h_data = new float[(size_t)B * 2*32*32*T_TOTAL];
    load_binary("cuda_assets/train_data.bin", h_data, data_bytes);
    float* d_data; CHECK_CUDA(cudaMalloc(&d_data, data_bytes)); CHECK_CUDA(cudaMemcpy(d_data, h_data, data_bytes, cudaMemcpyHostToDevice));
    int* h_labels = new int[B*NUM_CLASSES];
    load_binary("cuda_assets/train_labels.bin", h_labels, B*NUM_CLASSES*sizeof(int));

    int n_c1=B*64*32*32, n_c2=B*128*32*32, n_p1=B*128*16*16, n_c3=B*128*16*16, n_p2=B*128*8*8, n_fc1=B*128, n_fc2=B*10;
    float *d_c1,*d_c2,*d_p1,*d_c3,*d_p2,*d_fc1,*d_fc2;
    CHECK_CUDA(cudaMalloc(&d_c1, n_c1*4)); CHECK_CUDA(cudaMalloc(&d_c2, n_c2*4)); CHECK_CUDA(cudaMalloc(&d_p1, n_p1*4));
    CHECK_CUDA(cudaMalloc(&d_c3, n_c3*4)); CHECK_CUDA(cudaMalloc(&d_p2, n_p2*4));
    CHECK_CUDA(cudaMalloc(&d_fc1, n_fc1*4)); CHECK_CUDA(cudaMalloc(&d_fc2, n_fc2*4));

    // bucket buffers, sized for the largest single-spike input (conv2 input: B*64*32*32)
    size_t MAX_ELEMS = (size_t)B * 64 * 32 * 32;
    size_t MAX_SITES = (size_t)B * 32 * 32;
    int *d_bcin; float *d_btin; int *d_boff;
    CHECK_CUDA(cudaMalloc(&d_bcin, MAX_ELEMS*sizeof(int)));
    CHECK_CUDA(cudaMalloc(&d_btin, MAX_ELEMS*sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_boff, MAX_SITES*OFF_STRIDE*sizeof(int)));

    cudaEvent_t s,e; cudaEventCreate(&s); cudaEventCreate(&e);
    cudaDeviceSynchronize(); cudaEventRecord(s);

    // L1 conv1 (raw)
    conv1_raw_early<<<GET_BLOCKS(n_c1),256>>>(d_data, d_c1, w_conv1, B, decay_syn_conv, v_th_conv);
    // L2 conv2 (bucketized): input d_c1 [B,64,32,32]
    k_bucketize_conv<<<GET_BLOCKS(B*32*32),256>>>(d_c1, B,64,32,32, d_bcin,d_btin,d_boff);
    conv_bucketed_early<<<GET_BLOCKS(n_c2),256>>>(d_bcin,d_btin,d_boff, d_c2, w_conv2, B,64,128,32,32, decay_syn_conv, v_th_conv);
    // L3 pool1: input d_c2 [B,128,32,32] -> [B,128,16,16]
    pool_chunk_early<<<GET_BLOCKS(n_p1),256>>>(d_c2, d_p1, B,128,32,32, decay_syn_conv, v_th_conv);
    // L4 conv3 (bucketized): input d_p1 [B,128,16,16]
    k_bucketize_conv<<<GET_BLOCKS(B*16*16),256>>>(d_p1, B,128,16,16, d_bcin,d_btin,d_boff);
    conv_bucketed_early<<<GET_BLOCKS(n_c3),256>>>(d_bcin,d_btin,d_boff, d_c3, w_conv3, B,128,128,16,16, decay_syn_conv, v_th_conv);
    // L5 pool2: input d_c3 [B,128,16,16] -> [B,128,8,8]
    pool_chunk_early<<<GET_BLOCKS(n_p2),256>>>(d_c3, d_p2, B,128,16,16, decay_syn_conv, v_th_conv);
    // L6 fc1 (recurrent, bucketized FF): input d_p2 flattened [B,8192]
    k_bucketize_fc<<<GET_BLOCKS(B),256>>>(d_p2, B, 8192, d_bcin,d_btin,d_boff);
    fc1_bucketed_recurrent<<<B,128>>>(d_bcin,d_btin,d_boff, d_fc1, w_fc1, w_fc1_rec, B, 8192, 128, decay_syn_fc, v_th_fc);
    // L7 fc2 (bucketized): input d_fc1 [B,128]
    k_bucketize_fc<<<GET_BLOCKS(B),256>>>(d_fc1, B, 128, d_bcin,d_btin,d_boff);
    fc2_bucketed_early<<<GET_BLOCKS(n_fc2),256>>>(d_bcin,d_btin,d_boff, d_fc2, w_fc2, B, 128, 10, decay_syn_fc, v_th_fc);

    cudaEventRecord(e); cudaEventSynchronize(e);
    float ms=0; cudaEventElapsedTime(&ms,s,e);

    float* h_ft = new float[n_fc2];
    CHECK_CUDA(cudaMemcpy(h_ft, d_fc2, n_fc2*sizeof(float), cudaMemcpyDeviceToHost));
    int correct = 0;
    for (int i = 0; i < B; ++i) {
        int truth = 0, mg = h_labels[i*10];
        for (int c=1;c<10;++c) if (h_labels[i*10+c]>mg){ mg=h_labels[i*10+c]; truth=c; }
        int pred = 0; float mt = h_ft[i*10];
        for (int c=1;c<10;++c) if (h_ft[i*10+c]<mt){ mt=h_ft[i*10+c]; pred=c; }
        if (pred==truth) correct++;
    }
    std::cout << "=================================================" << std::endl;
    std::cout << "Inference time:  " << ms << " ms" << std::endl;
    std::cout << "Throughput:      " << (B*1000.0f)/ms << " samples/sec" << std::endl;
    std::cout << "Accuracy:        " << std::fixed << std::setprecision(2) << ((float)correct/B)*100.0f << " %" << std::endl;
    std::cout << "=================================================" << std::endl;
    unsigned long long mac=0; cudaMemcpyFromSymbol(&mac,g_mac,sizeof(mac));
    printf("BUCKET_MACS(conv2+conv3+fc2)= %llu\n",(unsigned long long)mac);
    return 0;
}
