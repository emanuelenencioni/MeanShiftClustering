#include "meanshift_cuda.h"
#include <cuda_runtime.h>
#include <cstdio>
#include <cstring>
#include <cmath>
#include <algorithm>
#include <chrono>

#define CUDA_CHECK(call)                                                       \
    do {                                                                       \
        cudaError_t _e = (call);                                               \
        if(_e != cudaSuccess) {                                                \
            std::fprintf(stderr, "CUDA error %s:%d: %s\n",                     \
                         __FILE__, __LINE__, cudaGetErrorString(_e));           \
            std::exit(1);                                                       \
        }                                                                       \
    } while(0)

__global__ void meanShiftKernel(
    const float* __restrict__ r_in, const float* __restrict__ g_in,
    const float* __restrict__ b_in, const float* __restrict__ x_in,
    const float* __restrict__ y_in,
    float* r_out, float* g_out, float* b_out,
    int n, float bw_sq, unsigned int* d_max_change_bits)
{
    extern __shared__ float sh[];
    float* sh_r = sh;
    float* sh_g = sh +     blockDim.x;
    float* sh_b = sh + 2 * blockDim.x;
    float* sh_x = sh + 3 * blockDim.x;
    float* sh_y = sh + 4 * blockDim.x;

    const int i  = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    const int tx = static_cast<int>(threadIdx.x);
    const int bs = static_cast<int>(blockDim.x);

    float src_r = 0.f, src_g = 0.f, src_b = 0.f, src_x = 0.f, src_y = 0.f;
    const bool active = (i < n);
    if(active) {
        src_r = r_in[i]; src_g = g_in[i]; src_b = b_in[i];
        src_x = x_in[i]; src_y = y_in[i];
    }

    float sum_r = 0.f, sum_g = 0.f, sum_b = 0.f;
    int count = 0;

    for(int tile_start = 0; tile_start < n; tile_start += bs) {
        int load_j = tile_start + tx;
        if(load_j < n) {
            sh_r[tx] = r_in[load_j];
            sh_g[tx] = g_in[load_j];
            sh_b[tx] = b_in[load_j];
            sh_x[tx] = x_in[load_j];
            sh_y[tx] = y_in[load_j];
        }
        __syncthreads();

        if(active) {
            int tile_end = min(bs, n - tile_start);
            for(int jj = 0; jj < tile_end; ++jj) {
                float dr = src_r - sh_r[jj];
                float dg = src_g - sh_g[jj];
                float db = src_b - sh_b[jj];
                float dx = src_x - sh_x[jj];
                float dy = src_y - sh_y[jj];
                float sq = dr*dr + dg*dg + db*db + dx*dx + dy*dy;
                if(sq <= bw_sq) {
                    sum_r += sh_r[jj];
                    sum_g += sh_g[jj];
                    sum_b += sh_b[jj];
                    count++;
                }
            }
        }
        __syncthreads();
    }

    if(active) {
        float new_r = (count > 0) ? sum_r / static_cast<float>(count) : src_r;
        float new_g = (count > 0) ? sum_g / static_cast<float>(count) : src_g;
        float new_b = (count > 0) ? sum_b / static_cast<float>(count) : src_b;

        float change = fmaxf(fabsf(new_r - src_r),
                      fmaxf(fabsf(new_g - src_g),
                            fabsf(new_b - src_b)));
        r_out[i] = new_r;
        g_out[i] = new_g;
        b_out[i] = new_b;

        if(!isnan(change))
            atomicMax(d_max_change_bits, __float_as_uint(change));
    }
}

MeanShiftResult meanShiftCUDA(std::vector<uint8_t>& data, int width, float bandwidth,
                               int max_iter, float tol,
                               bool show_pbar, int block_size)
{
    using clock = std::chrono::steady_clock;

    if(block_size != 128 && block_size != 256 && block_size != 512)
        block_size = 256;

    const int    n     = static_cast<int>(data.size() / 3);
    const float  bw_sq = bandwidth * bandwidth;
    const size_t bytes = static_cast<size_t>(n) * sizeof(float);

    auto t_conv_start = clock::now();
    std::vector<float> h_r(n), h_g(n), h_b(n), h_x(n), h_y(n);
    for(int i = 0; i < n; ++i) {
        h_r[i] = static_cast<float>(data[i * 3 + 0]);
        h_g[i] = static_cast<float>(data[i * 3 + 1]);
        h_b[i] = static_cast<float>(data[i * 3 + 2]);
        h_x[i] = static_cast<float>(i % width);
        h_y[i] = static_cast<float>(i / width);
    }

    float *d_r0, *d_g0, *d_b0;
    float *d_r1, *d_g1, *d_b1;
    float *d_x, *d_y;
    unsigned int* d_max_change_bits;

    CUDA_CHECK(cudaMalloc(&d_r0, bytes));
    CUDA_CHECK(cudaMalloc(&d_g0, bytes));
    CUDA_CHECK(cudaMalloc(&d_b0, bytes));
    CUDA_CHECK(cudaMalloc(&d_r1, bytes));
    CUDA_CHECK(cudaMalloc(&d_g1, bytes));
    CUDA_CHECK(cudaMalloc(&d_b1, bytes));
    CUDA_CHECK(cudaMalloc(&d_x,  bytes));
    CUDA_CHECK(cudaMalloc(&d_y,  bytes));
    CUDA_CHECK(cudaMalloc(&d_max_change_bits, sizeof(unsigned int)));

    CUDA_CHECK(cudaMemcpy(d_r0, h_r.data(), bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_g0, h_g.data(), bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_b0, h_b.data(), bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_x,  h_x.data(), bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_y,  h_y.data(), bytes, cudaMemcpyHostToDevice));

    auto t_conv_h2d_end = clock::now();
    double convert_ms = std::chrono::duration<double, std::milli>(
        t_conv_h2d_end - t_conv_start).count();

    float* r_in  = d_r0; float* g_in  = d_g0; float* b_in  = d_b0;
    float* r_out = d_r1; float* g_out = d_g1; float* b_out = d_b1;

    double total_shift_ms = 0.0;
    int iter = 0;
    std::vector<IterationInfo> iter_details;

    cudaEvent_t ev_start, ev_stop;
    CUDA_CHECK(cudaEventCreate(&ev_start));
    CUDA_CHECK(cudaEventCreate(&ev_stop));

    int grid = (n + block_size - 1) / block_size;
    size_t smem = 5 * static_cast<size_t>(block_size) * sizeof(float);

    for(; iter < max_iter; ++iter) {
        CUDA_CHECK(cudaMemset(d_max_change_bits, 0, sizeof(unsigned int)));

        CUDA_CHECK(cudaEventRecord(ev_start));
        meanShiftKernel<<<grid, block_size, smem>>>(
            r_in, g_in, b_in, d_x, d_y,
            r_out, g_out, b_out,
            n, bw_sq, d_max_change_bits);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaEventRecord(ev_stop));
        CUDA_CHECK(cudaEventSynchronize(ev_stop));

        float iter_ms_gpu = 0.f;
        CUDA_CHECK(cudaEventElapsedTime(&iter_ms_gpu, ev_start, ev_stop));
        total_shift_ms += static_cast<double>(iter_ms_gpu);

        unsigned int max_bits = 0;
        CUDA_CHECK(cudaMemcpy(&max_bits, d_max_change_bits,
                              sizeof(unsigned int), cudaMemcpyDeviceToHost));
        float max_change = 0.f;
        std::memcpy(&max_change, &max_bits, sizeof(float));

        iter_details.push_back({iter + 1,
                                static_cast<double>(iter_ms_gpu),
                                max_change});

        if(show_pbar)
            printProgressBar(iter + 1, max_iter, max_change);

        std::swap(r_in, r_out);
        std::swap(g_in, g_out);
        std::swap(b_in, b_out);

        if(max_change <= tol)
            break;
    }

    if(show_pbar)
        std::fprintf(stderr, "\n");

    CUDA_CHECK(cudaEventDestroy(ev_start));
    CUDA_CHECK(cudaEventDestroy(ev_stop));

    auto t_d2h_start = clock::now();

    CUDA_CHECK(cudaMemcpy(h_r.data(), r_in, bytes, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_g.data(), g_in, bytes, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_b.data(), b_in, bytes, cudaMemcpyDeviceToHost));

    for(int i = 0; i < n; ++i) {
        data[i * 3 + 0] = static_cast<uint8_t>(
            std::round(std::max(0.f, std::min(h_r[i], 255.f))));
        data[i * 3 + 1] = static_cast<uint8_t>(
            std::round(std::max(0.f, std::min(h_g[i], 255.f))));
        data[i * 3 + 2] = static_cast<uint8_t>(
            std::round(std::max(0.f, std::min(h_b[i], 255.f))));
    }

    auto t_d2h_end = clock::now();
    convert_ms += std::chrono::duration<double, std::milli>(
        t_d2h_end - t_d2h_start).count();

    cudaFree(d_r0); cudaFree(d_g0); cudaFree(d_b0);
    cudaFree(d_r1); cudaFree(d_g1); cudaFree(d_b1);
    cudaFree(d_x);  cudaFree(d_y);
    cudaFree(d_max_change_bits);

    return MeanShiftResult{static_cast<int>(iter_details.size()),
                           total_shift_ms, convert_ms, iter_details};
}

/* ─── Non-tiled kernel (direct global reads, no shared memory) ───────────── */

__global__ void meanShiftKernelNoTile(
    const float* __restrict__ r_in, const float* __restrict__ g_in,
    const float* __restrict__ b_in, const float* __restrict__ x_in,
    const float* __restrict__ y_in,
    float* r_out, float* g_out, float* b_out,
    int n, float bw_sq, unsigned int* d_max_change_bits)
{
    const int i  = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    const bool active = (i < n);

    float src_r = 0.f, src_g = 0.f, src_b = 0.f, src_x = 0.f, src_y = 0.f;
    if(active) {
        src_r = r_in[i]; src_g = g_in[i]; src_b = b_in[i];
        src_x = x_in[i]; src_y = y_in[i];
    }

    float sum_r = 0.f, sum_g = 0.f, sum_b = 0.f;
    int count = 0;

    if(active) {
        for(int j = 0; j < n; ++j) {
            float dr = src_r - r_in[j];
            float dg = src_g - g_in[j];
            float db = src_b - b_in[j];
            float dx = src_x - x_in[j];
            float dy = src_y - y_in[j];
            float sq = dr*dr + dg*dg + db*db + dx*dx + dy*dy;
            if(sq <= bw_sq) {
                sum_r += r_in[j];
                sum_g += g_in[j];
                sum_b += b_in[j];
                count++;
            }
        }
    }

    if(active) {
        float new_r = (count > 0) ? sum_r / static_cast<float>(count) : src_r;
        float new_g = (count > 0) ? sum_g / static_cast<float>(count) : src_g;
        float new_b = (count > 0) ? sum_b / static_cast<float>(count) : src_b;

        float change = fmaxf(fabsf(new_r - src_r),
                      fmaxf(fabsf(new_g - src_g),
                            fabsf(new_b - src_b)));
        r_out[i] = new_r;
        g_out[i] = new_g;
        b_out[i] = new_b;

        if(!isnan(change))
            atomicMax(d_max_change_bits, __float_as_uint(change));
    }
}

MeanShiftResult meanShiftCUDANoTile(std::vector<uint8_t>& data, int width,
                                     float bandwidth, int max_iter, float tol,
                                     bool show_pbar, int block_size)
{
    using clock = std::chrono::steady_clock;

    if(block_size != 128 && block_size != 256 && block_size != 512)
        block_size = 256;

    const int    n     = static_cast<int>(data.size() / 3);
    const float  bw_sq = bandwidth * bandwidth;
    const size_t bytes = static_cast<size_t>(n) * sizeof(float);

    auto t_conv_start = clock::now();
    std::vector<float> h_r(n), h_g(n), h_b(n), h_x(n), h_y(n);
    for(int i = 0; i < n; ++i) {
        h_r[i] = static_cast<float>(data[i * 3 + 0]);
        h_g[i] = static_cast<float>(data[i * 3 + 1]);
        h_b[i] = static_cast<float>(data[i * 3 + 2]);
        h_x[i] = static_cast<float>(i % width);
        h_y[i] = static_cast<float>(i / width);
    }

    float *d_r0, *d_g0, *d_b0;
    float *d_r1, *d_g1, *d_b1;
    float *d_x, *d_y;
    unsigned int* d_max_change_bits;

    CUDA_CHECK(cudaMalloc(&d_r0, bytes));
    CUDA_CHECK(cudaMalloc(&d_g0, bytes));
    CUDA_CHECK(cudaMalloc(&d_b0, bytes));
    CUDA_CHECK(cudaMalloc(&d_r1, bytes));
    CUDA_CHECK(cudaMalloc(&d_g1, bytes));
    CUDA_CHECK(cudaMalloc(&d_b1, bytes));
    CUDA_CHECK(cudaMalloc(&d_x,  bytes));
    CUDA_CHECK(cudaMalloc(&d_y,  bytes));
    CUDA_CHECK(cudaMalloc(&d_max_change_bits, sizeof(unsigned int)));

    CUDA_CHECK(cudaMemcpy(d_r0, h_r.data(), bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_g0, h_g.data(), bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_b0, h_b.data(), bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_x,  h_x.data(), bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_y,  h_y.data(), bytes, cudaMemcpyHostToDevice));

    auto t_conv_h2d_end = clock::now();
    double convert_ms = std::chrono::duration<double, std::milli>(
        t_conv_h2d_end - t_conv_start).count();

    float* r_in  = d_r0; float* g_in  = d_g0; float* b_in  = d_b0;
    float* r_out = d_r1; float* g_out = d_g1; float* b_out = d_b1;

    double total_shift_ms = 0.0;
    int iter = 0;
    std::vector<IterationInfo> iter_details;

    cudaEvent_t ev_start, ev_stop;
    CUDA_CHECK(cudaEventCreate(&ev_start));
    CUDA_CHECK(cudaEventCreate(&ev_stop));

    int grid = (n + block_size - 1) / block_size;

    for(; iter < max_iter; ++iter) {
        CUDA_CHECK(cudaMemset(d_max_change_bits, 0, sizeof(unsigned int)));

        CUDA_CHECK(cudaEventRecord(ev_start));
        meanShiftKernelNoTile<<<grid, block_size>>>(
            r_in, g_in, b_in, d_x, d_y,
            r_out, g_out, b_out,
            n, bw_sq, d_max_change_bits);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaEventRecord(ev_stop));
        CUDA_CHECK(cudaEventSynchronize(ev_stop));

        float iter_ms_gpu = 0.f;
        CUDA_CHECK(cudaEventElapsedTime(&iter_ms_gpu, ev_start, ev_stop));
        total_shift_ms += static_cast<double>(iter_ms_gpu);

        unsigned int max_bits = 0;
        CUDA_CHECK(cudaMemcpy(&max_bits, d_max_change_bits,
                              sizeof(unsigned int), cudaMemcpyDeviceToHost));
        float max_change = 0.f;
        std::memcpy(&max_change, &max_bits, sizeof(float));

        iter_details.push_back({iter + 1,
                                static_cast<double>(iter_ms_gpu),
                                max_change});

        if(show_pbar)
            printProgressBar(iter + 1, max_iter, max_change);

        std::swap(r_in, r_out);
        std::swap(g_in, g_out);
        std::swap(b_in, b_out);

        if(max_change <= tol)
            break;
    }

    if(show_pbar)
        std::fprintf(stderr, "\n");

    CUDA_CHECK(cudaEventDestroy(ev_start));
    CUDA_CHECK(cudaEventDestroy(ev_stop));

    auto t_d2h_start = clock::now();

    CUDA_CHECK(cudaMemcpy(h_r.data(), r_in, bytes, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_g.data(), g_in, bytes, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_b.data(), b_in, bytes, cudaMemcpyDeviceToHost));

    for(int i = 0; i < n; ++i) {
        data[i * 3 + 0] = static_cast<uint8_t>(
            std::round(std::max(0.f, std::min(h_r[i], 255.f))));
        data[i * 3 + 1] = static_cast<uint8_t>(
            std::round(std::max(0.f, std::min(h_g[i], 255.f))));
        data[i * 3 + 2] = static_cast<uint8_t>(
            std::round(std::max(0.f, std::min(h_b[i], 255.f))));
    }

    auto t_d2h_end = clock::now();
    convert_ms += std::chrono::duration<double, std::milli>(
        t_d2h_end - t_d2h_start).count();

    cudaFree(d_r0); cudaFree(d_g0); cudaFree(d_b0);
    cudaFree(d_r1); cudaFree(d_g1); cudaFree(d_b1);
    cudaFree(d_x);  cudaFree(d_y);
    cudaFree(d_max_change_bits);

    return MeanShiftResult{static_cast<int>(iter_details.size()),
                           total_shift_ms, convert_ms, iter_details};
}
