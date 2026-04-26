#include "meanshift_omp_soa.h"
#include <omp.h>
#include <chrono>
#include <algorithm>
#include <cstdio>

// Parallel SoA mean shift using OpenMP.
// Outer loop over pixels is parallelised; Jacobi semantics guarantee no races.
// next_r/g/b are hoisted outside the iteration loop to avoid repeated allocation.
MeanShiftResult meanShiftSoAOMP(std::vector<uint8_t>& data, int width, float bandwidth,
                                int max_iter, float tol, bool show_pbar, KernelFn kernel) {
    using clock = std::chrono::steady_clock;

    if(!kernel) kernel = makeKernel("flat");

    auto t_conv_start = clock::now();
    std::vector<float> current;
    convertToFloat(data, current);

    ImageSoA soa;
    convertToFloatSoA(current, soa, width);
    auto t_conv_end = clock::now();
    double convert_ms = std::chrono::duration<double, std::milli>(t_conv_end - t_conv_start).count();

    const float bandwidth_sq = bandwidth * bandwidth;
    int n_pixels = soa.n;

    // Hoist next buffers outside the iteration loop — allocate once, reuse every iteration.
    std::vector<float> next_r(n_pixels);
    std::vector<float> next_g(n_pixels);
    std::vector<float> next_b(n_pixels);

    double total_shift_ms = 0.0;
    int iter = 0;
    std::vector<IterationInfo> iter_details;

    for(; iter < max_iter; ++iter) {
        float max_change = 0.0f;

        auto t_shift_start = clock::now();

        #pragma omp parallel for schedule(static) reduction(max:max_change)
        for(int i = 0; i < n_pixels; ++i) {
            NeighborAccumulator acc;

            for(int j = 0; j < n_pixels; ++j) {
                float w = kernel(squaredDistanceSoA(soa, i, j), bandwidth_sq);
                if(w > 0.0f) {
                    acc.sum_r += w * soa.r[j];
                    acc.sum_g += w * soa.g[j];
                    acc.sum_b += w * soa.b[j];
                    acc.weight_sum += w;
                }
            }

            float change = 0.0f;
            if(acc.weight_sum > 0.0f) {
                float new_r = acc.sum_r / acc.weight_sum;
                float new_g = acc.sum_g / acc.weight_sum;
                float new_b = acc.sum_b / acc.weight_sum;

                float dr = new_r - soa.r[i];
                float dg = new_g - soa.g[i];
                float db = new_b - soa.b[i];

                change = std::max({std::abs(dr), std::abs(dg), std::abs(db)});

                next_r[i] = new_r;
                next_g[i] = new_g;
                next_b[i] = new_b;
            } else {
                next_r[i] = soa.r[i];
                next_g[i] = soa.g[i];
                next_b[i] = soa.b[i];
            }

            if(change > max_change)
                max_change = change;
        }

        auto t_shift_end = clock::now();
        double iter_ms = std::chrono::duration<double, std::milli>(t_shift_end - t_shift_start).count();
        total_shift_ms += iter_ms;
        iter_details.push_back({iter + 1, iter_ms, max_change});

        soa.r.swap(next_r);
        soa.g.swap(next_g);
        soa.b.swap(next_b);

        if(show_pbar)
            printProgressBar(iter + 1, max_iter, max_change);

        if(max_change <= tol)
            break;
    }

    if(show_pbar)
        std::fprintf(stderr, "\n");

    auto t_conv_out_start = clock::now();
    convertFromFloatSoA(soa, current);
    convertFromFloat(current, data);
    auto t_conv_out_end = clock::now();
    convert_ms += std::chrono::duration<double, std::milli>(t_conv_out_end - t_conv_out_start).count();

    return MeanShiftResult{static_cast<int>(iter_details.size()), total_shift_ms, convert_ms, iter_details};
}
