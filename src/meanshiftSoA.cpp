#include "meanshiftSoA.h"
#include <chrono>
#include <algorithm>
#include <cstdio>

// Helper to convert AoS float buffer to SoA representation
void convertToFloatSoA(const std::vector<float>& current, PixelSoA& soa, int width) {
    soa.n = static_cast<int>(current.size() / 3);
    soa.r.resize(soa.n);
    soa.g.resize(soa.n);
    soa.b.resize(soa.n);
    soa.x.resize(soa.n);
    soa.y.resize(soa.n);

    for(int i = 0; i < soa.n; ++i) {
        soa.r[i] = current[i * 3 + 0];
        soa.g[i] = current[i * 3 + 1];
        soa.b[i] = current[i * 3 + 2];
        soa.x[i] = static_cast<float>(i % width);
        soa.y[i] = static_cast<float>(i / width);
    }
}

// Helper to convert SoA back to AoS float buffer
void convertFromFloatSoA(const PixelSoA& soa, std::vector<float>& current) {
    current.resize(soa.n * 3);
    for(int i = 0; i < soa.n; ++i) {
        current[i * 3 + 0] = std::max(0.0f, std::min(soa.r[i], 255.0f));
        current[i * 3 + 1] = std::max(0.0f, std::min(soa.g[i], 255.0f));
        current[i * 3 + 2] = std::max(0.0f, std::min(soa.b[i], 255.0f));
    }
}

// 5D squared distance between pixels i and j within the same SoA struct
float squaredDistanceSoA(const PixelSoA& soa, int i, int j) {
    float dcol = soa.x[i] - soa.x[j];
    float drow = soa.y[i] - soa.y[j];
    float dr = soa.r[i] - soa.r[j];
    float dg = soa.g[i] - soa.g[j];
    float db = soa.b[i] - soa.b[j];
    return dcol * dcol + drow * drow + dr * dr + dg * dg + db * db;
}

// Brute-force mean shift using SoA, 5D feature space (x, y, R, G, B)
MeanShiftResult meanShiftSoA(std::vector<uint8_t>& data, int width, float bandwidth,
                             int max_iter, float tol, bool show_pbar) {
    using clock = std::chrono::steady_clock;

    std::vector<float> current;
    convertToFloat(data, current);

    PixelSoA soa;
    convertToFloatSoA(current, soa, width);

    const float bandwidth_sq = bandwidth * bandwidth;
    int n_pixels = soa.n;

    double total_shift_ms = 0.0;
    int iter = 0;
    std::vector<IterationInfo> iter_details;

    for(; iter < max_iter; ++iter) {
        std::vector<float> next_r(n_pixels);
        std::vector<float> next_g(n_pixels);
        std::vector<float> next_b(n_pixels);
        float max_change = 0.0f;

        auto t_shift_start = clock::now();

        for(int i = 0; i < n_pixels; ++i) {
            NeighborAccumulator acc;

            for(int j = 0; j < n_pixels; ++j) {
                if(squaredDistanceSoA(soa, i, j) <= bandwidth_sq) {
                    acc.sum_r += soa.r[j];
                    acc.sum_g += soa.g[j];
                    acc.sum_b += soa.b[j];
                    acc.count++;
                }
            }

            float change = 0.0f;
            if(acc.count > 0) {
                float new_r = acc.sum_r / acc.count;
                float new_g = acc.sum_g / acc.count;
                float new_b = acc.sum_b / acc.count;

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

    convertFromFloatSoA(soa, current);
    convertFromFloat(current, data);

    return MeanShiftResult{static_cast<int>(iter_details.size()), total_shift_ms, iter_details};
}
