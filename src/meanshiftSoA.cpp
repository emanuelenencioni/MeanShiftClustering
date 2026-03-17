#include "meanshiftSoA.h"
#include <chrono>
#include <algorithm>
#include <cstdio>

// Helper to convert AoS float buffer to SoA representation
void convertToFloatSoA(const std::vector<float>& current, PixelSoA& soa) {
    soa.n = static_cast<int>(current.size() / 3);
    soa.r.resize(soa.n);
    soa.g.resize(soa.n);
    soa.b.resize(soa.n);

    for(int i = 0; i < soa.n; ++i) {
        soa.r[i] = current[i * 3 + 0];
        soa.g[i] = current[i * 3 + 1];
        soa.b[i] = current[i * 3 + 2];
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

// SoA squared distance between pixels i and j within the same SoA struct
float squaredDistanceSoA(const PixelSoA& soa, int i, int j) {
    float dr = soa.r[i] - soa.r[j];
    float dg = soa.g[i] - soa.g[j];
    float db = soa.b[i] - soa.b[j];
    return dr * dr + dg * dg + db * db;
}

// Brute-force mean shift using SoA
MeanShiftResult meanShiftSoA(std::vector<uint8_t>& data, float bandwidth,
                             int max_iter, float tol, bool show_pbar) {
    using clock = std::chrono::steady_clock;

    std::vector<float> current;
    convertToFloat(data, current);

    PixelSoA soa;
    convertToFloatSoA(current, soa);

    const float bandwidth_sq = bandwidth * bandwidth;
    int n_pixels = soa.n;

    double total_shift_ms = 0.0;
    int iter = 0;
    std::vector<IterationInfo> iter_details;

    for(; iter < max_iter; ++iter) {
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

                soa.r[i] = new_r;
                soa.g[i] = new_g;
                soa.b[i] = new_b;
            }

            if(change > max_change)
                max_change = change;
        }

        auto t_shift_end = clock::now();
        double iter_ms = std::chrono::duration<double, std::milli>(t_shift_end - t_shift_start).count();
        total_shift_ms += iter_ms;
        iter_details.push_back({iter + 1, iter_ms, max_change});

        if(show_pbar)
            printProgressBar(iter + 1, max_iter, max_change);

        if(max_change <= tol)
            break;
    }

    if(show_pbar)
        std::fprintf(stderr, "\n");

    convertFromFloatSoA(soa, current);
    convertFromFloat(current, data);

    return MeanShiftResult{static_cast<int>(iter_details.size()), 0.0, total_shift_ms, iter_details};
}

// Grid-accelerated mean shift using SoA
MeanShiftResult meanShiftSoAOptimized(std::vector<uint8_t>& data, float bandwidth,
                                      int max_iter, float tol, bool show_pbar) {
    using clock = std::chrono::steady_clock;

    std::vector<float> current;
    convertToFloat(data, current);

    PixelSoA soa;
    convertToFloatSoA(current, soa);

    const float bandwidth_sq = bandwidth * bandwidth;
    const int n_pixels = soa.n;

    double total_grid_ms = 0.0;
    double total_shift_ms = 0.0;
    int iter = 0;
    std::vector<IterationInfo> iter_details;

    for(; iter < max_iter; ++iter) {
        std::vector<float> next_r(n_pixels);
        std::vector<float> next_g(n_pixels);
        std::vector<float> next_b(n_pixels);
        float max_change = 0.0f;

        auto t_iter_start = clock::now();
        auto t_grid_start = clock::now();

        std::unordered_map<Bin, std::vector<int>> grid;
        for(int j = 0; j < n_pixels; ++j) {
            Bin b{static_cast<int>(soa.r[j] / bandwidth),
                  static_cast<int>(soa.g[j] / bandwidth),
                  static_cast<int>(soa.b[j] / bandwidth)};
            grid[b].push_back(j);
        }

        auto t_grid_end = clock::now();
        total_grid_ms += std::chrono::duration<double, std::milli>(t_grid_end - t_grid_start).count();

        auto t_shift_start = clock::now();

        for(int i = 0; i < n_pixels; ++i) {
            NeighborAccumulator acc;

            Bin b{static_cast<int>(soa.r[i] / bandwidth),
                  static_cast<int>(soa.g[i] / bandwidth),
                  static_cast<int>(soa.b[i] / bandwidth)};

            for(int dx = -1; dx <= 1; ++dx) {
                for(int dy = -1; dy <= 1; ++dy) {
                    for(int dz = -1; dz <= 1; ++dz) {
                        Bin nb{b.x + dx, b.y + dy, b.z + dz};
                        auto it = grid.find(nb);
                        if(it != grid.end()) {
                            for(int j : it->second) {
                                if(squaredDistanceSoA(soa, i, j) <= bandwidth_sq) {
                                    acc.sum_r += soa.r[j];
                                    acc.sum_g += soa.g[j];
                                    acc.sum_b += soa.b[j];
                                    acc.count++;
                                }
                            }
                        }
                    }
                }
            }

            float change = 0.0f;
            if(acc.count > 0) {
                float new_r = acc.sum_r / acc.count;
                float new_g = acc.sum_g / acc.count;
                float new_b = acc.sum_b / acc.count;

                change = std::max({std::abs(new_r - soa.r[i]),
                                   std::abs(new_g - soa.g[i]),
                                   std::abs(new_b - soa.b[i])});

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
        total_shift_ms += std::chrono::duration<double, std::milli>(t_shift_end - t_shift_start).count();

        double iter_ms = std::chrono::duration<double, std::milli>(t_shift_end - t_iter_start).count();
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

    return MeanShiftResult{static_cast<int>(iter_details.size()), total_grid_ms, total_shift_ms, iter_details};
}
