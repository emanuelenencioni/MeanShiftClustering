#include "meanshift.h"
#include <chrono>
#include <algorithm>
#include <cstdio>

static void printProgressBar(int iter, int max_iter, float max_change) {
    const int bar_width = 30;
    float progress = static_cast<float>(iter) / max_iter;
    int filled = static_cast<int>(progress * bar_width);

    std::fprintf(stderr, "\rIter [%3d/%3d] [", iter, max_iter);
    for(int i = 0; i < bar_width; ++i)
        std::fputc(i < filled ? '#' : '.', stderr);
    std::fprintf(stderr, "] max_change=%7.3f", max_change);
    std::fflush(stderr);
}

float squaredDistance(const float* a, const float* b) {
    float sum = 0.0f;
    for(int k = 0; k < 3; ++k) {
        float diff = a[k] - b[k];
        sum += diff * diff;
    }
    return sum;
}

static void convertToFloat(const std::vector<uint8_t>& data, std::vector<float>& out) {
    out.resize(data.size());
    for(size_t i = 0; i < data.size(); ++i)
        out[i] = static_cast<float>(data[i]);
}

static void convertFromFloat(const std::vector<float>& current, std::vector<uint8_t>& data) {
    for(size_t i = 0; i < data.size(); ++i) {
        float val = std::max(0.0f, std::min(current[i], 255.0f));
        data[i] = static_cast<uint8_t>(std::round(val));
    }
}

// Brute-force: O(n^2) per iteration
MeanShiftResult meanShift(std::vector<uint8_t>& data, float bandwidth,
                          int max_iter, float tol, bool show_pbar) {
    using clock = std::chrono::steady_clock;

    std::vector<float> current;
    convertToFloat(data, current);

    const float bandwidth_sq = bandwidth * bandwidth;
    const int n_pixels = static_cast<int>(current.size() / 3);

    double total_shift_ms = 0.0;
    int iter = 0;

    for(; iter < max_iter; ++iter) {
        std::vector<float> next(current.size());
        float max_change = 0.0f;

        auto t_shift_start = clock::now();

        for(int i = 0; i < n_pixels; ++i) {
            const float* src = &current[i * 3];
            float sum[3] = {0.0f, 0.0f, 0.0f};
            int count = 0;

            for(int j = 0; j < n_pixels; ++j) {
                const float* neighbor = &current[j * 3];
                if(squaredDistance(src, neighbor) <= bandwidth_sq) {
                    sum[0] += neighbor[0];
                    sum[1] += neighbor[1];
                    sum[2] += neighbor[2];
                    count++;
                }
            }

            float change = 0.0f;
            if(count > 0) {
                for(int k = 0; k < 3; ++k) {
                    float new_val = sum[k] / count;
                    change = std::max(change, std::abs(new_val - src[k]));
                    next[i * 3 + k] = new_val;
                }
            } else {
                for(int k = 0; k < 3; ++k)
                    next[i * 3 + k] = src[k];
            }

            if(change > max_change)
                max_change = change;
        }

        auto t_shift_end = clock::now();
        total_shift_ms += std::chrono::duration<double, std::milli>(t_shift_end - t_shift_start).count();

        current.swap(next);

        if(show_pbar)
            printProgressBar(iter + 1, max_iter, max_change);

        if(max_change <= tol)
            break;
    }

    if(show_pbar)
        std::fprintf(stderr, "\n");

    convertFromFloat(current, data);

    return MeanShiftResult{iter + 1, 0.0, total_shift_ms};
}

// Grid-accelerated: O(n) amortized per iteration via spatial hashing
MeanShiftResult meanShiftOptimized(std::vector<uint8_t>& data, float bandwidth,
                                   int max_iter, float tol, bool show_pbar) {
    using clock = std::chrono::steady_clock;

    std::vector<float> current;
    convertToFloat(data, current);

    const float bandwidth_sq = bandwidth * bandwidth;
    const int n_pixels = static_cast<int>(current.size() / 3);

    double total_grid_ms = 0.0;
    double total_shift_ms = 0.0;
    int iter = 0;
    // Probably PRAGMA here for parallelization in future
    for(; iter < max_iter; ++iter) {
        std::vector<float> next(current.size());
        float max_change = 0.0f;

        auto t_grid_start = clock::now();

        std::unordered_map<Bin, std::vector<int>> grid;
        // maybe parallelize this loop in future, but need to be careful about concurrent writes to grid
        for(int j = 0; j < n_pixels; ++j) {
            const float* p = &current[j * 3];
            Bin b{static_cast<int>(p[0] / bandwidth),
                  static_cast<int>(p[1] / bandwidth),
                  static_cast<int>(p[2] / bandwidth)};
            grid[b].push_back(j);
        }

        auto t_grid_end = clock::now();
        total_grid_ms += std::chrono::duration<double, std::milli>(t_grid_end - t_grid_start).count();

        auto t_shift_start = clock::now();
        // parallelize this loop in future, but need to be careful about concurrent writes to next and max_change
        for(int i = 0; i < n_pixels; ++i) {
            const float* src = &current[i * 3];
            float sum[3] = {0.0f, 0.0f, 0.0f};
            int count = 0;

            Bin b{static_cast<int>(src[0] / bandwidth),
                  static_cast<int>(src[1] / bandwidth),
                  static_cast<int>(src[2] / bandwidth)};

            for(int dx = -1; dx <= 1; ++dx) {
                for(int dy = -1; dy <= 1; ++dy) {
                    for(int dz = -1; dz <= 1; ++dz) {
                        Bin nb{b.x + dx, b.y + dy, b.z + dz};
                        auto it = grid.find(nb);
                        if(it != grid.end()) {
                            for(int j : it->second) {
                                const float* neighbor = &current[j * 3];
                                if(squaredDistance(src, neighbor) <= bandwidth_sq) {
                                    sum[0] += neighbor[0];
                                    sum[1] += neighbor[1];
                                    sum[2] += neighbor[2];
                                    count++;
                                }
                            }
                        }
                    }
                }
            }

            float change = 0.0f;
            if(count > 0) {
                for(int k = 0; k < 3; ++k) {
                    float new_val = sum[k] / count;
                    change = std::max(change, std::abs(new_val - src[k]));
                    next[i * 3 + k] = new_val;
                }
            } else {
                for(int k = 0; k < 3; ++k)
                    next[i * 3 + k] = src[k];
            }

            if(change > max_change)
                max_change = change;
        }

        auto t_shift_end = clock::now();
        total_shift_ms += std::chrono::duration<double, std::milli>(t_shift_end - t_shift_start).count();

        current.swap(next);

        if(show_pbar)
            printProgressBar(iter + 1, max_iter, max_change);

        if(max_change <= tol)
            break;
    }

    if(show_pbar)
        std::fprintf(stderr, "\n");

    convertFromFloat(current, data);

    return MeanShiftResult{iter + 1, total_grid_ms, total_shift_ms};
}
