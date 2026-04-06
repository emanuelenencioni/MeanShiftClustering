#include "meanshift_baseline.h"
#include <chrono>
#include <algorithm>
#include <cstdio>
#include <cmath>

/* 5D squared distance between pixels i and j using flat stride-3 float buffer.
 * Spatial coords (x, y) are recomputed from the pixel index on every call —
 * this is the deliberate inefficiency that the seq version eliminates. */
static float squaredDistanceBaseline(const std::vector<float>& buf, int i, int j, int width) {
    float dx = static_cast<float>(i % width) - static_cast<float>(j % width);
    float dy = static_cast<float>(i / width) - static_cast<float>(j / width);
    float dr = buf[i * 3 + 0] - buf[j * 3 + 0];
    float dg = buf[i * 3 + 1] - buf[j * 3 + 1];
    float db = buf[i * 3 + 2] - buf[j * 3 + 2];
    return dx*dx + dy*dy + dr*dr + dg*dg + db*db;
}

/* Naive brute-force mean shift.
 * Operates directly on image.rgb_image (raw uint8_t*, RGB, stride 3).
 * - x, y recomputed via i%width / i/width inside the O(n^2) inner loop.
 * - next buffer allocated fresh each iteration (no hoisting).
 * - Jacobi update: reads from current[], writes to next[], then swaps.
 * Results written back to image.rgb_image in place. */
MeanShiftResult meanShiftBaseline(STBImage& image, float bandwidth,
                                  int max_iter, float tol, bool show_pbar) {
    using clock = std::chrono::steady_clock;

    const int width = image.width;
    const int n_pixels = image.width * image.height;
    uint8_t* raw = image.rgb_image;

    // Load raw uint8_t* into float working buffer
    std::vector<float> current(n_pixels * 3);
    for(int i = 0; i < n_pixels * 3; ++i)
        current[i] = static_cast<float>(raw[i]);

    const float bandwidth_sq = bandwidth * bandwidth;

    double total_shift_ms = 0.0;
    int iter = 0;
    std::vector<IterationInfo> iter_details;

    for(; iter < max_iter; ++iter) {
        std::vector<float> next(n_pixels * 3);
        float max_change = 0.0f;

        auto t_shift_start = clock::now();

        for(int i = 0; i < n_pixels; ++i) {
            float sum_r = 0.0f, sum_g = 0.0f, sum_b = 0.0f;
            int count = 0;

            for(int j = 0; j < n_pixels; ++j) {
                if(squaredDistanceBaseline(current, i, j, width) <= bandwidth_sq) {
                    sum_r += current[j * 3 + 0];
                    sum_g += current[j * 3 + 1];
                    sum_b += current[j * 3 + 2];
                    count++;
                }
            }

            float change = 0.0f;
            if(count > 0) {
                float new_r = sum_r / count;
                float new_g = sum_g / count;
                float new_b = sum_b / count;
                change = std::max({std::abs(new_r - current[i * 3 + 0]),
                                   std::abs(new_g - current[i * 3 + 1]),
                                   std::abs(new_b - current[i * 3 + 2])});
                next[i * 3 + 0] = new_r;
                next[i * 3 + 1] = new_g;
                next[i * 3 + 2] = new_b;
            } else {
                next[i * 3 + 0] = current[i * 3 + 0];
                next[i * 3 + 1] = current[i * 3 + 1];
                next[i * 3 + 2] = current[i * 3 + 2];
            }

            if(change > max_change)
                max_change = change;
        }

        auto t_shift_end = clock::now();
        double iter_ms = std::chrono::duration<double, std::milli>(t_shift_end - t_shift_start).count();
        total_shift_ms += iter_ms;
        iter_details.push_back({iter + 1, iter_ms, max_change});

        current.swap(next);

        if(show_pbar)
            printProgressBar(iter + 1, max_iter, max_change);

        if(max_change <= tol)
            break;
    }

    if(show_pbar)
        std::fprintf(stderr, "\n");

    // Write results back to raw uint8_t* buffer
    for(int i = 0; i < n_pixels * 3; ++i) {
        float val = std::max(0.0f, std::min(current[i], 255.0f));
        raw[i] = static_cast<uint8_t>(std::round(val));
    }

    return MeanShiftResult{static_cast<int>(iter_details.size()), total_shift_ms, iter_details};
}
