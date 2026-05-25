#include "meanshift_baseline.h"
#include <chrono>
#include <algorithm>
#include <cstdio>
#include <cmath>
#include <iostream>

/* --- Shared utility functions -------------------------------------------- */

void convertToFloat(const std::vector<uint8_t>& data, std::vector<float>& out) {
    out.resize(data.size());
    for(size_t i = 0; i < data.size(); ++i)
        out[i] = static_cast<float>(data[i]);
}

void convertFromFloat(const std::vector<float>& current, std::vector<uint8_t>& data) {
    for(size_t i = 0; i < data.size(); ++i) {
        float val = std::max(0.0f, std::min(current[i], 255.0f));
        data[i] = static_cast<uint8_t>(std::round(val));
    }
}

void printProgressBar(int iter, int max_iter, float max_change) {
    const int bar_width = 30;
    float progress = static_cast<float>(iter) / max_iter;
    int filled = static_cast<int>(progress * bar_width);

    std::fprintf(stderr, "\rIter [%3d/%3d] [", iter, max_iter);
    for(int i = 0; i < bar_width; ++i)
        std::fputc(i < filled ? '#' : '.', stderr);
    std::fprintf(stderr, "] max_change=%7.3f", max_change);
    std::fflush(stderr);
}

/* --- Baseline algorithm -------------------------------------------------- */

/* 5D squared distance between pixels i and j using stride-5 float buffer.
 * Layout per pixel: [r, g, b, x, y]. Flat kernel: returns true if within bandwidth. */
static inline bool withinBandwidthBaseline(const std::vector<float>& buf,
                                           int i, int j, float bw_sq) {
    float dr = buf[i * 5 + 0] - buf[j * 5 + 0];
    float dg = buf[i * 5 + 1] - buf[j * 5 + 1];
    float db = buf[i * 5 + 2] - buf[j * 5 + 2];
    float dx = buf[i * 5 + 3] - buf[j * 5 + 3];
    float dy = buf[i * 5 + 4] - buf[j * 5 + 4];
    return dx*dx + dy*dy + dr*dr + dg*dg + db*db <= bw_sq;
}

/* Naive brute-force mean shift.
 * Operates directly on image.rgb_image (raw uint8_t*, RGB, stride 3).
 * - x, y precomputed once into the working buffer (stride 5: r,g,b,x,y).
 * - next buffer allocated fresh each iteration (no hoisting).
 * - Flat kernel: all neighbors within bandwidth contribute equally.
 * - Jacobi update: reads from current[], writes to next[], then swaps.
 * Results written back to image.rgb_image in place. */
MeanShiftResult meanShiftBaseline(STBImage& image, float bandwidth,
                                  int max_iter, float tol, bool show_pbar) {
    using clock = std::chrono::steady_clock;

    const int width = image.width;
    const int n_pixels = image.width * image.height;
    uint8_t* raw = image.rgb_image;

    // Load raw uint8_t* into stride-5 float working buffer: [r, g, b, x, y]
    auto t_conv_start = clock::now();
    std::vector<float> current(n_pixels * 5);
    for(int i = 0; i < n_pixels; ++i) {
        current[i * 5 + 0] = static_cast<float>(raw[i * 3 + 0]);
        current[i * 5 + 1] = static_cast<float>(raw[i * 3 + 1]);
        current[i * 5 + 2] = static_cast<float>(raw[i * 3 + 2]);
        current[i * 5 + 3] = static_cast<float>(i % width);
        current[i * 5 + 4] = static_cast<float>(i / width);
    }
    auto t_conv_end = clock::now();
    double convert_ms = std::chrono::duration<double, std::milli>(t_conv_end - t_conv_start).count();

    const float bandwidth_sq = bandwidth * bandwidth;

    double total_shift_ms = 0.0;
    int iter = 0;
    std::vector<IterationInfo> iter_details;

    for(; iter < max_iter; ++iter) {
        std::vector<float> next(n_pixels * 5);
        float max_change = 0.0f;

        auto t_shift_start = clock::now();

        for(int i = 0; i < n_pixels; ++i) {
            float sum_r = 0.0f, sum_g = 0.0f, sum_b = 0.0f;
            float weight_sum = 0.0f;

            for(int j = 0; j < n_pixels; ++j) {
                if(withinBandwidthBaseline(current, i, j, bandwidth_sq)) {
                    sum_r += current[j * 5 + 0];
                    sum_g += current[j * 5 + 1];
                    sum_b += current[j * 5 + 2];
                    weight_sum += 1.0f;
                }
            }

            float change = 0.0f;
            if(weight_sum > 0.0f) {
                float new_r = sum_r / weight_sum;
                float new_g = sum_g / weight_sum;
                float new_b = sum_b / weight_sum;
                change = std::max({std::abs(new_r - current[i * 5 + 0]),
                                   std::abs(new_g - current[i * 5 + 1]),
                                   std::abs(new_b - current[i * 5 + 2])});
                next[i * 5 + 0] = new_r;
                next[i * 5 + 1] = new_g;
                next[i * 5 + 2] = new_b;
            } else {
                next[i * 5 + 0] = current[i * 5 + 0];
                next[i * 5 + 1] = current[i * 5 + 1];
                next[i * 5 + 2] = current[i * 5 + 2];
            }
            // carry x, y through unchanged
            next[i * 5 + 3] = current[i * 5 + 3];
            next[i * 5 + 4] = current[i * 5 + 4];

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

    // Write results back to raw uint8_t* buffer (only r, g, b channels)
    auto t_conv_out_start = clock::now();
    for(int i = 0; i < n_pixels; ++i) {
        raw[i * 3 + 0] = static_cast<uint8_t>(std::round(std::max(0.0f, std::min(current[i * 5 + 0], 255.0f))));
        raw[i * 3 + 1] = static_cast<uint8_t>(std::round(std::max(0.0f, std::min(current[i * 5 + 1], 255.0f))));
        raw[i * 3 + 2] = static_cast<uint8_t>(std::round(std::max(0.0f, std::min(current[i * 5 + 2], 255.0f))));
    }
    auto t_conv_out_end = clock::now();
    convert_ms += std::chrono::duration<double, std::milli>(t_conv_out_end - t_conv_out_start).count();

    return MeanShiftResult{static_cast<int>(iter_details.size()), total_shift_ms, convert_ms, iter_details};
}

