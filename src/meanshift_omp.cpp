#include "meanshift_omp.h"
#include <omp.h>
#include <chrono>
#include <algorithm>
#include <cstdio>

/* 5D squared distance between two pixels: spatial (x, y) + color (r, g, b). */
static float squaredDistanceOMP(const Pixel& a, const Pixel& b) {
    float dx = a.x - b.x;
    float dy = a.y - b.y;
    float dr = a.r - b.r;
    float dg = a.g - b.g;
    float db = a.b - b.b;
    return dx*dx + dy*dy + dr*dr + dg*dg + db*db;
}

/* Convert raw uint8_t RGB data to Pixel array.
 * Spatial coords (x, y) are computed once here and stored in each Pixel. */
static void toPixelsOMP(const std::vector<uint8_t>& data,
                        std::vector<Pixel>& pixels, int width) {
    int n = static_cast<int>(data.size() / 3);
    pixels.resize(n);
    for(int i = 0; i < n; ++i) {
        pixels[i] = { static_cast<float>(data[i * 3 + 0]),
                      static_cast<float>(data[i * 3 + 1]),
                      static_cast<float>(data[i * 3 + 2]),
                      static_cast<float>(i % width),
                      static_cast<float>(i / width) };
    }
}

/* Write r, g, b back from Pixel array to raw uint8_t data. x, y are ignored. */
static void fromPixelsOMP(const std::vector<Pixel>& pixels, std::vector<uint8_t>& data) {
    for(int i = 0; i < static_cast<int>(pixels.size()); ++i) {
        data[i * 3 + 0] = static_cast<uint8_t>(std::round(std::max(0.0f, std::min(pixels[i].r, 255.0f))));
        data[i * 3 + 1] = static_cast<uint8_t>(std::round(std::max(0.0f, std::min(pixels[i].g, 255.0f))));
        data[i * 3 + 2] = static_cast<uint8_t>(std::round(std::max(0.0f, std::min(pixels[i].b, 255.0f))));
    }
}

// Parallel AoS mean shift using OpenMP.
// Outer loop over pixels is parallelised; Jacobi semantics guarantee no races.
// next[] is hoisted outside the iteration loop to avoid repeated allocation.
MeanShiftResult meanShiftOMP(std::vector<uint8_t>& data, int width, float bandwidth,
                             int max_iter, float tol, bool show_pbar) {
    using clock = std::chrono::steady_clock;

    std::vector<Pixel> current;
    toPixelsOMP(data, current, width);

    const float bandwidth_sq = bandwidth * bandwidth;
    const int n_pixels = static_cast<int>(current.size());

    // Hoist next outside the iteration loop — allocate once, reuse every iteration.
    std::vector<Pixel> next(n_pixels);

    double total_shift_ms = 0.0;
    int iter = 0;
    std::vector<IterationInfo> iter_details;

    for(; iter < max_iter; ++iter) {
        float max_change = 0.0f;

        auto t_shift_start = clock::now();

        #pragma omp parallel for schedule(static) reduction(max:max_change)
        for(int i = 0; i < n_pixels; ++i) {
            const Pixel& src = current[i];
            float sum_r = 0.0f, sum_g = 0.0f, sum_b = 0.0f;
            int count = 0;

            for(int j = 0; j < n_pixels; ++j) {
                if(squaredDistanceOMP(src, current[j]) <= bandwidth_sq) {
                    sum_r += current[j].r;
                    sum_g += current[j].g;
                    sum_b += current[j].b;
                    count++;
                }
            }

            float change = 0.0f;
            if(count > 0) {
                float new_r = sum_r / count;
                float new_g = sum_g / count;
                float new_b = sum_b / count;
                change = std::max({std::abs(new_r - src.r),
                                   std::abs(new_g - src.g),
                                   std::abs(new_b - src.b)});
                next[i] = {new_r, new_g, new_b, src.x, src.y};
            } else {
                next[i] = src;
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

    fromPixelsOMP(current, data);

    return MeanShiftResult{static_cast<int>(iter_details.size()), total_shift_ms, iter_details};
}
