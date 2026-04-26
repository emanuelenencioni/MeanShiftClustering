#include <chrono>
#include <algorithm>
#include <cstdio>
#include <cstdlib>

#include "meanshift_seq.h"

// 5D squared distance between two pixels: spatial (x, y) + color (r, g, b).
static float squaredDistance(const Pixel& a, const Pixel& b) {
    float dx = a.x - b.x;
    float dy = a.y - b.y;
    float dr = a.r - b.r;
    float dg = a.g - b.g;
    float db = a.b - b.b;
    return dx*dx + dy*dy + dr*dr + dg*dg + db*db;
}

/* Convert raw uint8_t RGB data to Pixel array.
 * Spatial coords (x, y) are computed once here and stored in each Pixel. */
static void toPixels(const std::vector<uint8_t>& data,
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
static void fromPixels(const std::vector<Pixel>& pixels, std::vector<uint8_t>& data) {
    for(int i = 0; i < static_cast<int>(pixels.size()); ++i) {
        data[i * 3 + 0] = static_cast<uint8_t>(std::round(std::max(0.0f, std::min(pixels[i].r, 255.0f))));
        data[i * 3 + 1] = static_cast<uint8_t>(std::round(std::max(0.0f, std::min(pixels[i].g, 255.0f))));
        data[i * 3 + 2] = static_cast<uint8_t>(std::round(std::max(0.0f, std::min(pixels[i].b, 255.0f))));
    }
}

/* --- Mean shift ---------------------------------------------------------- */

// Brute-force: O(n^2) per iteration, 5D feature space (x, y, R, G, B)
MeanShiftResult meanShift(std::vector<uint8_t>& data, int width, float bandwidth,
                          int max_iter, float tol, bool show_pbar, KernelFn kernel) {
    using clock = std::chrono::steady_clock;

    if(!kernel) kernel = makeKernel("flat");

    std::vector<Pixel> current;
    toPixels(data, current, width);

    const float bandwidth_sq = bandwidth * bandwidth;
    const int n_pixels = static_cast<int>(current.size());

    double total_shift_ms = 0.0;
    int iter = 0;
    std::vector<IterationInfo> iter_details;

    for(; iter < max_iter; ++iter) {
        std::vector<Pixel> next(n_pixels);
        float max_change = 0.0f;

        auto t_shift_start = clock::now();

        for(int i = 0; i < n_pixels; ++i) {
            const Pixel& src = current[i];
            float sum_r = 0.0f, sum_g = 0.0f, sum_b = 0.0f;
            float weight_sum = 0.0f;

            for(int j = 0; j < n_pixels; ++j) {
                float w = kernel(squaredDistance(src, current[j]), bandwidth_sq);
                if(w > 0.0f) {
                    sum_r += w * current[j].r;
                    sum_g += w * current[j].g;
                    sum_b += w * current[j].b;
                    weight_sum += w;
                }
            }

            float change = 0.0f;
            if(weight_sum > 0.0f) {
                float new_r = sum_r / weight_sum;
                float new_g = sum_g / weight_sum;
                float new_b = sum_b / weight_sum;
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

    fromPixels(current, data);

    return MeanShiftResult{static_cast<int>(iter_details.size()), total_shift_ms, iter_details};
}
