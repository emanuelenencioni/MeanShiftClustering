#ifndef MEANSHIFT_BASELINE_H
#define MEANSHIFT_BASELINE_H

#include <cstdint>
#include <functional>
#include <string>
#include <vector>
#include <cmath>

#include "STBImage.h"

/* -------------------------------------------------------------------------
 * Shared types and utilities — all other mean shift headers include this file.
 * ------------------------------------------------------------------------- */

/* AoS pixel: color channels (shift each iteration) + fixed spatial coords. */
struct Pixel {
    float r, g, b;  // color — updated each mean shift iteration
    float x, y;     // spatial position — fixed at load time, never updated
};

struct IterationInfo {
    int iteration;
    double time_ms;
    float max_change;
};

struct MeanShiftResult {
    int iterations;
    double pixel_shift_ms;
    double convert_ms;   // input + output conversion time (toPixels / fromPixels)
    std::vector<IterationInfo> iter_details;
};

/* Kernel function type: takes squared distance and squared bandwidth,
 * returns a non-negative weight (0 means the neighbor is excluded). */
using KernelFn = std::function<float(float sq_dist, float bw_sq)>;

/* Factory: returns the kernel for the given name.
 * Recognised names: "flat", "gaussian", "epanechnikov".
 * Prints an error and calls std::exit(1) on an unknown name. */
KernelFn makeKernel(const std::string& name);

void convertToFloat(const std::vector<uint8_t>& data, std::vector<float>& out);
void convertFromFloat(const std::vector<float>& current, std::vector<uint8_t>& data);
void printProgressBar(int iter, int max_iter, float max_change);

/* -------------------------------------------------------------------------
 * Baseline algorithm
 * - Input/output: image.rgb_image (raw uint8_t*, RGB, stride 3), modified in place.
 * - Spatial coords (x, y) recomputed via i%width / i/width inside the O(n^2) inner loop.
 * - Internal float working buffer: std::vector<float>.
 * - next buffer allocated fresh each iteration (no hoisting).
 * - Jacobi update: reads from current[], writes to next[], then swaps.
 * This is the unoptimized baseline for performance comparison. */
MeanShiftResult meanShiftBaseline(STBImage& image, float bandwidth,
                                  int max_iter = 100, float tol = 1e-3f,
                                  bool show_pbar = false, KernelFn kernel = nullptr);

#endif // MEANSHIFT_BASELINE_H
