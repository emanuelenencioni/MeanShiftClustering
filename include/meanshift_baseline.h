#ifndef MEANSHIFT_BASELINE_H
#define MEANSHIFT_BASELINE_H

#include <cstdint>
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

void convertToFloat(const std::vector<uint8_t>& data, std::vector<float>& out);
void convertFromFloat(const std::vector<float>& current, std::vector<uint8_t>& data);
void printProgressBar(int iter, int max_iter, float max_change);

/* -------------------------------------------------------------------------
 * Baseline algorithm
 * - Input/output: image.rgb_image (raw uint8_t*, RGB, stride 3), modified in place.
 * - Spatial coords (x, y) precomputed into flat buffer (stride 5: r,g,b,x,y).
 * - next buffer allocated fresh each iteration (no hoisting).
 * - Jacobi update: reads from current[], writes to next[], then swaps.
 * This is the unoptimized baseline for performance comparison. */
MeanShiftResult meanShiftBaseline(STBImage& image, float bandwidth,
                                  int max_iter = 100, float tol = 1e-3f,
                                  bool show_pbar = false);

#endif // MEANSHIFT_BASELINE_H
