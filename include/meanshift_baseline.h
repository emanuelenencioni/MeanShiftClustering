#ifndef MEANSHIFT_BASELINE_H
#define MEANSHIFT_BASELINE_H

#include "meanshift.h"
#include "STBImage.h"

/* Naive brute-force mean shift operating directly on the raw STBImage pixel buffer.
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
