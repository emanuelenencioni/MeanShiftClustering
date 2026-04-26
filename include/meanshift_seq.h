#ifndef MEANSHIFT_SEQ_H
#define MEANSHIFT_SEQ_H

#include "meanshift_baseline.h"

/* Sequential optimized AoS mean shift.
 * - Spatial coords (x, y) precomputed once into Pixel struct at load time.
 * - Jacobi update: reads from current[], writes to next[], then swaps. */
MeanShiftResult meanShift(std::vector<uint8_t>& data, int width, float bandwidth,
                          int max_iter = 100, float tol = 1e-3f, bool show_pbar = false,
                          KernelFn kernel = nullptr);

#endif // MEANSHIFT_SEQ_H
