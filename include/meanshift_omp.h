#ifndef MEANSHIFT_OMP_H
#define MEANSHIFT_OMP_H

#include "meanshift_baseline.h"
#include <cstdint>
#include <vector>

// Parallel AoS mean shift using OpenMP.
// Identical algorithm to meanShift (seq); parallelised with #pragma omp parallel for.
// Thread count controlled via OMP_NUM_THREADS environment variable.
MeanShiftResult meanShiftOMP(std::vector<uint8_t>& data, int width, float bandwidth,
                             int max_iter = 100, float tol = 1e-3f,
                             bool show_pbar = false, KernelFn kernel = nullptr);

#endif // MEANSHIFT_OMP_H
