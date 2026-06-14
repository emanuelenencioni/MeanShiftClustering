#ifndef MEANSHIFT_CUDA_2D_H
#define MEANSHIFT_CUDA_2D_H

#include "meanshift_baseline.h"
#include <cstdint>
#include <vector>

MeanShiftResult meanShiftCUDA2D(std::vector<uint8_t>& data, int width, float bandwidth,
                                 int max_iter = 100, float tol = 1e-3f,
                                 bool show_pbar = false, int block_size = 256);

#endif // MEANSHIFT_CUDA_2D_H
