#ifndef MEANSHIFT_SOA_H
#define MEANSHIFT_SOA_H

#include "meanshift_baseline.h"
#include <cstdint>
#include <vector>

// SoA pixel representation - separate channels for better parallelism
struct ImageSoA {
    std::vector<float> r;
    std::vector<float> g;
    std::vector<float> b;
    std::vector<float> x;  // spatial column coordinate (fixed)
    std::vector<float> y;  // spatial row coordinate (fixed)
    int n;
};

// SoA neighbor accumulator — uses float weight_sum to support weighted kernels
struct NeighborAccumulator {
    float sum_r = 0.0f;
    float sum_g = 0.0f;
    float sum_b = 0.0f;
    float weight_sum = 0.0f;
};

// Convert AoS float buffer to SoA representation
void convertToFloatSoA(const std::vector<float>& current, ImageSoA& soa, int width);

// Convert SoA back to AoS float buffer
void convertFromFloatSoA(const ImageSoA& soa, std::vector<float>& current);

// SoA version of 5D squared distance between pixels i and j
float squaredDistanceSoA(const ImageSoA& soa, int i, int j);

// Mean shift using SoA - brute force O(n^2), 5D feature space (x, y, R, G, B)
MeanShiftResult meanShiftSoA(std::vector<uint8_t>& data, int width, float bandwidth,
                             int max_iter = 100, float tol = 1e-3f,
                             bool show_pbar = false, KernelFn kernel = nullptr);

#endif // MEANSHIFT_SOA_H
