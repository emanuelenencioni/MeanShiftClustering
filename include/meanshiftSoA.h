#ifndef MEANSHIFT_SOA_H
#define MEANSHIFT_SOA_H

#include "meanshift.h"
#include <cstdint>
#include <vector>

// SoA pixel representation - separate channels for better parallelism
struct PixelSoA {
    std::vector<float> r;
    std::vector<float> g;
    std::vector<float> b;
    int n;
};

// SoA neighbor accumulator
struct NeighborAccumulator {
    float sum_r = 0.0f;
    float sum_g = 0.0f;
    float sum_b = 0.0f;
    int count = 0;
};

// Convert AoS float buffer to SoA representation
void convertToFloatSoA(const std::vector<float>& current, PixelSoA& soa);

// Convert SoA back to AoS float buffer
void convertFromFloatSoA(const PixelSoA& soa, std::vector<float>& current);

// SoA version of squared distance between pixels i and j
float squaredDistanceSoA(const PixelSoA& soa, int i, int j);

// Mean shift using SoA - brute force O(n^2)
MeanShiftResult meanShiftSoA(std::vector<uint8_t>& data, float bandwidth,
                             int max_iter = 100, float tol = 1e-3f,
                             bool show_pbar = false);

// Mean shift using SoA - grid accelerated
MeanShiftResult meanShiftSoAOptimized(std::vector<uint8_t>& data, float bandwidth,
                                      int max_iter = 100, float tol = 1e-3f,
                                      bool show_pbar = false);

#endif // MEANSHIFT_SOA_H
