#ifndef MEANSHIFT_H
#define MEANSHIFT_H

#include <cstdint>
#include <vector>
#include <cmath>
#include <unordered_map>

struct Bin {
    int x, y, z;
    bool operator==(const Bin& other) const {
        return x == other.x && y == other.y && z == other.z;
    }
};

namespace std {
    template<> struct hash<Bin> {
        size_t operator()(const Bin& b) const {
            return hash<int>()(b.x) ^ (hash<int>()(b.y) << 1) ^ (hash<int>()(b.z) << 2);
        }
    };
}

struct IterationInfo {
    int iteration;
    double time_ms;
    float max_change;
};

struct MeanShiftResult {
    int iterations;
    double grid_build_ms;
    double pixel_shift_ms;
    std::vector<IterationInfo> iter_details;
};

float squaredDistance(const float* a, const float* b);

void convertToFloat(const std::vector<uint8_t>& data, std::vector<float>& out);
void convertFromFloat(const std::vector<float>& current, std::vector<uint8_t>& data);
void printProgressBar(int iter, int max_iter, float max_change);

MeanShiftResult meanShift(std::vector<uint8_t>& data, float bandwidth,
                          int max_iter = 100, float tol = 1e-3f, bool show_pbar = false);

MeanShiftResult meanShiftOptimized(std::vector<uint8_t>& data, float bandwidth,
                                   int max_iter = 100, float tol = 1e-3f, bool show_pbar = false);

#endif // MEANSHIFT_H
