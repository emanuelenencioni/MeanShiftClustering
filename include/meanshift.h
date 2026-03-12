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

struct MeanShiftResult {
    int iterations;
    double grid_build_ms;
    double pixel_shift_ms;
};

float squaredDistance(const float* a, const float* b);

MeanShiftResult meanShift(std::vector<uint8_t>& data, float bandwidth,
                          int max_iter = 100, float tol = 1e-3f, bool show_pbar = false);

MeanShiftResult meanShiftOptimized(std::vector<uint8_t>& data, float bandwidth,
                                   int max_iter = 100, float tol = 1e-3f, bool show_pbar = false);

#endif // MEANSHIFT_H
