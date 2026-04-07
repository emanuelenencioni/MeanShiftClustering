#ifndef MEANSHIFT_H
#define MEANSHIFT_H

#include <cstdint>
#include <vector>
#include <cmath>

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
    std::vector<IterationInfo> iter_details;
};

void convertToFloat(const std::vector<uint8_t>& data, std::vector<float>& out);
void convertFromFloat(const std::vector<float>& current, std::vector<uint8_t>& data);
void printProgressBar(int iter, int max_iter, float max_change);

MeanShiftResult meanShift(std::vector<uint8_t>& data, int width, float bandwidth,
                          int max_iter = 100, float tol = 1e-3f, bool show_pbar = false);

#endif // MEANSHIFT_H
