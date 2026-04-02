#ifndef MEANSHIFT_H
#define MEANSHIFT_H

#include <cstdint>
#include <vector>
#include <cmath>

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

/* 5D squared distance: (x, y, R, G, B).
 * ax, ay are the spatial coordinates of point a; bx, by of point b.
 * a[0..2] and b[0..2] are R, G, B values. */
float squaredDistance(const float* a, const float* b,
                      float ax, float ay, float bx, float by);

void convertToFloat(const std::vector<uint8_t>& data, std::vector<float>& out);
void convertFromFloat(const std::vector<float>& current, std::vector<uint8_t>& data);
void printProgressBar(int iter, int max_iter, float max_change);

MeanShiftResult meanShift(std::vector<uint8_t>& data, int width, float bandwidth,
                          int max_iter = 100, float tol = 1e-3f, bool show_pbar = false);

#endif // MEANSHIFT_H
