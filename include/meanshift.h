#ifndef MEANSHIFT_H
#define MEANSHIFT_H

#include <cstdint>
#include <functional>
#include <string>
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

/* Kernel function type: takes squared distance and squared bandwidth,
 * returns a non-negative weight (0 means the neighbor is excluded). */
using KernelFn = std::function<float(float sq_dist, float bw_sq)>;

/* Factory: returns the kernel for the given name.
 * Recognised names: "flat", "gaussian", "epanechnikov".
 * Prints an error and calls std::exit(1) on an unknown name. */
KernelFn makeKernel(const std::string& name);

void convertToFloat(const std::vector<uint8_t>& data, std::vector<float>& out);
void convertFromFloat(const std::vector<float>& current, std::vector<uint8_t>& data);
void printProgressBar(int iter, int max_iter, float max_change);

MeanShiftResult meanShift(std::vector<uint8_t>& data, int width, float bandwidth,
                          int max_iter = 100, float tol = 1e-3f, bool show_pbar = false,
                          KernelFn kernel = nullptr);

#endif // MEANSHIFT_H
