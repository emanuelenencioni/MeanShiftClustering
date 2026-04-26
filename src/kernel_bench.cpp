#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <string>
#include <vector>

#include "meanshift_baseline.h"

/* Kernel microbenchmark.
 *
 * Measures throughput of each kernel function in isolation.
 * Inputs are pre-generated before timing starts so generation cost is excluded.
 * 50% of sq_dist values fall inside the bandwidth (realistic mix).
 * Results are accumulated into a float sink and printed at the end to prevent
 * the compiler from eliminating the kernel calls as dead code. */

static const int N        = 10'000'000;  // calls per kernel
static const float BW     = 30.0f;
static const float BW_SQ  = BW * BW;    // 900.0

int main() {
    /* Pre-generate N input sq_dist values: 50% in [0, BW_SQ), 50% in [BW_SQ, 4*BW_SQ].
     * Shuffled so the two groups are interleaved — avoids branch-prediction advantage
     * from having all in-range values grouped together. */
    std::vector<float> inputs(N);
    {
        std::mt19937 rng(42);
        std::uniform_real_distribution<float> inside(0.0f, BW_SQ - 1.0f);
        std::uniform_real_distribution<float> outside(BW_SQ, 4.0f * BW_SQ);
        for(int i = 0; i < N; ++i)
            inputs[i] = (i % 2 == 0) ? inside(rng) : outside(rng);
        /* Shuffle to interleave in/out values */
        std::shuffle(inputs.begin(), inputs.end(), rng);
    }

    const char* names[] = {"flat", "gaussian", "epanechnikov"};

    std::printf("%-16s %12s %14s %12s\n",
                "kernel", "calls", "total_ms", "ns/call");
    std::printf("%-16s %12s %14s %12s\n",
                "------", "-----", "--------", "-------");

    float grand_sink = 0.0f;

    for(const char* name : names) {
        KernelFn kernel = makeKernel(std::string(name));

        float sink = 0.0f;
        using clock = std::chrono::steady_clock;

        auto t_start = clock::now();
        for(int i = 0; i < N; ++i)
            sink += kernel(inputs[i], BW_SQ);
        auto t_end = clock::now();

        double total_ms = std::chrono::duration<double, std::milli>(t_end - t_start).count();
        double ns_per_call = (total_ms * 1e6) / N;

        std::printf("%-16s %12d %14.2f %12.2f\n",
                    name, N, total_ms, ns_per_call);

        grand_sink += sink;
    }

    /* Print sink to defeat dead-code elimination — ignore this value. */
    std::printf("(sink=%.2f)\n", grand_sink);

    return 0;
}
