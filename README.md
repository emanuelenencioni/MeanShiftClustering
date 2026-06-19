# Mean Shift Clustering — Parallel Computing Project

Progressive optimization of the Mean Shift clustering algorithm for image segmentation, from a naive sequential baseline to GPU acceleration with CUDA.

## Features

Seven implementations, each targeting a different performance bottleneck:

| Variant | Description |
|---------|-------------|
| `baseline` | Naive AoS, stride-5 float buffer, next buffer allocated per iteration |
| `seq` | Optimized AoS with `Pixel` struct, x/y precomputed, next buffer hoisted |
| `soa` | Structure of Arrays layout → enables SIMD auto-vectorization |
| `omp` | OpenMP parallel AoS (same inner loop as seq, multi-threaded) |
| `omp_soa` | OpenMP parallel SoA (SIMD + multi-threading) |
| `cuda` | CUDA 1D grid with tiled shared-memory kernel |
| `cuda_2d` | CUDA 2D grid variant for comparison |

## Requirements

- CMake ≥ 3.10
- C++17 compiler (GCC, Clang)
- OpenCV
- OpenMP
- (Optional) CUDA 11.4+ with a compatible host compiler

## Build

```bash
cmake -S . -B build
cmake --build build
```

The CUDA variant is automatically enabled if a compatible CUDA toolkit is found, otherwise it is skipped.

## Usage

```bash
./build/mean_shift_seq <image> [bandwidth] [max_iter] [algorithm] [flags]
```

### Examples

```bash
# Default: bw=150, iter=100, seq
./build/mean_shift_seq Images/2.png

# Brute-force with custom parameters
./build/mean_shift_seq Images/2.png 30 50 baseline

# OpenMP with 5 iterations, no GUI
./build/mean_shift_seq Images/green400.png 40 5 omp --no-display

# CUDA (requires GPU)
./build/mean_shift_seq Images/2.png 30 10 cuda --no-display --no-output
```

### Algorithms

`baseline`, `seq`, `soa`, `omp`, `omp_soa`, `cuda`, `cuda_2d`

### Flags

| Flag | Effect |
|------|--------|
| `--pbar` | Show progress bar on stderr |
| `--no-display` | Skip GUI window |
| `--no-output` | Skip writing result PNG and log file |

## Key Results

All measurements on an AMD Ryzen 9 9900X (12 cores) + GTX 1060 (CUDA), 68K-pixel image, flat kernel, 5 iterations.

| Variant | Time (68K px) | Speedup vs baseline | Speedup vs seq |
|---------|--------------|-------------------|----------------|
| Baseline | 46.5 s | 1× | — |
| Seq (AoS) | 41.0 s | 1.13× | — |
| SoA | 15.9 s | 2.93× | — |
| OMP (12T) | 3.7 s | ~12.5× | ~11× |
| OMP SoA (12T) | 3.3 s | ~14× | ~12× |
| CUDA | 0.2 s | ~230× | ~203× |

## Project Structure

```
├── CMakeLists.txt          # Build configuration
├── benchmark.sh            # Automated benchmark script
├── profile.sh              # Perf profiling script
├── include/                # Headers
│   ├── meanshift_baseline.h
│   ├── meanshift_seq.h
│   ├── meanshift_soa.h
│   ├── meanshift_omp.h
│   ├── meanshift_omp_soa.h
│   ├── meanshift_cuda.h
│   ├── meanshift_cuda_2d.h
│   ├── STBImage.h
│   ├── stb_image.h          # Vendored
│   └── stb_image_write.h    # Vendored
├── src/                    # Implementation
│   ├── main.cpp
│   ├── STBImage.cpp
│   ├── meanshift_baseline.cpp
│   ├── meanshift_seq.cpp
│   ├── meanshift_soa.cpp
│   ├── meanshift_omp.cpp
│   ├── meanshift_omp_soa.cpp
│   ├── meanshift_cuda.cu
│   └── meanshift_cuda_2d.cu
├── Images/                 # Test images (BSD500-derived)
├── BSD500/                 # Original dataset
├── docs/                   # Documentation and analysis
│   ├── latex/main.tex      # Final report
└── results/                # Benchmark CSVs
```

## Dataset

The benchmark uses 25 images derived from the Berkeley Segmentation Dataset (BSD500): 5 source images at 5 resolution tiers (100×67 to 320×214, 6.7K to 68.5K pixels).
