# SoA vs AoS Analysis for Mean Shift Clustering

## 1. Theoretical Background

### Array of Structures (AoS)
```cpp
// Memory layout: [r0,g0,b0, r1,g1,b1, r2,g2,b2, ...]
std::vector<float> current;  // interleaved RGB, stride-3 access
```

### Structure of Arrays (SoA)
```cpp
struct PixelSoA {
    std::vector<float> r;  // [r0,r1,r2,...]
    std::vector<float> g;  // [g0,g1,g2,...]
    std::vector<float> b;  // [b0,b1,b2,...]
    int n;
};
```

---

## 2. Why SoA Is Generally Beneficial

| Benefit | Description |
|---------|-------------|
| **SIMD Vectorization** | Contiguous single-channel arrays map directly to SIMD registers |
| **Prefetcher Efficiency** | Clean sequential streams per channel; hardware prefetcher handles multiple independent streams well |
| **Parallelism** | Independent channel operations simplify OpenMP / GPU parallelization |
| **Memory Coalescing** | Better GPU memory access patterns (warp-wide coalesced loads) |

---

## 3. Analysis for This Project

### 3.1 Update Semantics Mismatch (Critical Confound)

**The brute-force AoS and SoA variants use different update semantics.**
This is the most important factor when comparing their performance:

| Variant | Update style | Mechanism |
|---------|-------------|-----------|
| `meanShift` (brute AoS) | **Jacobi** | Writes to `next` buffer, `current.swap(next)` at end of iteration |
| `meanShiftSoA` (brute SoA) | **Gauss-Seidel** | Writes directly to `soa.r[i]`, `soa.g[i]`, `soa.b[i]` in-place |
| `meanShiftOptimized` (grid AoS) | **Jacobi** | Writes to `next` buffer, swaps at end |
| `meanShiftSoAOptimized` (grid SoA) | **Jacobi** | Writes to `next_r/g/b` buffers, swaps at end |

Gauss-Seidel converges faster because each pixel immediately sees already-
updated neighbors from earlier in the same iteration. This typically results
in **fewer iterations** to reach the same tolerance.

**Consequence**: any observed speedup of `brute_soa` over `brute` conflates
two independent effects:

1. **Per-iteration time** difference (true AoS vs SoA layout effect)
2. **Iteration count** difference (Gauss-Seidel converges in fewer iterations)

To isolate the layout effect, compare per-iteration times (available in the
`[Per-iteration]` output table), not total runtime. Or compare the grid
variants, which both use Jacobi.

### 3.2 Cache Access Pattern Analysis

**Brute-force inner loop** (O(n) scan through all pixels per query pixel):

| Layout | Access pattern | Cache line utilization (64 bytes = 16 floats) |
|--------|---------------|----------------------------------------------|
| AoS | `current[j*3+0], current[j*3+1], current[j*3+2]` | ~5.3 pixels per cache line (3 floats/pixel) |
| SoA | `soa.r[j], soa.g[j], soa.b[j]` from 3 arrays | 16 values per cache line per channel |

Both access memory sequentially, so the hardware prefetcher works well for
both. However, SoA gives the prefetcher **3 independent sequential streams**
vs AoS's single interleaved stream. On modern CPUs that track multiple
prefetch streams, this is a wash. Cache line utilization per pixel is similar
(~5.3 pixels/line for AoS vs 16/3 = ~5.3 pixels/line for SoA when all 3
channels are needed).

**Grid-accelerated inner loop** (hash table lookup + neighbor scan):

The bottleneck is **hash map lookups** (`std::unordered_map::find`), which
cause random cache misses regardless of pixel data layout. SoA vs AoS has
minimal impact here because the hash table access dominates.

### 3.3 Data Size Analysis

For a 1920x1080 image (2,073,600 pixels):
- AoS `current` vector: 2,073,600 x 3 x 4 bytes = **24.9 MB**
- SoA `PixelSoA`: 3 x 2,073,600 x 4 bytes = **24.9 MB**

Identical memory footprint. Neither layout has a size advantage.

### 3.4 SIMD Vectorization

SoA is **better** for explicit SIMD than AoS in this codebase:

```cpp
// SoA: contiguous channel data, maps directly to SIMD registers
// Process 8 pixels at once with AVX2:
__m256 vr = _mm256_loadu_ps(&soa.r[j]);     // 8 consecutive r values
__m256 vg = _mm256_loadu_ps(&soa.g[j]);     // 8 consecutive g values
__m256 vb = _mm256_loadu_ps(&soa.b[j]);     // 8 consecutive b values
```

```cpp
// AoS: interleaved data requires gather instructions (slow) or
// deinterleaving shuffles to load 8 r-values from stride-3 layout:
// current = [r0,g0,b0, r1,g1,b1, r2,g2,b2, ...]
// _mm256_loadu_ps(&current[0]) loads [r0,g0,b0,r1,g1,b1,r2,g2] -- mixed!
// Need _mm256_i32gather_ps for stride-3, which is 2-4x slower than loadu
```

However, the current code does not use explicit SIMD intrinsics, and compiler
auto-vectorization effectiveness depends on many factors. This advantage is
theoretical until SIMD is explicitly implemented.

---

## 4. When SoA IS Recommended

SoA shines when:

| Scenario | Why SoA Helps |
|----------|---------------|
| **GPU Computing** | Coalesced memory access, shared memory friendly |
| **Explicit SIMD** | Contiguous single-channel arrays map to vector registers |
| **OpenMP Parallelization** | Separate channel arrays simplify parallel regions |
| **Large Datasets (>100M records)** | Reduced L2/L3 cache pressure when processing one channel at a time |
| **Channel-independent operations** | E.g., per-channel FFT, histogram, threshold |

---

## 5. Benchmark Data

**No pre-computed benchmarks are included in this document.** Previous versions
of this file contained fabricated benchmark numbers attributed to hardware that
was never tested.

To generate real benchmarks on your system, use the benchmark script:

```bash
# Full benchmark (all combinations, 5 runs each)
bash benchmark.sh

# Results written to results/benchmark_YYMMDD_HHMMSS.csv
```

When analyzing the results, separate:
- **Per-iteration time**: isolates the true AoS vs SoA layout effect
- **Iteration count**: reveals convergence differences due to update semantics
- **Total time**: conflates both effects (useful for end-user performance, but
  not for understanding *why* one is faster)

### What to look for

1. **`brute` vs `brute_soa`**: iteration counts will likely differ (Jacobi vs
   Gauss-Seidel). Compare per-iteration times for a fair layout comparison.
2. **`grid` vs `grid_soa`**: both use Jacobi, so iteration counts should match.
   Total time comparison is fair here.
3. **Bandwidth effect**: larger bandwidth increases neighbor count per pixel,
   amplifying any per-pixel compute differences between layouts.

---

## 6. Code Quality Trade-offs

### Pros of SoA Implementation
- Clean separation of concerns per channel
- Easier to add explicit SIMD (contiguous channel arrays)
- Simpler OpenMP parallel loop structure
- Better documentation of data flow intent

### Cons of SoA Implementation
- More complex initialization (conversion to/from SoA)
- Verbose function signatures (pass struct + index vs pointer arithmetic)
- Requires separate `convertToFloatSoA` / `convertFromFloatSoA` helpers
- Three separate array allocations instead of one

---

## 7. Recommendations

### For Sequential Performance

The grid variants (which dominate practical use) show **minimal difference**
between AoS and SoA because hash table lookups dominate runtime. For brute-
force, any observed difference is confounded by update semantics.

### For Future Parallelization

SoA becomes more attractive when adding:
1. **OpenMP** -- separate channel arrays simplify reduction and avoid false
   sharing on cache lines
2. **Explicit SIMD** -- contiguous channels map directly to vector registers
3. **GPU offload** -- SoA enables coalesced global memory access

### Fair Comparison Checklist

Before concluding AoS or SoA is faster, verify:
- [ ] Both variants use the **same update semantics** (both Jacobi or both
      Gauss-Seidel)
- [ ] **Iteration counts match** for the same input
- [ ] **Multiple runs** (5+) are averaged to account for variance
- [ ] **Per-iteration time** is compared, not just total time

---

## 8. Appendix: SoA SIMD Example

```cpp
// With SoA, SIMD distance computation is straightforward:
__m256 sum_sq = _mm256_setzero_ps();
for(int j = 0; j < n_pixels; j += 8) {
    __m256 dr = _mm256_sub_ps(_mm256_loadu_ps(&soa.r[j]), src_r_broadcast);
    __m256 dg = _mm256_sub_ps(_mm256_loadu_ps(&soa.g[j]), src_g_broadcast);
    __m256 db = _mm256_sub_ps(_mm256_loadu_ps(&soa.b[j]), src_b_broadcast);

    __m256 dist_sq = _mm256_fmadd_ps(dr, dr,
                     _mm256_fmadd_ps(dg, dg,
                     _mm256_mul_ps(db, db)));

    __m256 mask = _mm256_cmp_ps(dist_sq, bw_sq_broadcast, _CMP_LE_OQ);
    // Use mask to conditionally accumulate...
}
```

---

*Last updated: March 2026*
*Run `bash benchmark.sh` to generate real numbers on your hardware.*
