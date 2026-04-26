#!/usr/bin/env bash
#
# benchmark.sh — Run mean shift benchmark across a test matrix and output CSV.
#
# Usage:
#   bash benchmark.sh              # full matrix, 5 runs each
#   bash benchmark.sh --dry-run    # print what would run, don't execute
#
# Matrix:
#   Sequential : seq, soa          × images × bandwidths × 1 thread  × NUM_RUNS
#   Parallel   : omp, omp_soa      × images × bandwidths × THREADS   × NUM_RUNS
#
# Output: results/benchmark_YYMMDD_HHMMSS.csv

set -euo pipefail

# ── Configuration ─────────────────────────────────────────────────────────────

BINARY="./build/mean_shift_seq"
MAX_ITER=10
NUM_RUNS=5
KERNEL="flat"

# Images to benchmark (relative paths from project root)
IMAGES=(
    "Images/solid_100.png"          # 100x100,   tiny,  solid colour
    "Images/red400.png"             # 400x400,   small, solid colour
    "Images/green400.png"           # 400x400,   small, solid colour
    "Images/2.png"                  # 333x500,   medium, natural photo
    "BSD500/216041.jpg"             # 481x321,   medium, natural photo
    "BSD500/188005.jpg"             # 481x321,   medium, natural photo
    "BSD500/242078.jpg"             # 481x321,   medium, natural photo
    "BSD500/246016.jpg"             # 481x321,   medium, natural photo
    "BSD500/353013.jpg"             # 481x321,   medium, natural photo
    "Images/clusters_800.png"       # 800x600,   medium, few flat clusters
    "Images/gradient_800.png"       # 800x600,   medium, smooth gradient
    "Images/complex_textures.jpg"   # large, high complexity
    "Images/parrots.jpg"            # 3000x2000,  large, complex
)

# Sequential algorithms — always run with threads=1
SEQ_ALGORITHMS=("seq" "soa")

# Parallel algorithms — run once per entry in THREADS
OMP_ALGORITHMS=("omp" "omp_soa")

BANDWIDTHS=(20 50 100)

THREADS=(1 2 4 8 12 24)

# ── Argument parsing ──────────────────────────────────────────────────────────

DRY_RUN=0
for arg in "$@"; do
    case "$arg" in
        --dry-run) DRY_RUN=1 ;;
        *) echo "Unknown argument: $arg" >&2; exit 1 ;;
    esac
done

# ── Pre-flight checks ────────────────────────────────────────────────────────

if [[ ! -x "$BINARY" ]]; then
    echo "Binary not found: $BINARY" >&2
    echo "Run: cmake --build build" >&2
    exit 1
fi

missing=()
for img in "${IMAGES[@]}"; do
    [[ -f "$img" ]] || missing+=("$img")
done
if (( ${#missing[@]} > 0 )); then
    echo "Warning: missing images (will be skipped): ${missing[*]}" >&2
fi

# ── Output setup ──────────────────────────────────────────────────────────────

mkdir -p results
TIMESTAMP=$(date +%y%m%d_%H%M%S)
CSV="results/benchmark_${TIMESTAMP}.csv"

HEADER="image,algorithm,kernel,threads,bandwidth,run,iterations,total_ms,shift_ms,convert_ms,avg_iter_ms"
echo "$HEADER" > "$CSV"

# ── Count total runs ──────────────────────────────────────────────────────────

total=0
for img in "${IMAGES[@]}"; do
    [[ -f "$img" ]] || continue
    for bw in "${BANDWIDTHS[@]}"; do
        # sequential: 1 thread count per algo
        total=$(( total + ${#SEQ_ALGORITHMS[@]} * NUM_RUNS ))
        # parallel: one run per thread count per algo
        total=$(( total + ${#OMP_ALGORITHMS[@]} * ${#THREADS[@]} * NUM_RUNS ))
    done
done

echo "Benchmark:" >&2
echo "  Sequential : ${SEQ_ALGORITHMS[*]} — threads=1" >&2
echo "  Parallel   : ${OMP_ALGORITHMS[*]} — threads=${THREADS[*]}" >&2
echo "  Images     : ${#IMAGES[@]}  Bandwidths: ${#BANDWIDTHS[@]}  Runs: ${NUM_RUNS}" >&2
echo "  Total runs : $total" >&2
echo "  Output     : $CSV" >&2
echo "" >&2

# ── Run matrix ────────────────────────────────────────────────────────────────

run_idx=0

run_one() {
    local img="$1" algo="$2" threads="$3" bw="$4" run="$5"
    run_idx=$(( run_idx + 1 ))
    printf "\r[%d/%d] %-30s %-10s threads=%-3d bw=%-4d run=%d   " \
        "$run_idx" "$total" "$img" "$algo" "$threads" "$bw" "$run" >&2

    if (( DRY_RUN )); then
        return
    fi

    local output
    output=$(OMP_NUM_THREADS="$threads" \
        "$BINARY" "$img" "$bw" "$MAX_ITER" "$algo" \
            --kernel "$KERNEL" --no-display --no-output 2>/dev/null) || {
        echo "" >&2
        echo "  FAILED: $img $algo threads=$threads bw=$bw run=$run" >&2
        return
    }

    local iterations total_ms shift_ms convert_ms avg_iter_ms
    iterations=$(echo "$output"  | grep -oP 'Iterations:\s+\K[0-9]+'       || echo "0")
    total_ms=$(echo "$output"    | grep -oP 'Total:\s+\K[0-9.]+'            || echo "0")
    shift_ms=$(echo "$output"    | grep -oP 'Pixel shifting:\s+\K[0-9.]+'   || echo "0")
    convert_ms=$(echo "$output"  | grep -oP 'Convert:\s+\K[0-9.]+'          || echo "0")
    avg_iter_ms=$(echo "$output" | grep -oP 'Avg:\s+\K[0-9.]+'              || echo "0")

    echo "$img,$algo,$KERNEL,$threads,$bw,$run,$iterations,$total_ms,$shift_ms,$convert_ms,$avg_iter_ms" >> "$CSV"
}

for img in "${IMAGES[@]}"; do
    [[ -f "$img" ]] || { echo "  SKIP $img (not found)" >&2; continue; }

    for bw in "${BANDWIDTHS[@]}"; do
        # Sequential algorithms — fixed threads=1
        for algo in "${SEQ_ALGORITHMS[@]}"; do
            # Warmup: warm page cache and branch predictors for this (img, algo)
            if (( ! DRY_RUN )); then
                OMP_NUM_THREADS=1 "$BINARY" "$img" "${BANDWIDTHS[0]}" "$MAX_ITER" "$algo" \
                    --kernel "$KERNEL" --no-display --no-output >/dev/null 2>&1 || true
            fi
            for run in $(seq 1 "$NUM_RUNS"); do
                run_one "$img" "$algo" 1 "$bw" "$run"
            done
        done

        # Parallel algorithms — iterate over thread counts
        for algo in "${OMP_ALGORITHMS[@]}"; do
            for threads in "${THREADS[@]}"; do
                # Warmup: ensure OMP thread pool is live at the correct thread count
                if (( ! DRY_RUN )); then
                    OMP_NUM_THREADS="$threads" "$BINARY" "$img" "${BANDWIDTHS[0]}" "$MAX_ITER" "$algo" \
                        --kernel "$KERNEL" --no-display --no-output >/dev/null 2>&1 || true
                fi
                for run in $(seq 1 "$NUM_RUNS"); do
                    run_one "$img" "$algo" "$threads" "$bw" "$run"
                done
            done
        done
    done
done

echo "" >&2
echo "Done. Results written to: $CSV" >&2
