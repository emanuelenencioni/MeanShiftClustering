#!/usr/bin/env bash
#
# benchmark.sh — Run mean shift benchmark across a test matrix and output CSV.
#
# Usage:
#   bash benchmark.sh              # full matrix, 5 runs each
#   bash benchmark.sh --dry-run    # print what would run, don't execute
#
# Output: results/benchmark_YYMMDD_HHMMSS.csv

set -euo pipefail

# ── Configuration ─────────────────────────────────────────────────────────────

BINARY="./build/mean_shift_seq"
MAX_ITER=10
NUM_RUNS=5

# Images to benchmark (relative paths from project root)
IMAGES=(
    "Images/solid_100.png"          # 100x100,   tiny,  solid colour
    "Images/red400.png"             # 400x400,   small, solid colour
    "Images/green400.png"           # 400x400,   small, solid colour
    "Images/2.png"                  # 333x500,   medium, natural photo
    "BSD500/216041.jpg"  # 481x321,   medium, natural photo
    "BSD500/188005.jpg"  # 481x321,   medium, natural photo
    "BSD500/242078.jpg"  # 481x321,   medium, natural photo
    "BSD500/246016.jpg"  # 481x321,   medium, natural photo
    "BSD500/353013.jpg"             # 481x321,   medium, natural photo
    "Images/clusters_800.png"       # 800x600,   medium, few flat clusters
    "Images/gradient_800.png"       # 800x600,   medium, smooth gradient
    "Images/complex_textures.jpg"   # large, high complexity
    "Images/parrots.jpg"            # 3000x2000,  large, complex
)

ALGORITHMS=("seq" "soa")

BANDWIDTHS=(20 50 100)

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

HEADER="image,algorithm,bandwidth,run,iterations,total_ms,shift_ms,avg_iter_ms"
echo "$HEADER" > "$CSV"

# ── Count total runs ──────────────────────────────────────────────────────────

total=0
for img in "${IMAGES[@]}"; do
    [[ -f "$img" ]] || continue
    for algo in "${ALGORITHMS[@]}"; do
        for bw in "${BANDWIDTHS[@]}"; do
            total=$(( total + NUM_RUNS ))
        done
    done
done

echo "Benchmark: ${#IMAGES[@]} images × ${#ALGORITHMS[@]} algos × ${#BANDWIDTHS[@]} bandwidths × ${NUM_RUNS} runs = $total total runs" >&2
echo "Output: $CSV" >&2
echo "" >&2
# ── Run matrix ────────────────────────────────────────────────────────────────
run_idx=0
for img in "${IMAGES[@]}"; do
    [[ -f "$img" ]] || { echo "  SKIP $img (not found)" >&2; continue; }

    # Warmup execution for this image (not recorded)
    echo "Warming up cache for: $img, ${ALGORITHMS[0]}, bw=${BANDWIDTHS[0]}" >&2
    "$BINARY" "$img" "${BANDWIDTHS[0]}" "$MAX_ITER" "${ALGORITHMS[0]}" --no-display --no-output >/dev/null 2>&1 || true

    for algo in "${ALGORITHMS[@]}"; do
        for bw in "${BANDWIDTHS[@]}"; do
            for run in $(seq 1 "$NUM_RUNS"); do
                run_idx=$(( run_idx + 1 ))
                printf "\r[%d/%d] %-15s %-10s bw=%-4d run=%d   " \
                    "$run_idx" "$total" "$img" "$algo" "$bw" "$run" >&2

                if (( DRY_RUN )); then
                    continue
                fi

                # Run the binary and capture stdout
                output=$("$BINARY" "$img" "$bw" "$MAX_ITER" "$algo" --no-display --no-output 2>/dev/null) || {
                    echo "" >&2
                    echo "  FAILED: $img $algo bw=$bw run=$run" >&2
                    continue
                }

                # Parse fields from stdout
                iterations=$(echo "$output" | grep -oP 'Iterations:\s+\K[0-9]+' || echo "0")
                total_ms=$(echo "$output"   | grep -oP 'Total:\s+\K[0-9.]+' || echo "0")
                shift_ms=$(echo "$output"   | grep -oP 'Pixel shifting:\s+\K[0-9.]+' || echo "0")
                avg_iter_ms=$(echo "$output" | grep -oP 'Avg:\s+\K[0-9.]+' || echo "0")

                echo "$img,$algo,$bw,$run,$iterations,$total_ms,$shift_ms,$avg_iter_ms" >> "$CSV"
            done
        done
    done
done

echo "" >&2
echo "Done. Results written to: $CSV" >&2
