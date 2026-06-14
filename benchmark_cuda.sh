#!/usr/bin/env bash
#
# benchmark_cuda.sh — Run CUDA mean shift benchmark across a test matrix.
#
# Usage:
#   bash benchmark_cuda.sh                           # full matrix, 5 runs each
#   bash benchmark_cuda.sh --dry-run                 # print what would run, don't execute
#   bash benchmark_cuda.sh --resume results/foo.csv  # resume an interrupted run
#   bash benchmark_cuda.sh --resume results/foo.csv --dry-run
#
# Resume semantics:
#   - Reads the existing CSV to find already-completed (image,algo,block_size,bw) groups.
#   - A group is "done" if it has >= NUM_RUNS rows in the CSV.
#   - Partial groups (0 < rows < NUM_RUNS) are stripped and re-run from scratch.
#   - New rows are appended to the SAME CSV file (no new timestamped file).
#
# Matrix:
#   Algorithms: cuda, cuda_2d    × images × block_sizes × bandwidths × NUM_RUNS
#
# CSV columns:
#   image,algorithm,block_size,tx,ty,bandwidth,run,iterations,total_ms,shift_ms,convert_ms,avg_iter_ms
#
# Output: results/benchmark_cuda_YYMMDD_HHMMSS.csv  (or the --resume target)

set -euo pipefail

# ── Configuration ─────────────────────────────────────────────────────────────

BINARY="./build/mean_shift_seq"
MAX_ITER=5
NUM_RUNS=5

# All 25 BSD500-derived images — 5 sources × 5 resolution tiers.
# Together with CPU benchmark results, this enables direct AoS vs SoA vs OMP vs
# GPU speedup comparisons at every (tier, bandwidth) point.
IMAGES=(
    # tier 1 — 100×67 — 6 700 px
    "Images/bsd_55067_100x67.jpg"
    "Images/bsd_76002_100x67.jpg"
    "Images/bsd_124084_100x67.jpg"
    "Images/bsd_134052_100x67.jpg"
    "Images/bsd_187039_100x67.jpg"
    # tier 2 — 150×100 — 15 000 px
    "Images/bsd_55067_150x100.jpg"
    "Images/bsd_76002_150x100.jpg"
    "Images/bsd_124084_150x100.jpg"
    "Images/bsd_134052_150x100.jpg"
    "Images/bsd_187039_150x100.jpg"
    # tier 3 — 200×133 — 26 600 px
    "Images/bsd_55067_200x133.jpg"
    "Images/bsd_76002_200x133.jpg"
    "Images/bsd_124084_200x133.jpg"
    "Images/bsd_134052_200x133.jpg"
    "Images/bsd_187039_200x133.jpg"
    # tier 4 — 260×174 — 45 240 px
    "Images/bsd_55067_260x174.jpg"
    "Images/bsd_76002_260x174.jpg"
    "Images/bsd_124084_260x174.jpg"
    "Images/bsd_134052_260x174.jpg"
    "Images/bsd_187039_260x174.jpg"
    # tier 5 — 320×214 — 68 480 px
    "Images/bsd_55067_320x214.jpg"
    "Images/bsd_76002_320x214.jpg"
    "Images/bsd_124084_320x214.jpg"
    "Images/bsd_134052_320x214.jpg"
    "Images/bsd_187039_320x214.jpg"
)

# CUDA algorithms — both 1D and 2D grid for comparison
ALGORITHMS=("cuda" "cuda_2d")

# Block sizes (tile sizes): {128, 256, 512} as documented in cuda_analysis.md
BLOCK_SIZES=(128 256 512)

# Map block_size to (tx, ty) for CSV output
declare -A TX TY
TX[128]=16;    TY[128]=8
TX[256]=16;    TY[256]=16
TX[512]=32;    TY[512]=16

# Bandwidth values — same as CPU benchmark for direct comparison.
# On GPU, bandwidth affects warp divergence (more neighbors = less divergence).
BANDWIDTHS=(20 100 200 500)

# GPU timeout per invocation (seconds)
TIMEOUT=120

# ── Argument parsing ──────────────────────────────────────────────────────────

DRY_RUN=0
RESUME_CSV=""

for arg in "$@"; do
    case "$arg" in
        --dry-run) DRY_RUN=1 ;;
        --resume)  ;;   # handled by shift-pair below
        *)  ;;
    esac
done

i=1
for arg in "$@"; do
    if [[ "$arg" == "--resume" ]]; then
        next="${*:$((i+1)):1}"
        if [[ -z "$next" || "$next" == --* ]]; then
            echo "Error: --resume requires a CSV path argument." >&2
            exit 1
        fi
        RESUME_CSV="$next"
    fi
    i=$(( i + 1 ))
done

if [[ -n "$RESUME_CSV" && ! -f "$RESUME_CSV" ]]; then
    echo "Error: resume file not found: $RESUME_CSV" >&2
    exit 1
fi

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

# ── Resume: load completed groups from existing CSV ───────────────────────────
#
# skip["img|algo|block_size|bw"] = 1  →  group fully done (>= NUM_RUNS rows)

declare -A skip=()

if [[ -n "$RESUME_CSV" ]]; then
    echo "Resume: reading $RESUME_CSV ..." >&2

    declare -A _count=()
    while IFS=, read -r img algo block_size _tx _ty bw _rest; do
        [[ "$img" == "image" ]] && continue
        key="${img}|${algo}|${block_size}|${bw}"
        _count["$key"]=$(( ${_count["$key"]:-0} + 1 ))
    done < "$RESUME_CSV"

    partial_keys=()
    done_count=0
    for key in "${!_count[@]}"; do
        cnt="${_count[$key]}"
        if (( cnt >= NUM_RUNS )); then
            skip["$key"]=1
            done_count=$(( done_count + 1 ))
        else
            partial_keys+=("$key")
        fi
    done

    if (( ${#partial_keys[@]} > 0 )); then
        echo "Resume: found ${#partial_keys[@]} partial group(s) — stripping and re-running:" >&2
        for key in "${partial_keys[@]}"; do
            echo "  partial: $key (${_count[$key]} rows)" >&2
        done

        if (( ! DRY_RUN )); then
            tmp_csv="${RESUME_CSV}.tmp"
            head -1 "$RESUME_CSV" > "$tmp_csv"
            tail -n +2 "$RESUME_CSV" | while IFS=, read -r img algo block_size _tx _ty bw rest; do
                key="${img}|${algo}|${block_size}|${bw}"
                is_partial=0
                for pkey in "${partial_keys[@]}"; do
                    [[ "$key" == "$pkey" ]] && { is_partial=1; break; }
                done
                if (( is_partial == 0 )); then
                    echo "$img,$algo,$block_size,$_tx,$_ty,$bw,$rest"
                fi
            done >> "$tmp_csv"
            mv "$tmp_csv" "$RESUME_CSV"
            echo "Resume: stripped partial rows from CSV." >&2
        fi
    fi

    echo "Resume: $done_count group(s) already complete — will skip." >&2
    echo "" >&2
fi

# ── Output setup ──────────────────────────────────────────────────────────────

mkdir -p results

if [[ -n "$RESUME_CSV" ]]; then
    CSV="$RESUME_CSV"
else
    TIMESTAMP=$(date +%y%m%d_%H%M%S)
    CSV="results/benchmark_cuda_${TIMESTAMP}.csv"
    HEADER="image,algorithm,block_size,tx,ty,bandwidth,run,iterations,total_ms,shift_ms,convert_ms,avg_iter_ms"
    echo "$HEADER" > "$CSV"
fi

# ── Count total runs ──────────────────────────────────────────────────────────

total=0
for img in "${IMAGES[@]}"; do
    [[ -f "$img" ]] || continue
    for algo in "${ALGORITHMS[@]}"; do
        for bs in "${BLOCK_SIZES[@]}"; do
            for bw in "${BANDWIDTHS[@]}"; do
                total=$(( total + NUM_RUNS ))
            done
        done
    done
done

skipped_runs=0
for img in "${IMAGES[@]}"; do
    [[ -f "$img" ]] || continue
    for algo in "${ALGORITHMS[@]}"; do
        for bs in "${BLOCK_SIZES[@]}"; do
            for bw in "${BANDWIDTHS[@]}"; do
                key="${img}|${algo}|${bs}|${bw}"
                [[ "${skip[$key]+isset}" ]] && skipped_runs=$(( skipped_runs + NUM_RUNS ))
            done
        done
    done
done

echo "CUDA Benchmark:" >&2
echo "  Algorithms: ${ALGORITHMS[*]}" >&2
echo "  Block sizes: ${BLOCK_SIZES[*]}" >&2
echo "  Images    : ${#IMAGES[@]}  Bandwidths: ${#BANDWIDTHS[@]}  Runs: ${NUM_RUNS}" >&2
echo "  Total runs: $total  (skipping $skipped_runs already done)" >&2
echo "  Output    : $CSV" >&2
echo "" >&2

# ── Helpers ───────────────────────────────────────────────────────────────────

is_done() {
    local key="${1}|${2}|${3}|${4}"   # img algo block_size bw
    [[ "${skip[$key]+isset}" ]]
}

# ── Run matrix ────────────────────────────────────────────────────────────────

run_idx=$skipped_runs

run_one() {
    local img="$1" algo="$2" bs="$3" bw="$4" run="$5"
    local tx="${TX[$bs]}"
    local ty="${TY[$bs]}"
    run_idx=$(( run_idx + 1 ))
    printf "\r[%d/%d] %-30s %-8s bs=%-3d bw=%-4d run=%d   " \
        "$run_idx" "$total" "$img" "$algo" "$bs" "$bw" "$run" >&2

    if (( DRY_RUN )); then
        return
    fi

    local output
    output=$(timeout "$TIMEOUT" "$BINARY" "$img" "$bw" "$MAX_ITER" "$algo" \
        --no-display --no-output --block-size "$bs" 2>/dev/null) || {
        echo "" >&2
        echo "  FAILED/TIMEOUT: $img $algo bs=$bs bw=$bw run=$run" >&2
        return
    }

    local iterations total_ms shift_ms convert_ms avg_iter_ms
    iterations=$(echo "$output"  | grep -oP 'Iterations:\s+\K[0-9]+'       || echo "0")
    total_ms=$(echo "$output"    | grep -oP 'Total:\s+\K[0-9.]+'            || echo "0")
    shift_ms=$(echo "$output"    | grep -oP 'Pixel shifting:\s+\K[0-9.]+'   || echo "0")
    convert_ms=$(echo "$output"  | grep -oP 'Convert:\s+\K[0-9.]+'          || echo "0")
    avg_iter_ms=$(echo "$output" | grep -oP 'Avg:\s+\K[0-9.]+'              || echo "0")

    echo "$img,$algo,$bs,$tx,$ty,$bw,$run,$iterations,$total_ms,$shift_ms,$convert_ms,$avg_iter_ms" >> "$CSV"
}

for img in "${IMAGES[@]}"; do
    [[ -f "$img" ]] || { echo "  SKIP $img (not found)" >&2; continue; }

    for algo in "${ALGORITHMS[@]}"; do
        for bs in "${BLOCK_SIZES[@]}"; do
            for bw in "${BANDWIDTHS[@]}"; do
                if is_done "$img" "$algo" "$bs" "$bw"; then
                    continue
                fi

                # Warmup run (using the first bandwidth value)
                if (( ! DRY_RUN )); then
                    timeout "$TIMEOUT" "$BINARY" "$img" "${BANDWIDTHS[0]}" "$MAX_ITER" "$algo" \
                        --no-display --no-output --block-size "$bs" >/dev/null 2>&1 || true
                fi

                for run in $(seq 1 "$NUM_RUNS"); do
                    run_one "$img" "$algo" "$bs" "$bw" "$run"
                done
            done
        done
    done
done

echo "" >&2
echo "Done. Results written to: $CSV" >&2
