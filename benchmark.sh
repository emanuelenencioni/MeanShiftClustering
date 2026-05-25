#!/usr/bin/env bash
#
# benchmark.sh — Run mean shift benchmark across a test matrix and output CSV.
#
# Usage:
#   bash benchmark.sh                              # full matrix, 5 runs each
#   bash benchmark.sh --dry-run                    # print what would run, don't execute
#   bash benchmark.sh --resume results/foo.csv     # resume an interrupted run
#   bash benchmark.sh --resume results/foo.csv --dry-run
#
# Resume semantics:
#   - Reads the existing CSV to find already-completed (image,algo,threads,bw) groups.
#   - A group is "done" if it has >= NUM_RUNS rows in the CSV.
#   - Partial groups (0 < rows < NUM_RUNS) are stripped from the CSV and re-run
#     from scratch to avoid duplicate/mixed data.
#   - New rows are appended to the SAME CSV file (no new timestamped file).
#
# Matrix:
#   Sequential : seq, soa          × images × bandwidths × 1 thread  × NUM_RUNS
#   Parallel   : omp, omp_soa      × images × bandwidths × THREADS   × NUM_RUNS
#
# Output: results/benchmark_YYMMDD_HHMMSS.csv  (or the --resume target)

set -euo pipefail

# ── Configuration ─────────────────────────────────────────────────────────────

BINARY="./build/mean_shift_seq"
# MAX_ITER=5: runs 5 iterations per benchmark invocation.
# Enough to measure realistic per-iteration cost while keeping total runtime
# manageable (~20 hours for the full matrix on a 12-core machine — run overnight).
# All variants are brute-force O(n^2); images capped at ~68K pixels so that
# seq at 1 thread stays under ~105 s per run.
MAX_ITER=5
NUM_RUNS=5

# Dataset: all 25 BSD500-derived images — 5 sources × 5 resolution tiers.
# Post-processing averages timing results across the 5 sources within each
# tier, giving a tier-representative mean that is not biased by any single
# source image's content complexity.
#
# Resolution tiers (pixel counts follow an approximately geometric progression):
#
#   Tier | WxH      | Pixels  | Source IDs (all 5 run at this tier)
#   -----+----------+---------+-------------------------------------
#     1  | 100x67   |   6.7K  | 55067, 76002, 124084, 134052, 187039
#     2  | 150x100  |  15.0K  | 55067, 76002, 124084, 134052, 187039
#     3  | 200x133  |  26.6K  | 55067, 76002, 124084, 134052, 187039
#     4  | 260x174  |  45.2K  | 55067, 76002, 124084, 134052, 187039
#     5  | 320x214  |  68.5K  | 55067, 76002, 124084, 134052, 187039
#
# Estimated seq 1T per run: tier1 ~0.2s, tier5 ~105s.
# Total runs: 25 images × (2 seq + 2 omp × 6 threads) × 3 bw × 5 runs = 5250.
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

# Sequential algorithms — always run with threads=1
SEQ_ALGORITHMS=("seq" "soa")

# Parallel algorithms — run once per entry in THREADS
OMP_ALGORITHMS=("omp" "omp_soa")

BANDWIDTHS=(20 50 100)

THREADS=(1 2 4 8 12 24)

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

# Two-pass: pick up --resume <value>
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

# Validate unknown args
for arg in "$@"; do
    case "$arg" in
        --dry-run|--resume) ;;
        *)
            # skip values that follow --resume
            ;;
    esac
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
# skip["img|algo|threads|bw"] = 1   → group fully done (>= NUM_RUNS rows)
# partial groups are stripped from the CSV before we start appending

declare -A skip=()

if [[ -n "$RESUME_CSV" ]]; then
    echo "Resume: reading $RESUME_CSV ..." >&2

    # Count rows per (image,algo,threads,bw) key.
    # CSV columns: image,algorithm,threads,bandwidth,run,...
    #              1     2         3       4         5
    declare -A _count=()
    while IFS=, read -r img algo threads bw _rest; do
        [[ "$img" == "image" ]] && continue   # skip header
        key="${img}|${algo}|${threads}|${bw}"
        _count["$key"]=$(( ${_count["$key"]:-0} + 1 ))
    done < "$RESUME_CSV"

    # Classify: full vs partial
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
            # Build a grep pattern that matches any partial-key row and remove them.
            # We rewrite the CSV keeping only the header + rows NOT matching any partial key.
            tmp_csv="${RESUME_CSV}.tmp"
            # Write header
            head -1 "$RESUME_CSV" > "$tmp_csv"
            # For each data row, check if its key is partial; if so, drop it.
            tail -n +2 "$RESUME_CSV" | while IFS=, read -r img algo _kern threads bw rest; do
                key="${img}|${algo}|${threads}|${bw}"
                is_partial=0
                for pkey in "${partial_keys[@]}"; do
                    [[ "$key" == "$pkey" ]] && { is_partial=1; break; }
                done
                if (( is_partial == 0 )); then
                    echo "$img,$algo,$_kern,$threads,$bw,$rest"
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
    # Do not write header again — it already exists
else
    TIMESTAMP=$(date +%y%m%d_%H%M%S)
    CSV="results/benchmark_${TIMESTAMP}.csv"
    HEADER="image,algorithm,threads,bandwidth,run,iterations,total_ms,shift_ms,convert_ms,avg_iter_ms"
    echo "$HEADER" > "$CSV"
fi

# ── Count total runs ──────────────────────────────────────────────────────────

total=0
for img in "${IMAGES[@]}"; do
    [[ -f "$img" ]] || continue
    for bw in "${BANDWIDTHS[@]}"; do
        total=$(( total + ${#SEQ_ALGORITHMS[@]} * NUM_RUNS ))
        total=$(( total + ${#OMP_ALGORITHMS[@]} * ${#THREADS[@]} * NUM_RUNS ))
    done
done

# Count already-skipped runs to initialise the progress counter correctly
skipped_runs=0
for img in "${IMAGES[@]}"; do
    [[ -f "$img" ]] || continue
    for bw in "${BANDWIDTHS[@]}"; do
        for algo in "${SEQ_ALGORITHMS[@]}"; do
            key="${img}|${algo}|1|${bw}"
            [[ "${skip[$key]+isset}" ]] && skipped_runs=$(( skipped_runs + NUM_RUNS ))
        done
        for algo in "${OMP_ALGORITHMS[@]}"; do
            for threads in "${THREADS[@]}"; do
                key="${img}|${algo}|${threads}|${bw}"
                [[ "${skip[$key]+isset}" ]] && skipped_runs=$(( skipped_runs + NUM_RUNS ))
            done
        done
    done
done

echo "Benchmark:" >&2
echo "  Sequential : ${SEQ_ALGORITHMS[*]} — threads=1" >&2
echo "  Parallel   : ${OMP_ALGORITHMS[*]} — threads=${THREADS[*]}" >&2
echo "  Images     : ${#IMAGES[@]}  Bandwidths: ${#BANDWIDTHS[@]}  Runs: ${NUM_RUNS}" >&2
echo "  Total runs : $total  (skipping $skipped_runs already done)" >&2
echo "  Output     : $CSV" >&2
echo "" >&2

# ── Helpers ───────────────────────────────────────────────────────────────────

is_done() {
    local key="${1}|${2}|${3}|${4}"   # img algo threads bw
    [[ "${skip[$key]+isset}" ]]
}

# ── Run matrix ────────────────────────────────────────────────────────────────

run_idx=$skipped_runs

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
        timeout 360 "$BINARY" "$img" "$bw" "$MAX_ITER" "$algo" \
            --no-display --no-output 2>/dev/null) || {
        echo "" >&2
        echo "  FAILED/TIMEOUT: $img $algo threads=$threads bw=$bw run=$run" >&2
        return
    }

    local iterations total_ms shift_ms convert_ms avg_iter_ms
    iterations=$(echo "$output"  | grep -oP 'Iterations:\s+\K[0-9]+'       || echo "0")
    total_ms=$(echo "$output"    | grep -oP 'Total:\s+\K[0-9.]+'            || echo "0")
    shift_ms=$(echo "$output"    | grep -oP 'Pixel shifting:\s+\K[0-9.]+'   || echo "0")
    convert_ms=$(echo "$output"  | grep -oP 'Convert:\s+\K[0-9.]+'          || echo "0")
    avg_iter_ms=$(echo "$output" | grep -oP 'Avg:\s+\K[0-9.]+'              || echo "0")

    echo "$img,$algo,$threads,$bw,$run,$iterations,$total_ms,$shift_ms,$convert_ms,$avg_iter_ms" >> "$CSV"
}

for img in "${IMAGES[@]}"; do
    [[ -f "$img" ]] || { echo "  SKIP $img (not found)" >&2; continue; }

    for bw in "${BANDWIDTHS[@]}"; do
        # Sequential algorithms — fixed threads=1
        for algo in "${SEQ_ALGORITHMS[@]}"; do
            if is_done "$img" "$algo" "1" "$bw"; then
                continue
            fi
            # Warmup
            if (( ! DRY_RUN )); then
                OMP_NUM_THREADS=1 timeout 360 "$BINARY" "$img" "${BANDWIDTHS[0]}" "$MAX_ITER" "$algo" \
                    --no-display --no-output >/dev/null 2>&1 || true
            fi
            for run in $(seq 1 "$NUM_RUNS"); do
                run_one "$img" "$algo" 1 "$bw" "$run"
            done
        done

        # Parallel algorithms — iterate over thread counts
        for algo in "${OMP_ALGORITHMS[@]}"; do
            for threads in "${THREADS[@]}"; do
                if is_done "$img" "$algo" "$threads" "$bw"; then
                    continue
                fi
                # Warmup
                if (( ! DRY_RUN )); then
                    OMP_NUM_THREADS="$threads" timeout 360 "$BINARY" "$img" "${BANDWIDTHS[0]}" "$MAX_ITER" "$algo" \
                        --no-display --no-output >/dev/null 2>&1 || true
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
