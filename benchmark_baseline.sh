#!/usr/bin/env bash
#
# benchmark_baseline.sh — Benchmark the unoptimised baseline algorithm.
#
# Usage:
#   bash benchmark_baseline.sh                              # full matrix, 5 runs each
#   bash benchmark_baseline.sh --dry-run                    # print what would run, don't execute
#   bash benchmark_baseline.sh --resume results/foo.csv     # resume an interrupted run
#   bash benchmark_baseline.sh --resume results/foo.csv --dry-run
#
# Resume semantics:
#   - Reads the existing CSV to find already-completed (image,algo,threads,bw) groups.
#   - A group is "done" if it has >= NUM_RUNS rows in the CSV.
#   - Partial groups (0 < rows < NUM_RUNS) are stripped and re-run from scratch.
#   - New rows are appended to the SAME CSV file (no new timestamped file).
#
# Why only tiers 1–3:
#   baseline recomputes x,y coordinates inside the O(n^2) inner loop and
#   allocates next[] fresh every iteration. Estimated ~1.5–3x slower than seq.
#   Tier 4 (45K px) and tier 5 (68K px) would each take several hours per
#   source image at 1 thread — tiers 1–3 keep the total run under ~3 hours.
#
# Output: results/benchmark_baseline_YYMMDD_HHMMSS.csv  (or --resume target)

set -euo pipefail

# ── Configuration ─────────────────────────────────────────────────────────────

BINARY="./build/mean_shift_seq"
MAX_ITER=5
NUM_RUNS=5
KERNEL="flat"

# Tiers 1–3 only — all 5 BSD500 sources at each tier.
# Post-processing averages timing across sources within each tier.
#
#   Tier | WxH      | Pixels  | Sources
#   -----+----------+---------+----------------------------------
#     1  | 100x67   |   6.7K  | 55067, 76002, 124084, 134052, 187039
#     2  | 150x100  |  15.0K  | 55067, 76002, 124084, 134052, 187039
#     3  | 200x133  |  26.6K  | 55067, 76002, 124084, 134052, 187039
#
# Estimated baseline cost per run at 1 thread, MAX_ITER=5:
#   tier 1 ~2s, tier 2 ~8–15s, tier 3 ~25–48s
IMAGES=(
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
BANDWIDTHS=(20 50 100)

# ── Argument parsing ──────────────────────────────────────────────────────────

DRY_RUN=0
RESUME_CSV=""

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

for arg in "$@"; do
    case "$arg" in
        --dry-run) DRY_RUN=1 ;;
        --resume)  ;;
        *)         ;;
    esac
done

if [[ -n "$RESUME_CSV" && ! -f "$RESUME_CSV" ]]; then
    echo "Error: resume file not found: $RESUME_CSV" >&2
    exit 1
fi

# ── Pre-flight checks ─────────────────────────────────────────────────────────

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

# ── Resume: load completed groups ─────────────────────────────────────────────
#
# skip["img|algo|threads|bw"] = 1  →  group fully done (>= NUM_RUNS rows)

declare -A skip=()

if [[ -n "$RESUME_CSV" ]]; then
    echo "Resume: reading $RESUME_CSV ..." >&2

    declare -A _count=()
    while IFS=, read -r img algo _kern threads bw _rest; do
        [[ "$img" == "image" ]] && continue
        key="${img}|${algo}|${threads}|${bw}"
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
else
    TIMESTAMP=$(date +%y%m%d_%H%M%S)
    CSV="results/benchmark_baseline_${TIMESTAMP}.csv"
    HEADER="image,algorithm,kernel,threads,bandwidth,run,iterations,total_ms,shift_ms,convert_ms,avg_iter_ms"
    echo "$HEADER" > "$CSV"
fi

# ── Count total runs ──────────────────────────────────────────────────────────

total=0
for img in "${IMAGES[@]}"; do
    [[ -f "$img" ]] || continue
    for bw in "${BANDWIDTHS[@]}"; do
        total=$(( total + NUM_RUNS ))
    done
done

skipped_runs=0
for img in "${IMAGES[@]}"; do
    [[ -f "$img" ]] || continue
    for bw in "${BANDWIDTHS[@]}"; do
        key="${img}|baseline|1|${bw}"
        [[ "${skip[$key]+isset}" ]] && skipped_runs=$(( skipped_runs + NUM_RUNS ))
    done
done

echo "Benchmark (baseline only):" >&2
echo "  Algorithm  : baseline — threads=1 (sequential)" >&2
echo "  Images     : ${#IMAGES[@]}  (tiers 1–3, 5 sources each)" >&2
echo "  Bandwidths : ${BANDWIDTHS[*]}" >&2
echo "  Runs       : ${NUM_RUNS}" >&2
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
    local img="$1" bw="$2" run="$3"
    run_idx=$(( run_idx + 1 ))
    printf "\r[%d/%d] %-40s threads=1   bw=%-4d run=%d   " \
        "$run_idx" "$total" "$img" "$bw" "$run" >&2

    if (( DRY_RUN )); then
        return
    fi

    local output
    output=$(OMP_NUM_THREADS=1 \
        timeout 360 "$BINARY" "$img" "$bw" "$MAX_ITER" baseline \
            --kernel "$KERNEL" --no-display --no-output 2>/dev/null) || {
        echo "" >&2
        echo "  FAILED/TIMEOUT: $img baseline bw=$bw run=$run" >&2
        return
    }

    local iterations total_ms shift_ms convert_ms avg_iter_ms
    iterations=$(echo "$output"  | grep -oP 'Iterations:\s+\K[0-9]+'       || echo "0")
    total_ms=$(echo "$output"    | grep -oP 'Total:\s+\K[0-9.]+'            || echo "0")
    shift_ms=$(echo "$output"    | grep -oP 'Pixel shifting:\s+\K[0-9.]+'   || echo "0")
    convert_ms=$(echo "$output"  | grep -oP 'Convert:\s+\K[0-9.]+'          || echo "0")
    avg_iter_ms=$(echo "$output" | grep -oP 'Avg:\s+\K[0-9.]+'              || echo "0")

    echo "$img,baseline,$KERNEL,1,$bw,$run,$iterations,$total_ms,$shift_ms,$convert_ms,$avg_iter_ms" >> "$CSV"
}

for img in "${IMAGES[@]}"; do
    [[ -f "$img" ]] || { echo "  SKIP $img (not found)" >&2; continue; }

    for bw in "${BANDWIDTHS[@]}"; do
        if is_done "$img" "baseline" "1" "$bw"; then
            continue
        fi

        # Warmup run
        if (( ! DRY_RUN )); then
            OMP_NUM_THREADS=1 timeout 360 "$BINARY" "$img" "${BANDWIDTHS[0]}" "$MAX_ITER" baseline \
                --kernel "$KERNEL" --no-display --no-output >/dev/null 2>&1 || true
        fi

        for run in $(seq 1 "$NUM_RUNS"); do
            run_one "$img" "$bw" "$run"
        done
    done
done

echo "" >&2
echo "Done. Results written to: $CSV" >&2
