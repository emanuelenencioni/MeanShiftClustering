#!/usr/bin/env bash
#
# benchmark_large.sh — Compare cuda vs cuda_2d vs no-tiling on large images.
#
# Usage:
#   bash benchmark_large.sh                           # full matrix, 3 runs each
#   bash benchmark_large.sh --dry-run                 # print what would run
#   bash benchmark_large.sh --resume results/foo.csv  # resume interrupted run
#
# Matrix:
#   3 images × 3 algo-modes × 3 block_sizes × 2 bandwidths × 3 runs = 162 runs
#
# Algo-modes: cuda(tiled), cuda_2d(tiled), cuda(notiled)
#
# CSV columns:
#   image,algorithm,block_size,tx,ty,bandwidth,tiled,run,iterations,total_ms,shift_ms,convert_ms,avg_iter_ms

set -euo pipefail

BINARY="./build/mean_shift_seq"
MAX_ITER=5
NUM_RUNS=3

IMAGES=(
    "Images/bsd_187039_320x214.jpg"    # tier 5  —  68.5K px (baseline)
    "Images/720p_nature.jpg"           # 1280×720 — 922K px
    "Images/nature_1080p.jpg"          # 1920×1080 — 2.07M px
)

BLOCK_SIZES=(128 256 512)

declare -A TX TY
TX[128]=16;    TY[128]=8
TX[256]=16;    TY[256]=16
TX[512]=32;    TY[512]=16

BANDWIDTHS=(100 200)

# 1080p non-tiled can take ~6 min per run
TIMEOUT=600

# ── Argument parsing ──────────────────────────────────────────────────────────

DRY_RUN=0
RESUME_CSV=""

for arg in "$@"; do
    case "$arg" in --dry-run) DRY_RUN=1 ;; --resume) ;; esac
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

# ── Pre-flight ────────────────────────────────────────────────────────────────

if [[ ! -x "$BINARY" ]]; then
    echo "Binary not found: $BINARY — run: cmake --build build" >&2
    exit 1
fi

missing=()
for img in "${IMAGES[@]}"; do
    [[ -f "$img" ]] || missing+=("$img")
done
if (( ${#missing[@]} > 0 )); then
    echo "Warning: missing images (will be skipped): ${missing[*]}" >&2
fi

# ── Algo-mode definitions ─────────────────────────────────────────────────────
# Each entry: "algorithm,tiled_flag,extra_flags"
ALGO_MODES=(
    "cuda,true,"
    "cuda_2d,true,"
    "cuda,false,--no-tiling"
)

# ── Resume ────────────────────────────────────────────────────────────────────

declare -A skip=()

load_resume() {
    local csv="$1"
    echo "Resume: reading $csv ..." >&2
    declare -A _count=()
    while IFS=, read -r img algo bs _tx _ty bw tiled _rest; do
        [[ "$img" == "image" ]] && continue
        key="${img}|${algo}|${bs}|${bw}|${tiled}"
        _count["$key"]=$(( ${_count["$key"]:-0} + 1 ))
    done < "$csv"

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
            tmp_csv="${csv}.tmp"
            head -1 "$csv" > "$tmp_csv"
            tail -n +2 "$csv" | while IFS=, read -r img algo bs _tx _ty bw tiled rest; do
                key="${img}|${algo}|${bs}|${bw}|${tiled}"
                is_partial=0
                for pkey in "${partial_keys[@]}"; do
                    [[ "$key" == "$pkey" ]] && { is_partial=1; break; }
                done
                (( is_partial == 0 )) && echo "$img,$algo,$bs,$_tx,$_ty,$bw,$tiled,$rest"
            done >> "$tmp_csv"
            mv "$tmp_csv" "$csv"
            echo "Resume: stripped partial rows from CSV." >&2
        fi
    fi
    echo "Resume: $done_count group(s) already complete — will skip." >&2
    echo "" >&2
}

if [[ -n "$RESUME_CSV" ]]; then
    load_resume "$RESUME_CSV"
fi

# ── Output ────────────────────────────────────────────────────────────────────

mkdir -p results

if [[ -n "$RESUME_CSV" ]]; then
    CSV="$RESUME_CSV"
else
    TIMESTAMP=$(date +%y%m%d_%H%M%S)
    CSV="results/benchmark_large_${TIMESTAMP}.csv"
    HEADER="image,algorithm,block_size,tx,ty,bandwidth,tiled,run,iterations,total_ms,shift_ms,convert_ms,avg_iter_ms"
    echo "$HEADER" > "$CSV"
fi

# ── Count ─────────────────────────────────────────────────────────────────────

total=0
for img in "${IMAGES[@]}"; do
    [[ -f "$img" ]] || continue
    for am in "${ALGO_MODES[@]}"; do
        IFS=',' read -r _algo _tiled _flags <<< "$am"
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
    for am in "${ALGO_MODES[@]}"; do
        IFS=',' read -r _algo _tiled _flags <<< "$am"
        for bs in "${BLOCK_SIZES[@]}"; do
            for bw in "${BANDWIDTHS[@]}"; do
                key="${img}|${_algo}|${bs}|${bw}|${_tiled}"
                [[ "${skip[$key]+isset}" ]] && skipped_runs=$(( skipped_runs + NUM_RUNS ))
            done
        done
    done
done

echo "Large-image benchmark:" >&2
echo "  Images    : ${#IMAGES[@]} (320p, 720p, 1080p)" >&2
echo "  Modes     : ${#ALGO_MODES[@]} (cuda tiled, cuda_2d, cuda notiled)" >&2
echo "  Block sizes: ${BLOCK_SIZES[*]}" >&2
echo "  Bandwidths: ${BANDWIDTHS[*]}" >&2
echo "  Runs      : ${NUM_RUNS}" >&2
echo "  Total     : $total  (skipping $skipped_runs already done)" >&2
echo "  Output    : $CSV" >&2
echo "  Est. time : ~3-5 hours (1080p non-tiled is ~6 min/run)" >&2
echo "" >&2

# ── Helpers ───────────────────────────────────────────────────────────────────

is_done() {
    local key="${1}|${2}|${3}|${4}|${5}"   # img algo bs bw tiled
    [[ "${skip[$key]+isset}" ]]
}

run_idx=$skipped_runs

run_one() {
    local img="$1" algo="$2" tiled="$3" extra="$4" bs="$5" bw="$6" run="$7"
    local tx="${TX[$bs]}"
    local ty="${TY[$bs]}"
    local mode_label="${algo}"
    [[ "$tiled" == "true" ]] && mode_label+=",tiled" || mode_label+=",notiled"

    run_idx=$(( run_idx + 1 ))
    printf "\r[%d/%d] %-30s algo=%-12s bs=%-3d bw=%-4d run=%d   " \
        "$run_idx" "$total" "$img" "$mode_label" "$bs" "$bw" "$run" >&2

    if (( DRY_RUN )); then
        return
    fi

    local output
    output=$(timeout "$TIMEOUT" "$BINARY" "$img" "$bw" "$MAX_ITER" "$algo" \
        --no-display --no-output --block-size "$bs" $extra 2>/dev/null) || {
        echo "" >&2
        echo "  FAILED/TIMEOUT: $img $mode_label bs=$bs bw=$bw run=$run" >&2
        return
    }

    local iterations total_ms shift_ms convert_ms avg_iter_ms
    iterations=$(echo "$output"  | grep -oP 'Iterations:\s+\K[0-9]+'       || echo "0")
    total_ms=$(echo "$output"    | grep -oP 'Total:\s+\K[0-9.]+'            || echo "0")
    shift_ms=$(echo "$output"    | grep -oP 'Pixel shifting:\s+\K[0-9.]+'   || echo "0")
    convert_ms=$(echo "$output"  | grep -oP 'Convert:\s+\K[0-9.]+'          || echo "0")
    avg_iter_ms=$(echo "$output" | grep -oP 'Avg:\s+\K[0-9.]+'              || echo "0")

    echo "$img,$algo,$bs,$tx,$ty,$bw,$tiled,$run,$iterations,$total_ms,$shift_ms,$convert_ms,$avg_iter_ms" >> "$CSV"
}

# ── Run matrix ────────────────────────────────────────────────────────────────

for img in "${IMAGES[@]}"; do
    [[ -f "$img" ]] || { echo "  SKIP $img (not found)" >&2; continue; }

    for am in "${ALGO_MODES[@]}"; do
        IFS=',' read -r algo tiled extra_flags <<< "$am"

        for bs in "${BLOCK_SIZES[@]}"; do
            for bw in "${BANDWIDTHS[@]}"; do
                if is_done "$img" "$algo" "$bs" "$bw" "$tiled"; then
                    continue
                fi

                # Warmup
                if (( ! DRY_RUN )); then
                    timeout "$TIMEOUT" "$BINARY" "$img" "${BANDWIDTHS[0]}" "$MAX_ITER" "$algo" \
                        --no-display --no-output --block-size "$bs" $extra_flags >/dev/null 2>&1 || true
                fi

                for run in $(seq 1 "$NUM_RUNS"); do
                    run_one "$img" "$algo" "$tiled" "$extra_flags" "$bs" "$bw" "$run"
                done
            done
        done
    done
done

echo "" >&2
echo "Done. Results written to: $CSV" >&2
