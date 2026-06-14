#!/usr/bin/env bash
#
# benchmark_tile.sh — Compare tiled vs non-tiled CUDA mean shift kernels.
#
# Usage:
#   bash benchmark_tile.sh                           # full matrix, 3 runs each
#   bash benchmark_tile.sh --dry-run                 # print what would run
#   bash benchmark_tile.sh --resume results/foo.csv  # resume interrupted run
#
# Matrix:
#   cuda  (tiled)      × 5 images × 3 block_sizes × 2 bandwidths × 3 runs
#   cuda  (non-tiled)  × 5 images × 3 block_sizes × 2 bandwidths × 3 runs
#   Total: 180 runs
#
# CSV columns:
#   image,algorithm,block_size,tx,ty,bandwidth,tiled,run,iterations,total_ms,shift_ms,convert_ms,avg_iter_ms

set -euo pipefail

BINARY="./build/mean_shift_seq"
MAX_ITER=5
NUM_RUNS=3

# One source per tier — representative colour complexity
IMAGES=(
    "Images/bsd_55067_100x67.jpg"      # tier 1  —  6.7K px, low complexity
    "Images/bsd_76002_150x100.jpg"     # tier 2  — 15.0K px, medium-low
    "Images/bsd_124084_200x133.jpg"    # tier 3  — 26.6K px, medium
    "Images/bsd_134052_260x174.jpg"    # tier 4  — 45.2K px, high
    "Images/bsd_187039_320x214.jpg"    # tier 5  — 68.5K px, very high
)

BLOCK_SIZES=(128 256 512)

declare -A TX TY
TX[128]=16;    TY[128]=8
TX[256]=16;    TY[256]=16
TX[512]=32;    TY[512]=16

BANDWIDTHS=(20 100)

# Non-tiled needs more time (global reads are ~2× slower at L2 scale,
# but can spike higher under contention)
TIMEOUT=300

# ── Argument parsing ──────────────────────────────────────────────────────────

DRY_RUN=0
RESUME_CSV=""

for arg in "$@"; do
    case "$arg" in
        --dry-run) DRY_RUN=1 ;;
        --resume)  ;;
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

# ── Resume ────────────────────────────────────────────────────────────────────
# skip key: "img|block_size|bw|tiled"

declare -A skip=()

if [[ -n "$RESUME_CSV" ]]; then
    echo "Resume: reading $RESUME_CSV ..." >&2

    declare -A _count=()
    while IFS=, read -r img _algo bs _tx _ty bw tiled _rest; do
        [[ "$img" == "image" ]] && continue
        key="${img}|${bs}|${bw}|${tiled}"
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
            tail -n +2 "$RESUME_CSV" | while IFS=, read -r img algo bs _tx _ty bw tiled rest; do
                key="${img}|${bs}|${bw}|${tiled}"
                is_partial=0
                for pkey in "${partial_keys[@]}"; do
                    [[ "$key" == "$pkey" ]] && { is_partial=1; break; }
                done
                (( is_partial == 0 )) && echo "$img,$algo,$bs,$_tx,$_ty,$bw,$tiled,$rest"
            done >> "$tmp_csv"
            mv "$tmp_csv" "$RESUME_CSV"
            echo "Resume: stripped partial rows from CSV." >&2
        fi
    fi
    echo "Resume: $done_count group(s) already complete — will skip." >&2
    echo "" >&2
fi

# ── Output ────────────────────────────────────────────────────────────────────

mkdir -p results

if [[ -n "$RESUME_CSV" ]]; then
    CSV="$RESUME_CSV"
else
    TIMESTAMP=$(date +%y%m%d_%H%M%S)
    CSV="results/benchmark_tile_${TIMESTAMP}.csv"
    HEADER="image,algorithm,block_size,tx,ty,bandwidth,tiled,run,iterations,total_ms,shift_ms,convert_ms,avg_iter_ms"
    echo "$HEADER" > "$CSV"
fi

# ── Count ─────────────────────────────────────────────────────────────────────

total=0
for img in "${IMAGES[@]}"; do
    [[ -f "$img" ]] || continue
    for bs in "${BLOCK_SIZES[@]}"; do
        for bw in "${BANDWIDTHS[@]}"; do
            total=$(( total + 2 * NUM_RUNS ))   # tiled + non-tiled
        done
    done
done

skipped_runs=0
for img in "${IMAGES[@]}"; do
    [[ -f "$img" ]] || continue
    for bs in "${BLOCK_SIZES[@]}"; do
        for bw in "${BANDWIDTHS[@]}"; do
            for tiled in "true" "false"; do
                key="${img}|${bs}|${bw}|${tiled}"
                [[ "${skip[$key]+isset}" ]] && skipped_runs=$(( skipped_runs + NUM_RUNS ))
            done
        done
    done
done

echo "Tile benchmark:" >&2
echo "  Images    : ${#IMAGES[@]}  (1 per tier)" >&2
echo "  Block sizes: ${BLOCK_SIZES[*]}" >&2
echo "  Bandwidths: ${BANDWIDTHS[*]}" >&2
echo "  Modes     : tiled + non-tiled" >&2
echo "  Runs      : ${NUM_RUNS}" >&2
echo "  Total     : $total  (skipping $skipped_runs already done)" >&2
echo "  Output    : $CSV" >&2
echo "" >&2

# ── Helpers ───────────────────────────────────────────────────────────────────

is_done() {
    local key="${1}|${2}|${3}|${4}"   # img bs bw tiled
    [[ "${skip[$key]+isset}" ]]
}

# ── Run matrix ────────────────────────────────────────────────────────────────

run_idx=$skipped_runs

run_one() {
    local img="$1" bs="$2" bw="$3" tiled="$4" run="$5"
    local tx="${TX[$bs]}"
    local ty="${TY[$bs]}"
    local mode_label="tiled"
    local extra_flags=""
    if [[ "$tiled" != "true" ]]; then
        mode_label="notile"
        extra_flags="--no-tiling"
    fi

    run_idx=$(( run_idx + 1 ))
    printf "\r[%d/%d] %-30s bs=%-3d bw=%-4d %-6s run=%d   " \
        "$run_idx" "$total" "$img" "$bs" "$bw" "$mode_label" "$run" >&2

    if (( DRY_RUN )); then
        return
    fi

    local output
    output=$(timeout "$TIMEOUT" "$BINARY" "$img" "$bw" "$MAX_ITER" cuda \
        --no-display --no-output --block-size "$bs" $extra_flags 2>/dev/null) || {
        echo "" >&2
        echo "  FAILED/TIMEOUT: $img bs=$bs bw=$bw $mode_label run=$run" >&2
        return
    }

    local iterations total_ms shift_ms convert_ms avg_iter_ms
    iterations=$(echo "$output"  | grep -oP 'Iterations:\s+\K[0-9]+'       || echo "0")
    total_ms=$(echo "$output"    | grep -oP 'Total:\s+\K[0-9.]+'            || echo "0")
    shift_ms=$(echo "$output"    | grep -oP 'Pixel shifting:\s+\K[0-9.]+'   || echo "0")
    convert_ms=$(echo "$output"  | grep -oP 'Convert:\s+\K[0-9.]+'          || echo "0")
    avg_iter_ms=$(echo "$output" | grep -oP 'Avg:\s+\K[0-9.]+'              || echo "0")

    echo "$img,cuda,$bs,$tx,$ty,$bw,$tiled,$run,$iterations,$total_ms,$shift_ms,$convert_ms,$avg_iter_ms" >> "$CSV"
}

for img in "${IMAGES[@]}"; do
    [[ -f "$img" ]] || { echo "  SKIP $img (not found)" >&2; continue; }

    for bs in "${BLOCK_SIZES[@]}"; do
        for bw in "${BANDWIDTHS[@]}"; do
            for tiled in "true" "false"; do
                if is_done "$img" "$bs" "$bw" "$tiled"; then
                    continue
                fi

                # Warmup
                if (( ! DRY_RUN )); then
                    warmup_flags=""
                    [[ "$tiled" != "true" ]] && warmup_flags="--no-tiling"
                    timeout "$TIMEOUT" "$BINARY" "$img" "${BANDWIDTHS[0]}" "$MAX_ITER" cuda \
                        --no-display --no-output --block-size "$bs" $warmup_flags >/dev/null 2>&1 || true
                fi

                for run in $(seq 1 "$NUM_RUNS"); do
                    run_one "$img" "$bs" "$bw" "$tiled" "$run"
                done
            done
        done
    done
done

echo "" >&2
echo "Done. Results written to: $CSV" >&2
