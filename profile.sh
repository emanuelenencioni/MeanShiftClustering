#!/usr/bin/env bash
#
# profile.sh — Hardware-counter profiling via perf stat.
#
# Runs three targeted experiments using a small representative subset of
# configurations (not the full benchmark matrix) and writes per-event CSV rows.
#
# Usage:
#   bash profile.sh                                    # all three experiments
#   bash profile.sh --experiment layout                # one experiment only
#   bash profile.sh --experiment scaling
#   bash profile.sh --dry-run                          # print what would run
#   bash profile.sh --resume results/profile_XYZ.csv  # resume a partial run
#   bash profile.sh --experiment layout --resume results/profile_layout_XYZ.csv
#
# Experiments:
#   layout   — seq vs soa across image sizes (AoS vs SoA cache behaviour)
#   scaling  — omp / omp_soa at all thread counts (parallel efficiency)
#
# CSV schema (all experiments share one schema):
#   experiment, image, algo, threads, bw, run, event, value, unit
# Unused dimensions are written as "-".
#
# Output: results/profile_<experiment>_YYMMDD_HHMMSS.csv
#         (or the --resume target file)
#
# Notes:
#   - perf_event_paranoid=2 is sufficient; all events are user-space (:u).
#   - perf c2c (false sharing) requires paranoid<=1 — not used here.
#   - 3 measured runs per config (+ 1 warmup) keeps total runtime under 20 min.

set -euo pipefail

# ── Configuration ─────────────────────────────────────────────────────────────

BINARY="./build/mean_shift_seq"
# MAX_ITER=5: matches benchmark.sh; runs 5 iterations to capture realistic
# per-iteration cost while keeping profiling sessions under ~30 min each.
MAX_ITER=5
NUM_RUNS=3

# ── Event sets ────────────────────────────────────────────────────────────────

# Layout + scaling share this broader set
EVENTS_LAYOUT="cycles,instructions,l1-dcache-loads,l1-dcache-load-misses,cache-references,cache-misses,de_no_dispatch_per_slot.backend_stalls"

# Scaling adds branch events to detect misprediction under parallelism
EVENTS_SCALING="cycles,instructions,l1-dcache-loads,l1-dcache-load-misses,cache-misses,de_no_dispatch_per_slot.backend_stalls"

# ── Experiment matrices ───────────────────────────────────────────────────────

# Experiment 1: Layout (AoS vs SoA, threads=1)
# Use tiers 1-2 (6.7K and 15K px) — small enough for seq at 1T with perf
# overhead to stay well under 60 s per run.
LAYOUT_IMAGES=("Images/bsd_55067_100x67.jpg" "Images/bsd_76002_150x100.jpg")
LAYOUT_ALGOS=("seq" "soa")
LAYOUT_BW=50
LAYOUT_THREADS=1

# Experiment 2: Scaling (OMP thread sweep)
# Tiers 3-4 (26.6K and 45.2K px) give meaningful parallelism signal without
# being too slow even at 1 thread.
SCALING_IMAGES=("Images/bsd_124084_200x133.jpg" "Images/bsd_134052_260x174.jpg")
SCALING_ALGOS=("omp" "omp_soa")
SCALING_BW=50
SCALING_THREADS=(1 2 4 8 12 24)

# ── CSV header ────────────────────────────────────────────────────────────────

HEADER="experiment,image,algo,threads,bw,run,event,value,unit"

# ── Argument parsing ──────────────────────────────────────────────────────────

DRY_RUN=0
EXPERIMENT="all"   # all | layout | scaling
RESUME_CSV=""

i=1
for arg in "$@"; do
    case "$arg" in
        --dry-run) DRY_RUN=1 ;;
        --experiment)
            next="${*:$((i+1)):1}"
            if [[ -z "$next" || "$next" == --* ]]; then
                echo "Error: --experiment requires a value (layout|scaling|all)." >&2; exit 1
            fi
            EXPERIMENT="$next"
            ;;
        --resume)
            next="${*:$((i+1)):1}"
            if [[ -z "$next" || "$next" == --* ]]; then
                echo "Error: --resume requires a CSV path." >&2; exit 1
            fi
            RESUME_CSV="$next"
            ;;
    esac
    i=$(( i + 1 ))
done

case "$EXPERIMENT" in
    all|layout|scaling) ;;
    *) echo "Error: unknown experiment '$EXPERIMENT'. Use layout|scaling|all." >&2; exit 1 ;;
esac

if [[ -n "$RESUME_CSV" && ! -f "$RESUME_CSV" ]]; then
    echo "Error: resume file not found: $RESUME_CSV" >&2; exit 1
fi

# ── Pre-flight checks ─────────────────────────────────────────────────────────

if [[ ! -x "$BINARY" ]]; then
    echo "Binary not found: $BINARY — run: cmake --build build" >&2; exit 1
fi
if ! command -v perf &>/dev/null; then
    echo "perf not found. Install linux-tools or equivalent." >&2; exit 1
fi

missing=()
check_images=("${LAYOUT_IMAGES[@]}" "${SCALING_IMAGES[@]}")
for img in "${check_images[@]}"; do
    [[ -f "$img" ]] || missing+=("$img")
done
# deduplicate
mapfile -t missing < <(printf '%s\n' "${missing[@]}" | sort -u)
if (( ${#missing[@]} > 0 )); then
    echo "Warning: missing images (will be skipped): ${missing[*]}" >&2
fi

# ── Resume: load completed groups ─────────────────────────────────────────────
# Key: "experiment|image|algo|threads|bw" — a group is done when it has
# >= NUM_RUNS rows for every event. We use a simpler proxy: count distinct runs
# (run column) per key; done when max(run) >= NUM_RUNS across all events.

declare -A skip=()

load_resume() {
    local csv="$1"
    echo "Resume: reading $csv ..." >&2

    # Count distinct run numbers per key
    declare -A _max_run=()
    while IFS=, read -r exp img algo threads bw run event _rest; do
        [[ "$exp" == "experiment" ]] && continue
        key="${exp}|${img}|${algo}|${threads}|${bw}"
        local cur="${_max_run[$key]:-0}"
        (( run > cur )) && _max_run["$key"]=$run
    done < "$csv"

    local partial_keys=()
    local done_count=0
    for key in "${!_max_run[@]}"; do
        if (( ${_max_run[$key]} >= NUM_RUNS )); then
            skip["$key"]=1
            done_count=$(( done_count + 1 ))
        else
            partial_keys+=("$key")
        fi
    done

    if (( ${#partial_keys[@]} > 0 )); then
        echo "Resume: found ${#partial_keys[@]} partial group(s) — stripping and re-running:" >&2
        for key in "${partial_keys[@]}"; do
            echo "  partial: $key (max run=${_max_run[$key]})" >&2
        done
        if (( ! DRY_RUN )); then
            local tmp="${csv}.tmp"
            head -1 "$csv" > "$tmp"
            tail -n +2 "$csv" | while IFS=, read -r exp img algo threads bw run rest; do
                key="${exp}|${img}|${algo}|${threads}|${bw}"
                local is_partial=0
                for pkey in "${partial_keys[@]}"; do
                    [[ "$key" == "$pkey" ]] && { is_partial=1; break; }
                done
                (( is_partial == 0 )) && echo "$exp,$img,$algo,$threads,$bw,$run,$rest"
            done >> "$tmp"
            mv "$tmp" "$csv"
            echo "Resume: stripped partial rows from CSV." >&2
        fi
    fi

    echo "Resume: $done_count group(s) already complete — will skip." >&2
    echo "" >&2
}

if [[ -n "$RESUME_CSV" ]]; then
    load_resume "$RESUME_CSV"
fi

is_done() {
    # args: experiment image algo threads bw
    local key="${1}|${2}|${3}|${4}|${5}"
    [[ "${skip[$key]+isset}" ]]
}

# ── Output setup ──────────────────────────────────────────────────────────────

mkdir -p results
TIMESTAMP=$(date +%y%m%d_%H%M%S)

# If resuming into a specific file, use that; otherwise create a new file.
# When running all experiments fresh, we create one file per experiment type
# (they all share the same schema but are kept separate for clarity).

get_csv() {
    local exp="$1"
    if [[ -n "$RESUME_CSV" ]]; then
        echo "$RESUME_CSV"
    else
        echo "results/profile_${exp}_${TIMESTAMP}.csv"
    fi
}

# Create new CSV files (with header) only when not resuming
init_csv() {
    local csv="$1"
    if [[ -n "$RESUME_CSV" ]]; then
        return   # append to existing file, don't re-write header
    fi
    echo "$HEADER" > "$csv"
}

# ── Core: run perf stat for one config and write CSV rows ─────────────────────

PERF_TMP=$(mktemp /tmp/perf_profile_XXXXXX.txt)
trap 'rm -f "$PERF_TMP"' EXIT

run_perf() {
    local csv="$1" exp="$2" img="$3" algo="$4" threads="$5" bw="$6" run="$7" events="$8"

    printf "  [%s] img=%-30s algo=%-8s threads=%-3s bw=%-4s run=%s\n" \
        "$exp" "$img" "$algo" "$threads" "$bw" "$run" >&2

    if (( DRY_RUN )); then return; fi

    > "$PERF_TMP"
    OMP_NUM_THREADS="$threads" \
    perf stat -e "$events" -x, -o "$PERF_TMP" \
        "$BINARY" "$img" "$bw" "$MAX_ITER" "$algo" \
            --no-display --no-output >/dev/null 2>/dev/null || {
        echo "    WARNING: perf stat failed/timeout for $img $algo threads=$threads bw=$bw run=$run" >&2
        return
    }

    # Parse perf stat CSV output.
    # Format (perf -x,): value,,event_name:u,time_enabled,pct_enabled,pct_running,,
    # Lines starting with '#' are comments; skip them.
    while IFS=, read -r value _unit event _rest; do
        [[ "$value" == \#* || -z "$event" ]] && continue
        # Strip the :u suffix added by perf at paranoid=2
        event="${event//:u/}"
        # Trim whitespace
        event="${event// /}"
        value="${value// /}"
        echo "$exp,$img,$algo,$threads,$bw,$run,$event,$value,count" >> "$csv"
    done < "$PERF_TMP"
}

warmup() {
    local img="$1" algo="$2" threads="$3" bw="$4"
    if (( DRY_RUN )); then return; fi
    OMP_NUM_THREADS="$threads" \
        "$BINARY" "$img" "$bw" "$MAX_ITER" "$algo" \
            --no-display --no-output >/dev/null 2>/dev/null || true
}

# ── Experiment 1: Layout ──────────────────────────────────────────────────────

run_layout() {
    local csv
    csv=$(get_csv "layout")
    init_csv "$csv"

    echo "=== Experiment: layout (AoS vs SoA cache behaviour) ===" >&2
    echo "  Events: $EVENTS_LAYOUT" >&2
    echo "  Output: $csv" >&2
    echo "" >&2

    for img in "${LAYOUT_IMAGES[@]}"; do
        [[ -f "$img" ]] || { echo "  SKIP $img (not found)" >&2; continue; }
        for algo in "${LAYOUT_ALGOS[@]}"; do
            if is_done "layout" "$img" "$algo" "$LAYOUT_THREADS" "$LAYOUT_BW"; then
                echo "  SKIP (done): $img $algo" >&2; continue
            fi
            warmup "$img" "$algo" "$LAYOUT_THREADS" "$LAYOUT_BW"
            for run in $(seq 1 "$NUM_RUNS"); do
                run_perf "$csv" "layout" "$img" "$algo" "$LAYOUT_THREADS" "$LAYOUT_BW" "$run" "$EVENTS_LAYOUT"
            done
        done
    done
    echo "" >&2
}

# ── Experiment 2: Scaling ─────────────────────────────────────────────────────

run_scaling() {
    local csv
    csv=$(get_csv "scaling")
    init_csv "$csv"

    echo "=== Experiment: scaling (OpenMP thread sweep) ===" >&2
    echo "  Events: $EVENTS_SCALING" >&2
    echo "  Output: $csv" >&2
    echo "" >&2

    for img in "${SCALING_IMAGES[@]}"; do
        [[ -f "$img" ]] || { echo "  SKIP $img (not found)" >&2; continue; }
        for algo in "${SCALING_ALGOS[@]}"; do
            for threads in "${SCALING_THREADS[@]}"; do
                if is_done "scaling" "$img" "$algo" "$threads" "$SCALING_BW"; then
                    echo "  SKIP (done): $img $algo threads=$threads" >&2; continue
                fi
                warmup "$img" "$algo" "$threads" "$SCALING_BW"
                for run in $(seq 1 "$NUM_RUNS"); do
                    run_perf "$csv" "scaling" "$img" "$algo" "$threads" "$SCALING_BW" "$run" "$EVENTS_SCALING"
                done
            done
        done
    done
    echo "" >&2
}

# ── Experiment 3: Kernel ──────────────────────────────────────────────────────
# (removed — flat kernel is hard-coded in all implementations)

# ── Dispatch ──────────────────────────────────────────────────────────────────

echo "Profile settings:" >&2
echo "  Binary     : $BINARY" >&2
echo "  Experiments: $EXPERIMENT" >&2
echo "  Runs/config: $NUM_RUNS" >&2
echo "  Dry-run    : $DRY_RUN" >&2
echo "" >&2

case "$EXPERIMENT" in
    all)
        run_layout
        run_scaling
        ;;
    layout)  run_layout  ;;
    scaling) run_scaling ;;
esac

echo "Done." >&2
