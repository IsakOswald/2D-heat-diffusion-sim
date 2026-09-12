#!/usr/bin/env bash
set -euo pipefail

# Final overnight heat-diffusion benchmark
#
# Design:
#   - 3 increasing workloads
#   - 2 complete repeats
#   - 5-minute cooldown between EVERY benchmark run
#   - 15-minute cooldown between repeat 1 and repeat 2
#   - MPI ranks: 1, 3, 6
#   - optimized sequential baseline
#   - optimized sequential with auto-vectorization disabled
#   - explicit SIMD
#   - Metal
#   - blocking MPI
#   - non-blocking MPI
#   - MPI + SIMD
#
# Approximate idle time alone:
#   78 benchmark runs total
#   ~77 x 5 min cooldowns + 15 min repeat cooldown
#   => roughly 6.5-7 hours plus actual execution time
#
# Run from the CodeSubmission directory.

REPEATS=2
MPI_RANKS=(1 3 6)

WORKLOADS=(
  "3000 3000 1000"
  "5000 5000 1000"
  "6000 6000 1000"
)

COOLDOWN_BETWEEN_RUNS=300      # 5 minutes
COOLDOWN_BETWEEN_REPEATS=900   # 15 minutes

RESULTS_FILE="results.txt"
CONSOLE_LOG="benchmark_console.log"

GXX="/opt/homebrew/bin/g++-16"

if [[ ! -x "$GXX" ]]; then
    echo "[!] Expected GCC at $GXX but it was not found."
    echo "    Update GXX near the top of this script if your GCC path differs."
    exit 1
fi

for cmd in mpic++ mpirun xcrun clang++; do
    if ! command -v "$cmd" >/dev/null 2>&1; then
        echo "[!] Required command '$cmd' was not found in PATH."
        exit 1
    fi
done

echo "========================================"
echo " Final overnight benchmark"
echo "========================================"
echo "Repeats: $REPEATS"
echo "Cooldown between runs: ${COOLDOWN_BETWEEN_RUNS}s"
echo "Cooldown between repeats: ${COOLDOWN_BETWEEN_REPEATS}s"
echo "MPI ranks: ${MPI_RANKS[*]}"
echo "Workloads:"
for workload in "${WORKLOADS[@]}"; do
    echo "  $workload"
done
echo

# Preserve previous measurements.
if [[ -f "$RESULTS_FILE" && -s "$RESULTS_FILE" ]]; then
    stamp="$(date +%Y%m%d_%H%M%S)"
    mv "$RESULTS_FILE" "results_${stamp}.txt"
    echo "[i] Existing results.txt moved to results_${stamp}.txt"
fi

: > "$RESULTS_FILE"
: > "$CONSOLE_LOG"

# Make a temporary sequential source only for the no-vectorization executable,
# changing its logger label so results.txt can distinguish the two baselines.
NOVEC_SOURCE=".heat_timed_seq_novec_tmp.cpp"
cp heat_timed_seq.cpp "$NOVEC_SOURCE"

python3 - "$NOVEC_SOURCE" <<'PY'
from pathlib import Path
import sys

p = Path(sys.argv[1])
text = p.read_text()

old = '"Sequential Flat Float"'
new = '"Sequential No Auto-Vectorization"'

if old not in text:
    raise SystemExit(
        "[!] Could not find the sequential logger label in the temporary source."
    )

p.write_text(text.replace(old, new, 1))
PY

cleanup() {
    rm -f "$NOVEC_SOURCE"
}
trap cleanup EXIT

echo "[1/7] Compiling optimized sequential..."
"$GXX" -std=c++20 -O3 \
    heat_timed_seq.cpp \
    -o heat_timed_seq

echo "[2/7] Compiling sequential with compiler auto-vectorization disabled..."
"$GXX" -std=c++20 -O3 -fno-tree-vectorize \
    "$NOVEC_SOURCE" \
    -o heat_timed_seq_novec

echo "[3/7] Compiling explicit SIMD..."
"$GXX" -std=c++20 -O3 \
    heat_timed_simd.cpp \
    -o heat_timed_simd

read -r -a MPI_COMPILE_FLAGS <<< "$(mpic++ --showme:compile)"
read -r -a MPI_LINK_FLAGS <<< "$(mpic++ --showme:link)"

echo "[4/7] Compiling blocking MPI..."
"$GXX" -std=c++20 -O3 \
    "${MPI_COMPILE_FLAGS[@]}" \
    heat_timed_blocking_mpi.cpp \
    "${MPI_LINK_FLAGS[@]}" \
    -o heat_timed_blocking_mpi

echo "[5/7] Compiling non-blocking MPI..."
"$GXX" -std=c++20 -O3 \
    "${MPI_COMPILE_FLAGS[@]}" \
    heat_timed_nonblocking_mpi.cpp \
    "${MPI_LINK_FLAGS[@]}" \
    -o heat_timed_nonblocking_mpi

echo "[6/7] Compiling MPI + SIMD..."
"$GXX" -std=c++20 -O3 \
    "${MPI_COMPILE_FLAGS[@]}" \
    heat_timed_mpi_simd.cpp \
    "${MPI_LINK_FLAGS[@]}" \
    -o heat_timed_mpi_simd

echo "[7/7] Compiling Metal..."
xcrun -sdk macosx metal \
    -c heat_kernel.metal \
    -o heat_kernel.air

xcrun -sdk macosx metallib \
    heat_kernel.air \
    -o heat_kernel.metallib

clang++ -std=c++20 -O3 \
    heat_timed_metal.mm \
    -framework Foundation \
    -framework Metal \
    -o heat_timed_metal

echo
echo "[i] Compilation complete."
echo

RUN_COUNTER=0
TOTAL_RUNS=$(( REPEATS * ${#WORKLOADS[@]} * (4 + 3 * ${#MPI_RANKS[@]}) ))

cool_down() {
    local seconds="$1"
    local reason="$2"

    echo
    echo "[i] Cooling for ${seconds}s (${reason})..."
    sleep "$seconds"
}

run_test() {
    local label="$1"
    shift

    RUN_COUNTER=$((RUN_COUNTER + 1))

    echo
    echo "=================================================="
    echo " RUN $RUN_COUNTER / $TOTAL_RUNS"
    echo " $label"
    echo "=================================================="
    echo "Command: $*"

    {
        echo
        echo "=================================================="
        echo "RUN $RUN_COUNTER / $TOTAL_RUNS"
        echo "$label"
        echo "Command: $*"
        echo "Started: $(date)"
        "$@"
        status=$?
        echo "Finished: $(date)"
        echo "Exit code: $status"
        echo "=================================================="
        return $status
    } >> "$CONSOLE_LOG" 2>&1
}

run_workload_forward() {
    local rep="$1"
    local rows="$2"
    local cols="$3"
    local steps="$4"

    run_test "Repeat $rep | Sequential O3 | ${rows}x${cols} | ${steps} steps" \
        ./heat_timed_seq "$rows" "$cols" "$steps"
    cool_down "$COOLDOWN_BETWEEN_RUNS" "between benchmark runs"

    run_test "Repeat $rep | Sequential O3 No Auto-Vectorization | ${rows}x${cols} | ${steps} steps" \
        ./heat_timed_seq_novec "$rows" "$cols" "$steps"
    cool_down "$COOLDOWN_BETWEEN_RUNS" "between benchmark runs"

    run_test "Repeat $rep | Explicit SIMD | ${rows}x${cols} | ${steps} steps" \
        ./heat_timed_simd "$rows" "$cols" "$steps"
    cool_down "$COOLDOWN_BETWEEN_RUNS" "between benchmark runs"

    run_test "Repeat $rep | Metal | ${rows}x${cols} | ${steps} steps" \
        ./heat_timed_metal "$rows" "$cols" "$steps"
    cool_down "$COOLDOWN_BETWEEN_RUNS" "between benchmark runs"

    for ranks in "${MPI_RANKS[@]}"; do
        run_test "Repeat $rep | MPI Blocking | ${ranks} ranks | ${rows}x${cols} | ${steps} steps" \
            mpirun -np "$ranks" ./heat_timed_blocking_mpi "$rows" "$cols" "$steps"
        cool_down "$COOLDOWN_BETWEEN_RUNS" "between benchmark runs"

        run_test "Repeat $rep | MPI Nonblocking | ${ranks} ranks | ${rows}x${cols} | ${steps} steps" \
            mpirun -np "$ranks" ./heat_timed_nonblocking_mpi "$rows" "$cols" "$steps"
        cool_down "$COOLDOWN_BETWEEN_RUNS" "between benchmark runs"

        run_test "Repeat $rep | MPI+SIMD | ${ranks} ranks | ${rows}x${cols} | ${steps} steps" \
            mpirun -np "$ranks" ./heat_timed_mpi_simd "$rows" "$cols" "$steps"

        # On the final run of repeat 1, skip the normal 5-minute pause;
        # the dedicated 15-minute repeat cooldown replaces it.
        if [[ "$rep" -eq 1 ]]; then
            if [[ "$RUN_COUNTER" -lt $((TOTAL_RUNS / 2)) ]]; then
                cool_down "$COOLDOWN_BETWEEN_RUNS" "between benchmark runs"
            fi
        elif [[ "$RUN_COUNTER" -lt "$TOTAL_RUNS" ]]; then
            cool_down "$COOLDOWN_BETWEEN_RUNS" "between benchmark runs"
        fi
    done
}

run_workload_reverse() {
    local rep="$1"
    local rows="$2"
    local cols="$3"
    local steps="$4"

    # Reverse broad order in repeat 2 to reduce systematic ordering bias.
    for ((idx=${#MPI_RANKS[@]}-1; idx>=0; --idx)); do
        local ranks="${MPI_RANKS[$idx]}"

        run_test "Repeat $rep | MPI+SIMD | ${ranks} ranks | ${rows}x${cols} | ${steps} steps" \
            mpirun -np "$ranks" ./heat_timed_mpi_simd "$rows" "$cols" "$steps"
        cool_down "$COOLDOWN_BETWEEN_RUNS" "between benchmark runs"

        run_test "Repeat $rep | MPI Nonblocking | ${ranks} ranks | ${rows}x${cols} | ${steps} steps" \
            mpirun -np "$ranks" ./heat_timed_nonblocking_mpi "$rows" "$cols" "$steps"
        cool_down "$COOLDOWN_BETWEEN_RUNS" "between benchmark runs"

        run_test "Repeat $rep | MPI Blocking | ${ranks} ranks | ${rows}x${cols} | ${steps} steps" \
            mpirun -np "$ranks" ./heat_timed_blocking_mpi "$rows" "$cols" "$steps"
        cool_down "$COOLDOWN_BETWEEN_RUNS" "between benchmark runs"
    done

    run_test "Repeat $rep | Metal | ${rows}x${cols} | ${steps} steps" \
        ./heat_timed_metal "$rows" "$cols" "$steps"
    cool_down "$COOLDOWN_BETWEEN_RUNS" "between benchmark runs"

    run_test "Repeat $rep | Explicit SIMD | ${rows}x${cols} | ${steps} steps" \
        ./heat_timed_simd "$rows" "$cols" "$steps"
    cool_down "$COOLDOWN_BETWEEN_RUNS" "between benchmark runs"

    run_test "Repeat $rep | Sequential O3 No Auto-Vectorization | ${rows}x${cols} | ${steps} steps" \
        ./heat_timed_seq_novec "$rows" "$cols" "$steps"
    cool_down "$COOLDOWN_BETWEEN_RUNS" "between benchmark runs"

    run_test "Repeat $rep | Sequential O3 | ${rows}x${cols} | ${steps} steps" \
        ./heat_timed_seq "$rows" "$cols" "$steps"

    if [[ "$RUN_COUNTER" -lt "$TOTAL_RUNS" ]]; then
        cool_down "$COOLDOWN_BETWEEN_RUNS" "between benchmark runs"
    fi
}

# Repeat 1: small -> large, normal implementation order.
echo
echo "##################################################"
echo " REPEAT 1 / 2"
echo "##################################################"

for workload in "${WORKLOADS[@]}"; do
    read -r rows cols steps <<< "$workload"
    run_workload_forward 1 "$rows" "$cols" "$steps"
done

cool_down "$COOLDOWN_BETWEEN_REPEATS" "between complete repeat 1 and repeat 2"

# Repeat 2: large -> small, reversed implementation order.
# This helps reduce bias from always giving the same tests the coolest machine.
echo
echo "##################################################"
echo " REPEAT 2 / 2"
echo "##################################################"

for ((w=${#WORKLOADS[@]}-1; w>=0; --w)); do
    read -r rows cols steps <<< "${WORKLOADS[$w]}"
    run_workload_reverse 2 "$rows" "$cols" "$steps"
done

echo
echo "========================================"
echo " All benchmark runs finished."
echo " Results:     $RESULTS_FILE"
echo " Console log: $CONSOLE_LOG"
echo "========================================"
