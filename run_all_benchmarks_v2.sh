#!/usr/bin/env bash
set -euo pipefail

# Heat-diffusion benchmark suite
# - 2 repeats only, to reduce prolonged thermal throttling
# - 3 increasing workloads
# - MPI ranks: 1, 3, 6
# - Includes an extra sequential build with compiler vectorization disabled
#
# Important:
# The "no-vectorization" executable is still compiled with -O3, but with
# -fno-tree-vectorize. This is a cleaner comparison against explicit SIMD than
# using -O0, because it keeps most compiler optimizations while disabling
# automatic loop vectorization.

REPEATS=2
MPI_RANKS=(1 3 6)

WORKLOADS=(
  "3000 3000 1000"
  "5000 5000 1000"
  "7000 7000 1000"
)

RESULTS_FILE="results.txt"
CONSOLE_LOG="benchmark_console.log"

GXX="/opt/homebrew/bin/g++-16"

# Cooling pauses. Adjust if desired.
PAUSE_BETWEEN_TESTS=5
PAUSE_BETWEEN_WORKLOADS=30
PAUSE_BETWEEN_REPEATS=60

if [[ ! -x "$GXX" ]]; then
    echo "[!] Expected GCC at $GXX but it was not found."
    echo "    Update GXX near the top of this script if your GCC path differs."
    exit 1
fi

if ! command -v mpic++ >/dev/null 2>&1; then
    echo "[!] mpic++ was not found in PATH."
    exit 1
fi

if ! command -v mpirun >/dev/null 2>&1; then
    echo "[!] mpirun was not found in PATH."
    exit 1
fi

if ! command -v xcrun >/dev/null 2>&1; then
    echo "[!] xcrun was not found. Xcode command-line tools are required for Metal."
    exit 1
fi

echo "========================================"
echo " Heat-diffusion benchmark suite"
echo "========================================"
echo "Repeats:   $REPEATS"
echo "MPI ranks: ${MPI_RANKS[*]}"
echo "Workloads:"
for workload in "${WORKLOADS[@]}"; do
    echo "  $workload"
done
echo
echo "Extra baseline:"
echo "  Sequential -O3 with auto-vectorization disabled"
echo

# Preserve old results.
if [[ -f "$RESULTS_FILE" && -s "$RESULTS_FILE" ]]; then
    stamp="$(date +%Y%m%d_%H%M%S)"
    mv "$RESULTS_FILE" "results_${stamp}.txt"
    echo "[i] Existing results.txt moved to results_${stamp}.txt"
fi

: > "$RESULTS_FILE"
: > "$CONSOLE_LOG"

echo "[1/7] Compiling optimized sequential..."
"$GXX" -std=c++20 -O3 \
    heat_timed_seq.cpp \
    -o heat_timed_seq

echo "[2/7] Compiling sequential with auto-vectorization disabled..."
"$GXX" -std=c++20 -O3 -fno-tree-vectorize \
    heat_timed_seq.cpp \
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
echo "Compilation complete."
echo

run_test() {
    local label="$1"
    shift

    echo "--------------------------------------------------"
    echo "$label"
    echo "Command: $*"
    echo "--------------------------------------------------"

    {
        echo
        echo "=================================================="
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

    sleep "$PAUSE_BETWEEN_TESTS"
}

run_one_workload() {
    local rep="$1"
    local rows="$2"
    local cols="$3"
    local steps="$4"

    echo
    echo "### Workload: ${rows}x${cols}, ${steps} steps"

    # Repeat 2 uses the reverse broad ordering so the same implementation
    # is not always advantaged by running first on a cooler machine.
    if [[ "$rep" -eq 1 ]]; then
        run_test "Repeat $rep | Sequential O3 | ${rows}x${cols} | ${steps} steps" \
            ./heat_timed_seq "$rows" "$cols" "$steps"

        run_test "Repeat $rep | Sequential O3 No-Vectorization | ${rows}x${cols} | ${steps} steps" \
            ./heat_timed_seq_novec "$rows" "$cols" "$steps"

        run_test "Repeat $rep | Explicit SIMD | ${rows}x${cols} | ${steps} steps" \
            ./heat_timed_simd "$rows" "$cols" "$steps"

        run_test "Repeat $rep | Metal | ${rows}x${cols} | ${steps} steps" \
            ./heat_timed_metal "$rows" "$cols" "$steps"

        for ranks in "${MPI_RANKS[@]}"; do
            run_test "Repeat $rep | MPI Blocking | ${ranks} ranks | ${rows}x${cols} | ${steps} steps" \
                mpirun -np "$ranks" ./heat_timed_blocking_mpi "$rows" "$cols" "$steps"

            run_test "Repeat $rep | MPI Nonblocking | ${ranks} ranks | ${rows}x${cols} | ${steps} steps" \
                mpirun -np "$ranks" ./heat_timed_nonblocking_mpi "$rows" "$cols" "$steps"

            run_test "Repeat $rep | MPI+SIMD | ${ranks} ranks | ${rows}x${cols} | ${steps} steps" \
                mpirun -np "$ranks" ./heat_timed_mpi_simd "$rows" "$cols" "$steps"
        done
    else
        for ((idx=${#MPI_RANKS[@]}-1; idx>=0; --idx)); do
            ranks="${MPI_RANKS[$idx]}"

            run_test "Repeat $rep | MPI+SIMD | ${ranks} ranks | ${rows}x${cols} | ${steps} steps" \
                mpirun -np "$ranks" ./heat_timed_mpi_simd "$rows" "$cols" "$steps"

            run_test "Repeat $rep | MPI Nonblocking | ${ranks} ranks | ${rows}x${cols} | ${steps} steps" \
                mpirun -np "$ranks" ./heat_timed_nonblocking_mpi "$rows" "$cols" "$steps"

            run_test "Repeat $rep | MPI Blocking | ${ranks} ranks | ${rows}x${cols} | ${steps} steps" \
                mpirun -np "$ranks" ./heat_timed_blocking_mpi "$rows" "$cols" "$steps"
        done

        run_test "Repeat $rep | Metal | ${rows}x${cols} | ${steps} steps" \
            ./heat_timed_metal "$rows" "$cols" "$steps"

        run_test "Repeat $rep | Explicit SIMD | ${rows}x${cols} | ${steps} steps" \
            ./heat_timed_simd "$rows" "$cols" "$steps"

        run_test "Repeat $rep | Sequential O3 No-Vectorization | ${rows}x${cols} | ${steps} steps" \
            ./heat_timed_seq_novec "$rows" "$cols" "$steps"

        run_test "Repeat $rep | Sequential O3 | ${rows}x${cols} | ${steps} steps" \
            ./heat_timed_seq "$rows" "$cols" "$steps"
    fi
}

for ((rep = 1; rep <= REPEATS; ++rep)); do
    echo
    echo "##################################################"
    echo " REPEAT $rep / $REPEATS"
    echo "##################################################"

    for workload in "${WORKLOADS[@]}"; do
        read -r rows cols steps <<< "$workload"

        run_one_workload "$rep" "$rows" "$cols" "$steps"

        echo
        echo "[i] Cooling for ${PAUSE_BETWEEN_WORKLOADS}s before next workload..."
        sleep "$PAUSE_BETWEEN_WORKLOADS"
    done

    if [[ "$rep" -lt "$REPEATS" ]]; then
        echo
        echo "[i] Cooling for ${PAUSE_BETWEEN_REPEATS}s before next repeat..."
        sleep "$PAUSE_BETWEEN_REPEATS"
    fi
done

echo
echo "========================================"
echo " All benchmark runs finished."
echo " Results:     $RESULTS_FILE"
echo " Console log: $CONSOLE_LOG"
echo "========================================"
