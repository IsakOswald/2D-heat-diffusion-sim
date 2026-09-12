#!/usr/bin/env bash
set -euo pipefail

REPEATS=5
MPI_RANKS=(1 3 6)

WORKLOADS=(
  "3000 3000 1000"
  "5000 5000 1000"
  "7000 7000 1000"
)

RESULTS_FILE="results.txt"
CONSOLE_LOG="benchmark_console.log"
GXX="/opt/homebrew/bin/g++-16"

if [[ ! -x "$GXX" ]]; then
    echo "[!] Expected GCC at $GXX but it was not found."
    echo "    Update GXX near the top of this script if your path differs."
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

if [[ -f "$RESULTS_FILE" && -s "$RESULTS_FILE" ]]; then
    stamp="$(date +%Y%m%d_%H%M%S)"
    mv "$RESULTS_FILE" "results_${stamp}.txt"
    echo "[i] Existing results.txt moved to results_${stamp}.txt"
fi

: > "$RESULTS_FILE"
: > "$CONSOLE_LOG"

echo "[1/6] Compiling sequential..."
"$GXX" -std=c++20 -O3 heat_timed_seq.cpp -o heat_timed_seq

echo "[2/6] Compiling explicit SIMD..."
"$GXX" -std=c++20 -O3 heat_timed_simd.cpp -o heat_timed_simd

read -r -a MPI_COMPILE_FLAGS <<< "$(mpic++ --showme:compile)"
read -r -a MPI_LINK_FLAGS <<< "$(mpic++ --showme:link)"

echo "[3/6] Compiling blocking MPI..."
"$GXX" -std=c++20 -O3 "${MPI_COMPILE_FLAGS[@]}" heat_timed_blocking_mpi.cpp "${MPI_LINK_FLAGS[@]}" -o heat_timed_blocking_mpi

echo "[4/6] Compiling non-blocking MPI..."
"$GXX" -std=c++20 -O3 "${MPI_COMPILE_FLAGS[@]}" heat_timed_nonblocking_mpi.cpp "${MPI_LINK_FLAGS[@]}" -o heat_timed_nonblocking_mpi

echo "[5/6] Compiling MPI + SIMD..."
"$GXX" -std=c++20 -O3 "${MPI_COMPILE_FLAGS[@]}" heat_timed_mpi_simd.cpp "${MPI_LINK_FLAGS[@]}" -o heat_timed_mpi_simd

echo "[6/6] Compiling Metal kernel + host..."
xcrun -sdk macosx metal -c heat_kernel.metal -o heat_kernel.air
xcrun -sdk macosx metallib heat_kernel.air -o heat_kernel.metallib
clang++ -std=c++20 -O3 heat_timed_metal.mm -framework Foundation -framework Metal -o heat_timed_metal

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

    sleep 1
}

for ((rep = 1; rep <= REPEATS; ++rep)); do
    echo
    echo "##################################################"
    echo " REPEAT $rep / $REPEATS"
    echo "##################################################"

    for workload in "${WORKLOADS[@]}"; do
        read -r rows cols steps <<< "$workload"

        echo
        echo "### Workload: ${rows}x${cols}, ${steps} steps"

        run_test "Repeat $rep | Sequential | ${rows}x${cols} | ${steps} steps" ./heat_timed_seq "$rows" "$cols" "$steps"
        run_test "Repeat $rep | SIMD | ${rows}x${cols} | ${steps} steps" ./heat_timed_simd "$rows" "$cols" "$steps"
        run_test "Repeat $rep | Metal | ${rows}x${cols} | ${steps} steps" ./heat_timed_metal "$rows" "$cols" "$steps"

        for ranks in "${MPI_RANKS[@]}"; do
            run_test "Repeat $rep | MPI Blocking | ${ranks} ranks | ${rows}x${cols} | ${steps} steps" mpirun -np "$ranks" ./heat_timed_blocking_mpi "$rows" "$cols" "$steps"
            run_test "Repeat $rep | MPI Nonblocking | ${ranks} ranks | ${rows}x${cols} | ${steps} steps" mpirun -np "$ranks" ./heat_timed_nonblocking_mpi "$rows" "$cols" "$steps"
            run_test "Repeat $rep | MPI+SIMD | ${ranks} ranks | ${rows}x${cols} | ${steps} steps" mpirun -np "$ranks" ./heat_timed_mpi_simd "$rows" "$cols" "$steps"
        done
    done
done

echo
echo "========================================"
echo " All benchmark runs finished."
echo " Results:     $RESULTS_FILE"
echo " Console log: $CONSOLE_LOG"
echo "========================================"
