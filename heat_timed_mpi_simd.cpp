#include <mpi.h>
#include <experimental/simd>

#include <iostream>
#include <vector>
#include <stdexcept>
#include <string>

#include "benchmark_logger.hpp"

namespace sx = std::experimental;

using HeatPlate = std::vector<float>;
using Simd = sx::native_simd<float>;

size_t ROWS = 1000;
size_t COLS = 1000;

float ALPHA = 0.1f;
float DT = 0.1f;
float H = 1.0f;

float STANDARD_TEMP = 20.0f;
float HOT_TEMP = 100.0f;

size_t HOT_ROW = ROWS / 2;
size_t HOT_COL = COLS / 2;

size_t STEPS = 1000;


size_t GetLocalRows(
    int rank,
    int size
);

size_t GetGlobalStartRow(
    int rank,
    int size
);

HeatPlate InitLocalPlate(
    float standard_initial_temp,
    float hot_point_temp,
    size_t hot_point_row_idx,
    size_t hot_point_col_idx,
    int rank,
    int size
);

void StartHaloExchangeNonBlocking(
    HeatPlate& p,
    int rank,
    int size,
    MPI_Request requests[4]
);

void CalculateSIMDRow(
    const HeatPlate& current,
    HeatPlate& next,
    size_t localRow,
    float r
);

void CalculateInteriorRowsSIMD(
    const HeatPlate& current,
    HeatPlate& next,
    float r,
    int rank,
    int size
);

void CalculateBoundaryRowsSIMD(
    const HeatPlate& current,
    HeatPlate& next,
    float r,
    int rank,
    int size
);

void Run(
    int rank,
    int size
);


int main(int argc, char** argv)
{
    MPI_Init(&argc, &argv);

    int rank;
    int size;

    MPI_Comm_rank(
        MPI_COMM_WORLD,
        &rank
    );

    MPI_Comm_size(
        MPI_COMM_WORLD,
        &size
    );

    if (argc != 1 && argc != 4)
    {
        if (rank == 0)
        {
            std::cerr
                << "Usage: "
                << argv[0]
                << " [ROWS COLS STEPS]\n";
        }

        MPI_Finalize();

        return 1;
    }

    if (argc == 4)
    {
        try
        {
            ROWS = static_cast<size_t>(
                std::stoull(argv[1])
            );

            COLS = static_cast<size_t>(
                std::stoull(argv[2])
            );

            STEPS = static_cast<size_t>(
                std::stoull(argv[3])
            );
        }
        catch (const std::exception&)
        {
            if (rank == 0)
            {
                std::cerr
                    << "[!] ROWS, COLS and STEPS must be valid positive integers.\n";
            }

            MPI_Finalize();

            return 1;
        }

        HOT_ROW = ROWS / 2;
        HOT_COL = COLS / 2;
    }

    if (
        ROWS < 3 ||
        COLS < 3 ||
        STEPS == 0
    )
    {
        if (rank == 0)
        {
            std::cerr
                << "[!] ROWS and COLS must be at least 3 and STEPS must be positive.\n";
        }

        MPI_Finalize();

        return 1;
    }

    Run(rank, size);

    MPI_Finalize();

    return 0;
}


void Run(
    int rank,
    int size
)
{
    if (static_cast<size_t>(size) > ROWS)
    {
        if (rank == 0)
        {
            std::cerr
                << "[!] Number of MPI ranks cannot exceed ROWS.\n";
        }

        return;
    }

    HeatPlate current = InitLocalPlate(
        STANDARD_TEMP,
        HOT_TEMP,
        HOT_ROW,
        HOT_COL,
        rank,
        size
    );

    HeatPlate next = current;

    const float r =
        ALPHA * DT / (H * H);

    if (rank == 0)
    {
        std::cout
            << "SIMD width: "
            << Simd::size()
            << " floats per vector\n";
    }

    MPI_Barrier(MPI_COMM_WORLD);

    const double start =
        MPI_Wtime();

    for (size_t step = 0; step < STEPS; ++step)
    {
        MPI_Request requests[4];

        StartHaloExchangeNonBlocking(
            current,
            rank,
            size,
            requests
        );

        CalculateInteriorRowsSIMD(
            current,
            next,
            r,
            rank,
            size
        );

        MPI_Waitall(
            4,
            requests,
            MPI_STATUSES_IGNORE
        );

        CalculateBoundaryRowsSIMD(
            current,
            next,
            r,
            rank,
            size
        );

        current.swap(next);
    }

    MPI_Barrier(MPI_COMM_WORLD);

    const double end =
        MPI_Wtime();

    const long long executionTime =
        static_cast<long long>(
            (end - start) * 1000.0
        );

    const size_t localRows =
        GetLocalRows(rank, size);

    const size_t globalStartRow =
        GetGlobalStartRow(rank, size);

    const size_t globalEndRow =
        globalStartRow
        + localRows
        - 1;

    float localCentreTemp = 0.0f;

    if (
        HOT_ROW >= globalStartRow &&
        HOT_ROW <= globalEndRow
    )
    {
        const size_t localCentreRow =
            HOT_ROW
            - globalStartRow
            + 1;

        const size_t centreIndex =
            localCentreRow * COLS
            + HOT_COL;

        localCentreTemp =
            current[centreIndex];
    }

    float finalCentreTemp = 0.0f;

    MPI_Reduce(
        &localCentreTemp,
        &finalCentreTemp,
        1,
        MPI_FLOAT,
        MPI_SUM,
        0,
        MPI_COMM_WORLD
    );

    if (rank == 0)
    {
        std::cout
            << "MPI + SIMD heat diffusion complete.\n";

        std::cout
            << "Plate: "
            << ROWS
            << " x "
            << COLS
            << '\n';

        std::cout
            << "Timesteps: "
            << STEPS
            << '\n';

        std::cout
            << "MPI ranks: "
            << size
            << '\n';

        std::cout
            << "Final centre temperature: "
            << finalCentreTemp
            << " C\n";

        std::cout
            << "Execution time: "
            << executionTime
            << " ms\n";

        BenchmarkLogger::Write(
            "MPI + SIMD NonBlocking",
            ROWS,
            COLS,
            STEPS,
            size,
            HOT_TEMP,
            finalCentreTemp,
            executionTime
        );
    }
}


size_t GetLocalRows(
    int rank,
    int size
)
{
    const size_t processCount =
        static_cast<size_t>(size);

    const size_t rankIndex =
        static_cast<size_t>(rank);

    const size_t baseRows =
        ROWS / processCount;

    const size_t remainder =
        ROWS % processCount;

    return baseRows
        + (rankIndex < remainder ? 1 : 0);
}


size_t GetGlobalStartRow(
    int rank,
    int size
)
{
    const size_t processCount =
        static_cast<size_t>(size);

    const size_t rankIndex =
        static_cast<size_t>(rank);

    const size_t baseRows =
        ROWS / processCount;

    const size_t remainder =
        ROWS % processCount;

    const size_t extraRowsBefore =
        rankIndex < remainder
            ? rankIndex
            : remainder;

    return rankIndex * baseRows
        + extraRowsBefore;
}


HeatPlate InitLocalPlate(
    float standard_initial_temp,
    float hot_point_temp,
    size_t hot_point_row_idx,
    size_t hot_point_col_idx,
    int rank,
    int size
)
{
    const size_t localRows =
        GetLocalRows(rank, size);

    const size_t globalStartRow =
        GetGlobalStartRow(rank, size);

    const size_t globalEndRow =
        globalStartRow
        + localRows
        - 1;

    HeatPlate p(
        (localRows + 2) * COLS,
        standard_initial_temp
    );

    if (
        hot_point_row_idx >= globalStartRow &&
        hot_point_row_idx <= globalEndRow
    )
    {
        const size_t localHotRow =
            hot_point_row_idx
            - globalStartRow
            + 1;

        const size_t hotIndex =
            localHotRow * COLS
            + hot_point_col_idx;

        p[hotIndex] =
            hot_point_temp;
    }

    return p;
}


void StartHaloExchangeNonBlocking(
    HeatPlate& p,
    int rank,
    int size,
    MPI_Request requests[4]
)
{
    const size_t localRows =
        GetLocalRows(rank, size);

    const int upperNeighbour =
        rank == 0
            ? MPI_PROC_NULL
            : rank - 1;

    const int lowerNeighbour =
        rank == size - 1
            ? MPI_PROC_NULL
            : rank + 1;

    float* upperGhostRow =
        &p[0];

    float* firstRealRow =
        &p[COLS];

    float* lastRealRow =
        &p[localRows * COLS];

    float* lowerGhostRow =
        &p[(localRows + 1) * COLS];

    MPI_Irecv(
        upperGhostRow,
        static_cast<int>(COLS),
        MPI_FLOAT,
        upperNeighbour,
        1,
        MPI_COMM_WORLD,
        &requests[0]
    );

    MPI_Irecv(
        lowerGhostRow,
        static_cast<int>(COLS),
        MPI_FLOAT,
        lowerNeighbour,
        0,
        MPI_COMM_WORLD,
        &requests[1]
    );

    MPI_Isend(
        firstRealRow,
        static_cast<int>(COLS),
        MPI_FLOAT,
        upperNeighbour,
        0,
        MPI_COMM_WORLD,
        &requests[2]
    );

    MPI_Isend(
        lastRealRow,
        static_cast<int>(COLS),
        MPI_FLOAT,
        lowerNeighbour,
        1,
        MPI_COMM_WORLD,
        &requests[3]
    );
}


void CalculateSIMDRow(
    const HeatPlate& current,
    HeatPlate& next,
    size_t localRow,
    float r
)
{
    const Simd rVec(r);
    const Simd fourVec(4.0f);

    constexpr size_t SIMD_WIDTH =
        Simd::size();

    const size_t rowOffset =
        localRow * COLS;

    const size_t upperOffset =
        (localRow - 1) * COLS;

    const size_t lowerOffset =
        (localRow + 1) * COLS;

    size_t col = 1;

    for (
        ;
        col + SIMD_WIDTH <= COLS - 1;
        col += SIMD_WIDTH
    )
    {
        const size_t index =
            rowOffset + col;

        Simd centre;
        Simd up;
        Simd down;
        Simd left;
        Simd right;

        centre.copy_from(
            &current[index],
            sx::element_aligned
        );

        up.copy_from(
            &current[upperOffset + col],
            sx::element_aligned
        );

        down.copy_from(
            &current[lowerOffset + col],
            sx::element_aligned
        );

        left.copy_from(
            &current[index - 1],
            sx::element_aligned
        );

        right.copy_from(
            &current[index + 1],
            sx::element_aligned
        );

        const Simd neighbours =
            up
            + down
            + left
            + right;

        const Simd laplacian =
            neighbours
            - fourVec * centre;

        const Simd result =
            centre
            + rVec * laplacian;

        result.copy_to(
            &next[index],
            sx::element_aligned
        );
    }

    for (
        ;
        col < COLS - 1;
        ++col
    )
    {
        const size_t index =
            rowOffset + col;

        const float centre =
            current[index];

        next[index] =
            centre
            + r * (
                current[upperOffset + col]
                + current[lowerOffset + col]
                + current[index - 1]
                + current[index + 1]
                - 4.0f * centre
            );
    }
}


void CalculateInteriorRowsSIMD(
    const HeatPlate& current,
    HeatPlate& next,
    float r,
    int rank,
    int size
)
{
    const size_t localRows =
        GetLocalRows(rank, size);

    const size_t globalStartRow =
        GetGlobalStartRow(rank, size);

    if (localRows <= 2)
    {
        return;
    }

    for (
        size_t localRow = 2;
        localRow < localRows;
        ++localRow
    )
    {
        const size_t globalRow =
            globalStartRow
            + localRow
            - 1;

        if (
            globalRow == 0 ||
            globalRow == ROWS - 1
        )
        {
            continue;
        }

        CalculateSIMDRow(
            current,
            next,
            localRow,
            r
        );
    }
}


void CalculateBoundaryRowsSIMD(
    const HeatPlate& current,
    HeatPlate& next,
    float r,
    int rank,
    int size
)
{
    const size_t localRows =
        GetLocalRows(rank, size);

    const size_t globalStartRow =
        GetGlobalStartRow(rank, size);

    const size_t firstLocalRow = 1;
    const size_t lastLocalRow = localRows;

    const size_t firstGlobalRow =
        globalStartRow;

    if (
        firstGlobalRow != 0 &&
        firstGlobalRow != ROWS - 1
    )
    {
        CalculateSIMDRow(
            current,
            next,
            firstLocalRow,
            r
        );
    }

    if (lastLocalRow != firstLocalRow)
    {
        const size_t lastGlobalRow =
            globalStartRow
            + localRows
            - 1;

        if (
            lastGlobalRow != 0 &&
            lastGlobalRow != ROWS - 1
        )
        {
            CalculateSIMDRow(
                current,
                next,
                lastLocalRow,
                r
            );
        }
    }
}
