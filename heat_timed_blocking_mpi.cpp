#include <mpi.h>

#include <iostream>
#include <vector>
#include <stdexcept>
#include <string>

#include "benchmark_logger.hpp"

using HeatPlate = std::vector<float>;

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

void ExchangeHalosBlocking(
    HeatPlate& p,
    int rank,
    int size
);

void CalculateNextPlate(
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

    MPI_Barrier(MPI_COMM_WORLD);

    const double start =
        MPI_Wtime();

    for (size_t step = 0; step < STEPS; ++step)
    {
        ExchangeHalosBlocking(
            current,
            rank,
            size
        );

        CalculateNextPlate(
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
            << "Blocking MPI heat diffusion complete.\n";

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
            "MPI Blocking Flat Float",
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


void ExchangeHalosBlocking(
    HeatPlate& p,
    int rank,
    int size
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

    MPI_Sendrecv(
        firstRealRow,
        static_cast<int>(COLS),
        MPI_FLOAT,
        upperNeighbour,
        0,

        lowerGhostRow,
        static_cast<int>(COLS),
        MPI_FLOAT,
        lowerNeighbour,
        0,

        MPI_COMM_WORLD,
        MPI_STATUS_IGNORE
    );

    MPI_Sendrecv(
        lastRealRow,
        static_cast<int>(COLS),
        MPI_FLOAT,
        lowerNeighbour,
        1,

        upperGhostRow,
        static_cast<int>(COLS),
        MPI_FLOAT,
        upperNeighbour,
        1,

        MPI_COMM_WORLD,
        MPI_STATUS_IGNORE
    );
}


void CalculateNextPlate(
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

    for (
        size_t localRow = 1;
        localRow <= localRows;
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

        const size_t rowOffset =
            localRow * COLS;

        const size_t upperOffset =
            (localRow - 1) * COLS;

        const size_t lowerOffset =
            (localRow + 1) * COLS;

        for (
            size_t col = 1;
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
}
