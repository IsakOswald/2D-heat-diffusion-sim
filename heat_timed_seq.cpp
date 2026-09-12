#include <iostream>
#include <vector>
#include <stdexcept>
#include <string>
#include <chrono>

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


HeatPlate InitPlate(
    float standard_initial_temp,
    float hot_point_temp,
    size_t hot_point_row_idx,
    size_t hot_point_col_idx
);

void CalculateNextPlate(
    const HeatPlate& current,
    HeatPlate& next,
    float r
);

void Run();

int main(int argc, char** argv)
{
    if (argc != 1 && argc != 4)
    {
        std::cerr
            << "Usage: "
            << argv[0]
            << " [ROWS COLS STEPS]\n";

        return 1;
    }

    if (argc == 4)
    {
        try
        {
            ROWS = static_cast<size_t>(
                std::stoull(argv[1]) //u long long cast.
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
            std::cerr
                << "[!] ROWS, COLS and STEPS must be valid positive integers.\n";

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
        std::cerr
            << "[!] ROWS and COLS must be at least 3 and STEPS must be positive.\n";

        return 1;
    }

    Run();

    return 0;
}


void Run()
{
    HeatPlate current = InitPlate(
        STANDARD_TEMP,
        HOT_TEMP,
        HOT_ROW,
        HOT_COL
    );

    HeatPlate next = current;

    const float r =
        ALPHA * DT / (H * H);

    const auto start =
        std::chrono::high_resolution_clock::now();

    for (size_t step = 0; step < STEPS; ++step)
    {
        CalculateNextPlate(
            current,
            next,
            r
        );

        current.swap(next);
    }

    const auto end =
        std::chrono::high_resolution_clock::now();

    const long long executionTime =
        std::chrono::duration_cast<
            std::chrono::milliseconds
        >(end - start).count();

    const size_t centreIndex =
        HOT_ROW * COLS + HOT_COL;

    const float finalCentreTemp =
        current[centreIndex];

    std::cout
        << "Sequential heat diffusion complete.\n";

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
        << "Final centre temperature: "
        << finalCentreTemp
        << " C\n";

    std::cout
        << "Execution time: "
        << executionTime
        << " ms\n";

    BenchmarkLogger::Write(
        "Sequential Flat Float",
        ROWS,
        COLS,
        STEPS,
        1,
        HOT_TEMP,
        finalCentreTemp,
        executionTime
    );
}


HeatPlate InitPlate(
    float standard_initial_temp,
    float hot_point_temp,
    size_t hot_point_row_idx,
    size_t hot_point_col_idx
)
{
    if (
        hot_point_row_idx >= ROWS ||
        hot_point_col_idx >= COLS
    )
    {
        throw std::runtime_error(
            "[!] Invalid hot point index."
        );
    }

    //Allocates ROWS * COLS floats and fills them with the starting temp.
    HeatPlate p(
        ROWS * COLS,
        standard_initial_temp
    );

    //This is important. This is how we traverse the flattened 2D plate. Generally:
    // row * COLS + col.
    const size_t hotIndex =
        hot_point_row_idx * COLS
        + hot_point_col_idx;

    p[hotIndex] =
        hot_point_temp;

    return p;
}


void CalculateNextPlate(
    const HeatPlate& current,
    HeatPlate& next,
    float r
)
{
    //Ecxclude the top and bottom row since they cells above/below them.
    for (
        size_t row = 1;
        row < ROWS - 1;
        ++row
    )
    {
        const size_t rowOffset =
            row * COLS;

        const size_t upperOffset =
            (row - 1) * COLS;

        const size_t lowerOffset =
            (row + 1) * COLS;
        
        //Exclude the leftmost and rightmost cells since they dont have neighbours to their left/right.
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
