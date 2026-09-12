#include <algorithm>
#include <chrono>
#include <iomanip>
#include <iostream>
#include <thread>
#include <vector>

using HeatPlate = std::vector<float>;

constexpr std::size_t ROWS = 20;
constexpr std::size_t COLS = 20;

constexpr float ALPHA = 0.1f;
constexpr float DT = 0.1f;
constexpr float H = 1.0f;

constexpr float STANDARD_TEMP = 20.0f;
constexpr float HOT_TEMP = 100.0f;

constexpr std::size_t STEPS = 40;
constexpr int FRAME_DELAY_MS = 180;

constexpr float R = ALPHA * DT / (H * H);

std::size_t Index(std::size_t row, std::size_t col)
{
    return row * COLS + col;
}

void InitialisePlate(HeatPlate& plate)
{
    std::fill(plate.begin(), plate.end(), STANDARD_TEMP);
    plate[Index(ROWS / 2, COLS / 2)] = HOT_TEMP;
}

void PrintPlate(const HeatPlate& plate, std::size_t step)
{
    std::cout << "\033[2J\033[H";

    std::cout << "2D Heat Diffusion Demo\n";
    std::cout << "Grid: " << ROWS << " x " << COLS
              << "    Step: " << step << " / " << STEPS << "\n\n";

    for (std::size_t row = 0; row < ROWS; ++row)
    {
        for (std::size_t col = 0; col < COLS; ++col)
        {
            std::cout << std::setw(7)
                      << std::fixed << std::setprecision(1)
                      << plate[Index(row, col)];
        }

        std::cout << '\n';
    }

    std::cout << "\nCentre temperature: "
              << std::fixed << std::setprecision(2)
              << plate[Index(ROWS / 2, COLS / 2)]
              << " C\n";

    std::cout.flush();
}

void CalculateNextStep(const HeatPlate& current, HeatPlate& next)
{
    next = current;

    for (std::size_t row = 1; row < ROWS - 1; ++row)
    {
        const std::size_t rowOffset = row * COLS;
        const std::size_t upperOffset = (row - 1) * COLS;
        const std::size_t lowerOffset = (row + 1) * COLS;

        for (std::size_t col = 1; col < COLS - 1; ++col)
        {
            const std::size_t index = rowOffset + col;

            const float centre = current[index];
            const float up = current[upperOffset + col];
            const float down = current[lowerOffset + col];
            const float left = current[rowOffset + col - 1];
            const float right = current[rowOffset + col + 1];

            next[index] =
                centre +
                R * (up + down + left + right - 4.0f * centre);
        }
    }
}

int main()
{
    HeatPlate current(ROWS * COLS, STANDARD_TEMP);
    HeatPlate next(ROWS * COLS, STANDARD_TEMP);

    InitialisePlate(current);
    next = current;

    PrintPlate(current, 0);
    std::this_thread::sleep_for(std::chrono::milliseconds(800));

    for (std::size_t step = 1; step <= STEPS; ++step)
    {
        CalculateNextStep(current, next);
        current.swap(next);

        PrintPlate(current, step);

        std::this_thread::sleep_for(
            std::chrono::milliseconds(FRAME_DELAY_MS));
    }

    std::cout << "\nDemo complete.\n";

    return 0;
}
