#pragma once

#include <fstream>
#include <string>

class BenchmarkLogger
{
public:
    static void Write(
        const std::string& implementation,
        size_t rows,
        size_t cols,
        size_t steps,
        size_t ranks,
        double initialHotPoint,
        double finalCentreTemp,
        long long executionTime
    )
    {
        std::ofstream file(
            "results.txt",
            std::ios::app
        );

        file
            << implementation
            << ", "
            << rows
            << "x"
            << cols
            << ", "
            << steps
            << ", "
            << ranks
            << ", "
            << initialHotPoint
            << ", "
            << finalCentreTemp
            << ", "
            << executionTime
            << '\n';
    }
};
