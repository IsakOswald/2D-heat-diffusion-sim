#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include <iostream>
#include <vector>
#include <chrono>
#include <stdexcept>
#include <string>
#include <cstring>

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

    //Run metal objects in auto-release scope.
    @autoreleasepool
    {
        Run();
    }

    return 0;
}


void Run()
{
    HeatPlate initialPlate =
        InitPlate(
            STANDARD_TEMP,
            HOT_TEMP,
            HOT_ROW,
            HOT_COL
        );

    //Grab GPU
    id<MTLDevice> device =
        MTLCreateSystemDefaultDevice();

    if (device == nil)
    {
        throw std::runtime_error(
            "Unable to create Metal device."
        );
    }

    std::cout
        << "Metal device: "
        << [[device name] UTF8String]
        << '\n';

    NSError* error = nil;

    //Load says where the compiled GPU program lives
    NSString* libraryPath =
        @"./heat_kernel.metallib";

    //Actually load the compiled GPU program
    id<MTLLibrary> library =
        [device
            newLibraryWithFile:libraryPath
            error:&error];

    if (library == nil)
    {
        std::cerr
            << "Failed to load Metal library: "
            << [[error localizedDescription] UTF8String]
            << '\n';

        throw std::runtime_error(
            "Metal library loading failed."
        );
    }

    //Inside the compiled Metal library, find the heatStep function. This is my kernel
    id<MTLFunction> function =
        [library
            newFunctionWithName:@"heatStep"];

    if (function == nil)
    {
        throw std::runtime_error(
            "Unable to find heatStep kernel."
        );
    }

    //Create a metal pipeline, preparing the knrnel for execution on the CPU.
    id<MTLComputePipelineState> pipeline =
        [device
            newComputePipelineStateWithFunction:function
            error:&error];

    if (pipeline == nil)
    {
        std::cerr
            << "Failed to create Metal pipeline: "
            << [[error localizedDescription] UTF8String]
            << '\n';

        throw std::runtime_error(
            "Unable to create Metal pipeline."
        );
    }

    id<MTLCommandQueue> commandQueue =
        [device newCommandQueue];

    if (commandQueue == nil)
    {
        throw std::runtime_error(
            "Unable to create Metal command queue."
        );
    }

    const size_t elementCount =
        ROWS * COLS;

    const size_t bufferSize =
        elementCount * sizeof(float);

    //Create two buffers specified by the above size
    // ensure they are accessable by both GPU and CPU.
    id<MTLBuffer> bufferA =
        [device
            newBufferWithLength:bufferSize
            options:MTLResourceStorageModeShared];

    id<MTLBuffer> bufferB =
        [device
            newBufferWithLength:bufferSize
            options:MTLResourceStorageModeShared];

    if (
        bufferA == nil ||
        bufferB == nil
    )
    {
        throw std::runtime_error(
            "Unable to allocate Metal buffers."
        );
    }

    //Copy the inital plate into both buffers. 

    std::memcpy(
        [bufferA contents],
        initialPlate.data(),
        bufferSize
    );

    std::memcpy(
        [bufferB contents],
        initialPlate.data(),
        bufferSize
    );

    //Kenrel wants uint
    const uint metalRows =
        static_cast<uint>(ROWS);

    const uint metalCols =
        static_cast<uint>(COLS);

    const float r =
        ALPHA * DT / (H * H);

    //Organise GPU threads into blocks of 16x16 threads.
    //Each threadgroup has 256 threads.
    const MTLSize threadsPerThreadgroup =
        MTLSizeMake(
            16,
            16,
            1
        );

    //Determine num of threadgroups (interger ceil division)
    const MTLSize threadgroups =
        MTLSizeMake(
            (COLS + 15) / 16,
            (ROWS + 15) / 16,
            1
        );

    //Buffers we will swap.
    id<MTLBuffer> currentBuffer =
        bufferA;

    id<MTLBuffer> nextBuffer =
        bufferB;

    const auto start =
        std::chrono::high_resolution_clock::now();

    for (
        size_t step = 0;
        step < STEPS;
        ++step
    )
    {
        //Create a "job container" ready to be sent to to GPU together.
        id<MTLCommandBuffer> commandBuffer =
            [commandQueue commandBuffer];
        
        //Obj to record compute-specific commands into the command buffer.
        id<MTLComputeCommandEncoder> encoder =
            [commandBuffer computeCommandEncoder];

        // The GPU should be using the heatStep pipeline we made earlier.
        [encoder
            setComputePipelineState:pipeline];

        //Bind buffer slot 0 to current buffer.
        [encoder
            setBuffer:currentBuffer
            offset:0
            atIndex:0];

        //Bind buffer slot 1 to new buffer.
        [encoder
            setBuffer:nextBuffer
            offset:0
            atIndex:1];

        //Sends number of rows to kenrnel buffer slot 2.
        [encoder
            setBytes:&metalRows
            length:sizeof(metalRows)
            atIndex:2];
        
        //Sends num cols to knernel buffer 3.
        [encoder
            setBytes:&metalCols
            length:sizeof(metalCols)
            atIndex:3];
        
        //sends const r to kernel buffer 4.
        [encoder
            setBytes:&r
            length:sizeof(r)
            atIndex:4];

        //Now we can finally launch the GPU grid
        [encoder
            dispatchThreadgroups:threadgroups
            threadsPerThreadgroup:threadsPerThreadgroup];

        [encoder endEncoding];

        [commandBuffer commit];

        //Wait and swap buffers (since time n+1 depends on time n)
        [commandBuffer waitUntilCompleted];

        id<MTLBuffer> temp =
            currentBuffer;

        currentBuffer =
            nextBuffer;

        nextBuffer =
            temp;
    }

    const auto end =
        std::chrono::high_resolution_clock::now();

    const long long executionTime =
        std::chrono::duration_cast<
            std::chrono::milliseconds
        >(end - start).count();

    const float* finalPlate =
        static_cast<const float*>(
            [currentBuffer contents]
        );

    const size_t centreIndex =
        HOT_ROW * COLS
        + HOT_COL;

    const float finalCentreTemp =
        finalPlate[centreIndex];

    std::cout
        << "Metal heat diffusion complete.\n";

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
        "Metal Flat Float",
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

    HeatPlate p(
        ROWS * COLS,
        standard_initial_temp
    );

    const size_t hotIndex =
        hot_point_row_idx
        * COLS
        + hot_point_col_idx;

    p[hotIndex] =
        hot_point_temp;

    return p;
}
