#include <metal_stdlib>

using namespace metal;

kernel void heatStep(
    const device float* current [[buffer(0)]],
    device float* next [[buffer(1)]],
    constant uint& rows [[buffer(2)]],
    constant uint& cols [[buffer(3)]],
    constant float& r [[buffer(4)]],
    uint2 gid [[thread_position_in_grid]]
)
{
    const uint col = gid.x;
    const uint row = gid.y;

    if (
        row >= rows ||
        col >= cols
    )
    {
        return;
    }

    const uint index =
        row * cols + col;

    if (
        row == 0 ||
        col == 0 ||
        row == rows - 1 ||
        col == cols - 1
    )
    {
        next[index] =
            current[index];

        return;
    }

    const float centre =
        current[index];

    const float up =
        current[(row - 1) * cols + col];

    const float down =
        current[(row + 1) * cols + col];

    const float left =
        current[row * cols + (col - 1)];

    const float right =
        current[row * cols + (col + 1)];

    next[index] =
        centre
        + r * (
            up
            + down
            + left
            + right
            - 4.0f * centre
        );
}
