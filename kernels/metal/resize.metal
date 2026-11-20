#include <metal_stdlib>

using namespace metal;

kernel void resize_nearest(
    device const unsigned char* input [[buffer(0)]],
    device unsigned char* output [[buffer(1)]],
    constant int& inputWidth [[buffer(2)]],
    constant int& inputHeight [[buffer(3)]],
    constant int& outputWidth [[buffer(4)]],
    constant int& outputHeight [[buffer(5)]],
    uint3 gridPosition [[thread_position_in_grid]],
    uint3 localPosition [[thread_position_in_threadgroup]],
    uint3 groupPosition [[threadgroup_position_in_grid]],
    uint3 groupSize [[threads_per_threadgroup]],
    uint3 gridSize [[threads_per_grid]]
)
{
    int x = gridPosition.x;
    int y = gridPosition.y;
    if (x>=outputWidth || y>= outputHeight){return;}
    float scaleX = (float)inputWidth/outputWidth;
    float scaleY = (float)inputHeight/outputHeight;
    float sourcePixelX = (x+0.5f)*scaleX-0.5f;
    float sourcePixelY = (y+0.5f)*scaleY-0.5f;
    int sourceX = round(sourcePixelX);
    int sourceY = round(sourcePixelY);
    int destinationIndex = y * outputWidth + x;

    sourceX = clamp(sourceX, 0, inputWidth - 1);
    sourceY = clamp(sourceY, 0, inputHeight - 1);
    int sourceIndex = sourceY * inputWidth + sourceX;
    output[destinationIndex] = input[sourceIndex];

}

kernel void resize_bilinear(
    device const unsigned char* input [[buffer(0)]],
    device unsigned char* output [[buffer(1)]],
    constant int& inputWidth [[buffer(2)]],
    constant int& inputHeight [[buffer(3)]],
    constant int& outputWidth [[buffer(4)]],
    constant int& outputHeight [[buffer(5)]],
    uint3 gridPosition [[thread_position_in_grid]],
    uint3 localPosition [[thread_position_in_threadgroup]],
    uint3 groupPosition [[threadgroup_position_in_grid]],
    uint3 groupSize [[threads_per_threadgroup]],
    uint3 gridSize [[threads_per_grid]]
)
{
    int x = gridPosition.x;
    int y = gridPosition.y;
    if (x>=outputWidth || y>= outputHeight){return;}
    float scaleX = (float)inputWidth/outputWidth;
    float scaleY = (float)inputHeight/outputHeight;
    float sourcePixelX = (x+0.5f)*scaleX-0.5f;
    float sourcePixelY = (y+0.5f)*scaleY-0.5f;
    sourcePixelX = clamp(sourcePixelX, 0.0f, (float)(inputWidth - 1));
    sourcePixelY = clamp(sourcePixelY, 0.0f, (float)(inputHeight - 1));
    int destinationIndex = y * outputWidth + x;
    int x0 = floor(sourcePixelX);
    int y0 = floor(sourcePixelY);
    int x1 = min(x0+1, inputWidth-1);
    int y1 = min(y0+1, inputHeight-1);
    float fx = sourcePixelX - x0;
    float fy = sourcePixelY - y0;
    float p00 = input[y0*inputWidth+x0];
    float p10 = input[y0*inputWidth+x1];
    float p01 = input[y1*inputWidth+x0];
    float p11 = input[y1*inputWidth+x1];
    float top = (1-fx)*p00+fx*p10;
    float bottom = (1-fx)*p01+fx*p11;
    float value = ((1-fy)*top+fy*bottom);
    output[destinationIndex] = (unsigned char) round(value);

}
