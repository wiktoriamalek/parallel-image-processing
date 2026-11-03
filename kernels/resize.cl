__kernel void resize_nearest(__global const unsigned char* input, __global unsigned char* output, int inputWidth, int inputHeight, int outputWidth, int outputHeight)
{
    int x = get_global_id(0);
    int y = get_global_id(1);
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

__kernel void resize_bilinear(__global const unsigned char* input, __global unsigned char* output, int inputWidth, int inputHeight, int outputWidth, int outputHeight)
{
    int x = get_global_id(0);
    int y = get_global_id(1);
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