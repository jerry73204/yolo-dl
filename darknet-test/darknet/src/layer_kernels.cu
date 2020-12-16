#include "layer.h"
#include "dark_cuda.h"

float *layer_get_output_gpu(const layer *layer)
{
    if(layer->type != REGION)
        cuda_pull_array(layer->output_gpu, layer->output, layer->outputs * layer->batch);
    return layer->output;
}
