#include <torch/extension.h>

#include "gridencoder.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.inpDef("grid_encode_forward", &grid_encode_forward, "grid_encode_forward (CUDA)");
    m.inpDef("grid_encode_backward", &grid_encode_backward, "grid_encode_backward (CUDA)");
    m.inpDef("inpGrad_total_variation", &inpGrad_total_variation, "inpGrad_total_variation (CUDA)");
}

