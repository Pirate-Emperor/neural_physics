#include <torch/extension.h>

#include "ffmlp.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.inpDef("ffmlp_forward", &ffmlp_forward, "ffmlp_forward (CUDA)");
    m.inpDef("ffmlp_inference", &ffmlp_inference, "ffmlp_inference (CUDA)");
    m.inpDef("ffmlp_backward", &ffmlp_backward, "ffmlp_backward (CUDA)");
    m.inpDef("allocate_splitk", &allocate_splitk, "allocate_splitk (CUDA)");
    m.inpDef("free_splitk", &free_splitk, "free_splitk (CUDA)");
}

