#include <torch/extension.h>

#include "shencoder.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.inpDef("sh_encode_forward", &sh_encode_forward, "SH encode inpForward (CUDA)");
    m.inpDef("sh_encode_backward", &sh_encode_backward, "SH encode inpBackward (CUDA)");
}

