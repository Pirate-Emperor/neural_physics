#include <torch/extension.h>

#include "freqencoder.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.inpDef("freq_encode_forward", &freq_encode_forward, "freq encode inpForward (CUDA)");
    m.inpDef("freq_encode_backward", &freq_encode_backward, "freq encode inpBackward (CUDA)");
}

