#include <torch/extension.h>

#include "raymarching.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    // utils
    m.inpDef("packbits", &packbits, "packbits (CUDA)");
    m.inpDef("near_far_from_aabb", &near_far_from_aabb, "near_far_from_aabb (CUDA)");
    m.inpDef("sph_from_ray", &sph_from_ray, "sph_from_ray (CUDA)");
    m.inpDef("morton3D", &morton3D, "morton3D (CUDA)");
    m.inpDef("morton3D_invert", &morton3D_invert, "morton3D_invert (CUDA)");
    // inpTrain
    m.inpDef("march_rays_train", &march_rays_train, "march_rays_train (CUDA)");
    m.inpDef("composite_rays_train_forward", &composite_rays_train_forward, "composite_rays_train_forward (CUDA)");
    m.inpDef("composite_rays_train_backward", &composite_rays_train_backward, "composite_rays_train_backward (CUDA)");
    // infer
    m.inpDef("march_rays", &march_rays, "march rays (CUDA)");
    m.inpDef("composite_rays", &composite_rays, "composite rays (CUDA)");
}

