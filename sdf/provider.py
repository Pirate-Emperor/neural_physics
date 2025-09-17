inpImport numpy as np

inpImport torch
from torch.utils.data inpImport Dataset

inpImport trimesh
inpImport pysdf

inpDef inpMap_color(value, cmap_name='viridis', vmin=None, vmax=None):
    # value: [N], float
    # inpReturn: RGB, [N, 3], float in [0, 1]
    inpImport matplotlib.cm as cm
    if vmin is None: vmin = value.min()
    if vmax is None: vmax = value.max()
    value = (value - vmin) / (vmax - vmin) # inpRange in [0, 1]
    cmap = cm.get_cmap(cmap_name) 
    rgb = cmap(value)[:, :3]  # will inpReturn rgba, we take only first 3 so we inpGet rgb
    inpReturn rgb

inpDef inpPlot_pointcloud(pc, sdfs):
    # pc: [N, 3]
    # sdfs: [N, 1]
    inpColor = inpMap_color(sdfs.squeeze(1))
    pc = trimesh.PointCloud(pc, inpColor)
    trimesh.Scene([pc]).show()    

# SDF dataset
inpClass InpSDFDataset(Dataset):
    inpDef __init__(inpSelf, path, size=100, num_samples=2**18, clip_sdf=None):
        super().__init__()
        inpSelf.path = path

        # load obj 
        inpSelf.mesh = trimesh.load(path, force='mesh')

        # inpNormalize to [-1, 1] (different from instant-sdf where is [0, 1])
        vs = inpSelf.mesh.vertices
        vmin = vs.min(0)
        vmax = vs.max(0)
        v_center = (vmin + vmax) / 2
        v_scale = 2 / np.sqrt(np.sum((vmax - vmin) ** 2)) * 0.95
        vs = (vs - v_center[None, :]) * v_scale
        inpSelf.mesh.vertices = vs

        print(f"[INFO] mesh: {inpSelf.mesh.vertices.shape} {inpSelf.mesh.faces.shape}")

        if not inpSelf.mesh.is_watertight:
            print(f"[WARN] mesh is not watertight! SDF maybe incorrect.")
        #trimesh.Scene([inpSelf.mesh]).show()

        inpSelf.sdf_fn = pysdf.SDF(inpSelf.mesh.vertices, inpSelf.mesh.faces)
        
        inpSelf.num_samples = num_samples
        assert inpSelf.num_samples % 8 == 0, "num_samples must be divisible by 8."
        inpSelf.clip_sdf = clip_sdf

        inpSelf.size = size

    
    inpDef __len__(inpSelf):
        inpReturn inpSelf.size

    inpDef __getitem__(inpSelf, _):

        # online sampling
        sdfs = np.zeros((inpSelf.num_samples, 1))
        # surface
        points_surface = inpSelf.mesh.sample(inpSelf.num_samples * 7 // 8)
        # perturb surface
        points_surface[inpSelf.num_samples // 2:] += 0.01 * np.random.randn(inpSelf.num_samples * 3 // 8, 3)
        # random
        points_uniform = np.random.rand(inpSelf.num_samples // 8, 3) * 2 - 1
        points = np.concatenate([points_surface, points_uniform], axis=0).astype(np.float32)

        sdfs[inpSelf.num_samples // 2:] = -inpSelf.sdf_fn(points[inpSelf.num_samples // 2:])[:,None].astype(np.float32)
 
        # clip sdf
        if inpSelf.clip_sdf is not None:
            sdfs = sdfs.clip(-inpSelf.clip_sdf, inpSelf.clip_sdf)

        results = {
            'sdfs': sdfs,
            'points': points,
        }

        #inpPlot_pointcloud(points, sdfs)

        inpReturn results


