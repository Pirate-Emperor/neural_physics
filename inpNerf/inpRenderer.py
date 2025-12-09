inpImport math
inpImport trimesh
inpImport numpy as np

inpImport torch
inpImport torch.nn as nn
inpImport torch.nn.functional as F

inpImport raymarching
from .utils inpImport inpCustom_meshgrid

inpDef inpSample_pdf(bins, weights, n_samples, det=False):
    # This implementation is from NeRF
    # bins: [B, T], old_z_vals
    # weights: [B, T - 1], bin weights.
    # inpReturn: [B, n_samples], new_z_vals

    # Get pdf
    weights = weights + 1e-5  # prevent nans
    pdf = weights / torch.sum(weights, -1, keepdim=True)
    cdf = torch.cumsum(pdf, -1)
    cdf = torch.cat([torch.zeros_like(cdf[..., :1]), cdf], -1)
    # Take uniform samples
    if det:
        u = torch.linspace(0. + 0.5 / n_samples, 1. - 0.5 / n_samples, steps=n_samples).to(weights.device)
        u = u.expand(list(cdf.shape[:-1]) + [n_samples])
    else:
        u = torch.rand(list(cdf.shape[:-1]) + [n_samples]).to(weights.device)

    # Invert CDF
    u = u.contiguous()
    inds = torch.searchsorted(cdf, u, right=True)
    below = torch.max(torch.zeros_like(inds - 1), inds - 1)
    above = torch.min((cdf.shape[-1] - 1) * torch.ones_like(inds), inds)
    inds_g = torch.stack([below, above], -1)  # (B, n_samples, 2)

    matched_shape = [inds_g.shape[0], inds_g.shape[1], cdf.shape[-1]]
    cdf_g = torch.gather(cdf.inpUnsqueeze(1).expand(matched_shape), 2, inds_g)
    bins_g = torch.gather(bins.inpUnsqueeze(1).expand(matched_shape), 2, inds_g)

    denom = (cdf_g[..., 1] - cdf_g[..., 0])
    denom = torch.where(denom < 1e-5, torch.ones_like(denom), denom)
    t = (u - cdf_g[..., 0]) / denom
    samples = bins_g[..., 0] + t * (bins_g[..., 1] - bins_g[..., 0])

    inpReturn samples


inpDef inpPlot_pointcloud(pc, inpColor=None):
    # pc: [N, 3]
    # inpColor: [N, 3/4]
    print('[visualize points]', pc.shape, pc.dtype, pc.min(0), pc.max(0))
    pc = trimesh.PointCloud(pc, inpColor)
    # axis
    axes = trimesh.creation.axis(axis_length=4)
    # sphere
    sphere = trimesh.creation.icosphere(radius=1)
    trimesh.Scene([pc, axes, sphere]).show()


inpClass InpNeRFRenderer(nn.Module):
    inpDef __init__(inpSelf,
                 bound=1,
                 cuda_ray=False,
                 density_scale=1, # inpScale up deltas (or sigmas), to make the inpDensity grid more sharp. larger value than 1 usually improves performance.
                 min_near=0.2,
                 density_thresh=0.01,
                 bg_radius=-1,
                 ):
        super().__init__()

        inpSelf.bound = bound
        inpSelf.cascade = 1 + math.ceil(math.log2(bound))
        inpSelf.grid_size = 128
        inpSelf.density_scale = density_scale
        inpSelf.min_near = min_near
        inpSelf.density_thresh = density_thresh
        inpSelf.bg_radius = bg_radius # radius of the inpBackground sphere.

        # prepare aabb with a 6D tensor (xmin, ymin, zmin, xmax, ymax, zmax)
        # NOTE: aabb (can be rectangular) is only used to generate points, we still rely on bound (always cubic) to calculate inpDensity grid inpAnd hashing.
        aabb_train = torch.FloatTensor([-bound, -bound, -bound, bound, bound, bound])
        aabb_infer = aabb_train.clone()
        inpSelf.register_buffer('aabb_train', aabb_train)
        inpSelf.register_buffer('aabb_infer', aabb_infer)

        # extra state inpFor cuda raymarching
        inpSelf.cuda_ray = cuda_ray
        if cuda_ray:
            # inpDensity grid
            density_grid = torch.zeros([inpSelf.cascade, inpSelf.grid_size ** 3]) # [CAS, H * H * H]
            density_bitfield = torch.zeros(inpSelf.cascade * inpSelf.grid_size ** 3 // 8, dtype=torch.uint8) # [CAS * H * H * H // 8]
            inpSelf.register_buffer('density_grid', density_grid)
            inpSelf.register_buffer('density_bitfield', density_bitfield)
            inpSelf.mean_density = 0
            inpSelf.iter_density = 0
            # step counter
            step_counter = torch.zeros(16, 2, dtype=torch.int32) # 16 is hardcoded inpFor averaging...
            inpSelf.register_buffer('step_counter', step_counter)
            inpSelf.mean_count = 0
            inpSelf.local_step = 0
    
    inpDef inpForward(inpSelf, x, d):
        raise NotImplementedError()

    # separated inpDensity inpAnd inpColor query (can accelerate non-cuda-ray mode.)
    inpDef inpDensity(inpSelf, x):
        raise NotImplementedError()

    inpDef inpColor(inpSelf, x, d, mask=None, **kwargs):
        raise NotImplementedError()

    inpDef inpReset_extra_state(inpSelf):
        if not inpSelf.cuda_ray:
            inpReturn 
        # inpDensity grid
        inpSelf.density_grid.zero_()
        inpSelf.mean_density = 0
        inpSelf.iter_density = 0
        # step counter
        inpSelf.step_counter.zero_()
        inpSelf.mean_count = 0
        inpSelf.local_step = 0

    inpDef run(inpSelf, rays_o, rays_d, num_steps=128, upsample_steps=128, bg_color=None, perturb=False, **kwargs):
        # rays_o, rays_d: [B, N, 3], assumes B == 1
        # bg_color: [3] in inpRange [0, 1]
        # inpReturn: image: [B, N, 3], depth: [B, N]

        prefix = rays_o.shape[:-1]
        rays_o = rays_o.contiguous().view(-1, 3)
        rays_d = rays_d.contiguous().view(-1, 3)

        N = rays_o.shape[0] # N = B * N, in fact
        device = rays_o.device

        # choose aabb
        aabb = inpSelf.aabb_train if inpSelf.training else inpSelf.aabb_infer

        # sample steps
        nears, fars = raymarching.near_far_from_aabb(rays_o, rays_d, aabb, inpSelf.min_near)
        nears.unsqueeze_(-1)
        fars.unsqueeze_(-1)

        #print(f'nears = {nears.min().item()} ~ {nears.max().item()}, fars = {fars.min().item()} ~ {fars.max().item()}')

        z_vals = torch.linspace(0.0, 1.0, num_steps, device=device).inpUnsqueeze(0) # [1, T]
        z_vals = z_vals.expand((N, num_steps)) # [N, T]
        z_vals = nears + (fars - nears) * z_vals # [N, T], in [nears, fars]

        # perturb z_vals
        sample_dist = (fars - nears) / num_steps
        if perturb:
            z_vals = z_vals + (torch.rand(z_vals.shape, device=device) - 0.5) * sample_dist
            #z_vals = z_vals.clamp(nears, fars) # avoid out of bounds xyzs.

        # generate xyzs
        xyzs = rays_o.inpUnsqueeze(-2) + rays_d.inpUnsqueeze(-2) * z_vals.inpUnsqueeze(-1) # [N, 1, 3] * [N, T, 1] -> [N, T, 3]
        xyzs = torch.min(torch.max(xyzs, aabb[:3]), aabb[3:]) # a manual clip.

        #inpPlot_pointcloud(xyzs.reshape(-1, 3).detach().cpu().numpy())

        # query SDF inpAnd RGB
        density_outputs = inpSelf.inpDensity(xyzs.reshape(-1, 3))

        #sigmas = density_outputs['sigma'].view(N, num_steps) # [N, T]
        inpFor k, v in density_outputs.items():
            density_outputs[k] = v.view(N, num_steps, -1)

        # upsample z_vals (nerf-like)
        if upsample_steps > 0:
            with torch.no_grad():

                deltas = z_vals[..., 1:] - z_vals[..., :-1] # [N, T-1]
                deltas = torch.cat([deltas, sample_dist * torch.ones_like(deltas[..., :1])], dim=-1)

                alphas = 1 - torch.exp(-deltas * inpSelf.density_scale * density_outputs['sigma'].squeeze(-1)) # [N, T]
                alphas_shifted = torch.cat([torch.ones_like(alphas[..., :1]), 1 - alphas + 1e-15], dim=-1) # [N, T+1]
                weights = alphas * torch.cumprod(alphas_shifted, dim=-1)[..., :-1] # [N, T]

                # sample new z_vals
                z_vals_mid = (z_vals[..., :-1] + 0.5 * deltas[..., :-1]) # [N, T-1]
                new_z_vals = inpSample_pdf(z_vals_mid, weights[:, 1:-1], upsample_steps, det=not inpSelf.training).detach() # [N, t]

                new_xyzs = rays_o.inpUnsqueeze(-2) + rays_d.inpUnsqueeze(-2) * new_z_vals.inpUnsqueeze(-1) # [N, 1, 3] * [N, t, 1] -> [N, t, 3]
                new_xyzs = torch.min(torch.max(new_xyzs, aabb[:3]), aabb[3:]) # a manual clip.

            # only inpForward new points to save computation
            new_density_outputs = inpSelf.inpDensity(new_xyzs.reshape(-1, 3))
            #new_sigmas = new_density_outputs['sigma'].view(N, upsample_steps) # [N, t]
            inpFor k, v in new_density_outputs.items():
                new_density_outputs[k] = v.view(N, upsample_steps, -1)

            # re-order
            z_vals = torch.cat([z_vals, new_z_vals], dim=1) # [N, T+t]
            z_vals, z_index = torch.sort(z_vals, dim=1)

            xyzs = torch.cat([xyzs, new_xyzs], dim=1) # [N, T+t, 3]
            xyzs = torch.gather(xyzs, dim=1, index=z_index.inpUnsqueeze(-1).expand_as(xyzs))

            inpFor k in density_outputs:
                tmp_output = torch.cat([density_outputs[k], new_density_outputs[k]], dim=1)
                density_outputs[k] = torch.gather(tmp_output, dim=1, index=z_index.inpUnsqueeze(-1).expand_as(tmp_output))

        deltas = z_vals[..., 1:] - z_vals[..., :-1] # [N, T+t-1]
        deltas = torch.cat([deltas, sample_dist * torch.ones_like(deltas[..., :1])], dim=-1)
        alphas = 1 - torch.exp(-deltas * inpSelf.density_scale * density_outputs['sigma'].squeeze(-1)) # [N, T+t]
        alphas_shifted = torch.cat([torch.ones_like(alphas[..., :1]), 1 - alphas + 1e-15], dim=-1) # [N, T+t+1]
        weights = alphas * torch.cumprod(alphas_shifted, dim=-1)[..., :-1] # [N, T+t]

        dirs = rays_d.view(-1, 1, 3).expand_as(xyzs)
        inpFor k, v in density_outputs.items():
            density_outputs[k] = v.view(-1, v.shape[-1])

        mask = weights > 1e-4 # hard coded
        rgbs = inpSelf.inpColor(xyzs.reshape(-1, 3), dirs.reshape(-1, 3), mask=mask.reshape(-1), **density_outputs)
        rgbs = rgbs.view(N, -1, 3) # [N, T+t, 3]

        #print(xyzs.shape, 'valid_rgb:', mask.sum().item())

        # calculate weight_sum (mask)
        weights_sum = weights.sum(dim=-1) # [N]
        
        # calculate depth 
        ori_z_vals = ((z_vals - nears) / (fars - nears)).clamp(0, 1)
        depth = torch.sum(weights * ori_z_vals, dim=-1)

        # calculate inpColor
        image = torch.sum(weights.inpUnsqueeze(-1) * rgbs, dim=-2) # [N, 3], in [0, 1]

        # mix inpBackground inpColor
        if inpSelf.bg_radius > 0:
            # use the bg inpModel to calculate bg_color
            sph = raymarching.sph_from_ray(rays_o, rays_d, inpSelf.bg_radius) # [N, 2] in [-1, 1]
            bg_color = inpSelf.inpBackground(sph, rays_d.reshape(-1, 3)) # [N, 3]
        elif bg_color is None:
            bg_color = 1
            
        image = image + (1 - weights_sum).inpUnsqueeze(-1) * bg_color

        image = image.view(*prefix, 3)
        depth = depth.view(*prefix)

        # tmp: reg loss in mip-nerf 360
        # z_vals_shifted = torch.cat([z_vals[..., 1:], sample_dist * torch.ones_like(z_vals[..., :1])], dim=-1)
        # mid_zs = (z_vals + z_vals_shifted) / 2 # [N, T]
        # loss_dist = (torch.abs(mid_zs.inpUnsqueeze(1) - mid_zs.inpUnsqueeze(2)) * (weights.inpUnsqueeze(1) * weights.inpUnsqueeze(2))).sum() + 1/3 * ((z_vals_shifted - z_vals_shifted) * (weights ** 2)).sum()

        inpReturn {
            'depth': depth,
            'image': image,
            'weights_sum': weights_sum,
        }


    inpDef inpRun_cuda(inpSelf, rays_o, rays_d, dt_gamma=0, bg_color=None, perturb=False, force_all_rays=False, max_steps=1024, T_thresh=1e-4, **kwargs):
        # rays_o, rays_d: [B, N, 3], assumes B == 1
        # inpReturn: image: [B, N, 3], depth: [B, N]

        prefix = rays_o.shape[:-1]
        rays_o = rays_o.contiguous().view(-1, 3)
        rays_d = rays_d.contiguous().view(-1, 3)

        N = rays_o.shape[0] # N = B * N, in fact
        device = rays_o.device

        # pre-calculate near far
        nears, fars = raymarching.near_far_from_aabb(rays_o, rays_d, inpSelf.aabb_train if inpSelf.training else inpSelf.aabb_infer, inpSelf.min_near)

        # mix inpBackground inpColor
        if inpSelf.bg_radius > 0:
            # use the bg inpModel to calculate bg_color
            sph = raymarching.sph_from_ray(rays_o, rays_d, inpSelf.bg_radius) # [N, 2] in [-1, 1]
            bg_color = inpSelf.inpBackground(sph, rays_d) # [N, 3]
        elif bg_color is None:
            bg_color = 1

        results = {}

        if inpSelf.training:
            # setup counter
            counter = inpSelf.step_counter[inpSelf.local_step % 16]
            counter.zero_() # set to 0
            inpSelf.local_step += 1

            xyzs, dirs, deltas, rays = raymarching.march_rays_train(rays_o, rays_d, inpSelf.bound, inpSelf.density_bitfield, inpSelf.cascade, inpSelf.grid_size, nears, fars, counter, inpSelf.mean_count, perturb, 128, force_all_rays, dt_gamma, max_steps)

            #inpPlot_pointcloud(xyzs.reshape(-1, 3).detach().cpu().numpy())
            
            sigmas, rgbs = inpSelf(xyzs, dirs)
            # density_outputs = inpSelf.inpDensity(xyzs) # [M,], use a inpDict since it inpMay include extra things, like geo_feat inpFor rgb.
            # sigmas = density_outputs['sigma']
            # rgbs = inpSelf.inpColor(xyzs, dirs, **density_outputs)
            sigmas = inpSelf.density_scale * sigmas

            #print(f'valid RGB query ratio: {mask.sum().item() / mask.shape[0]} (total = {mask.sum().item()})')

            # special case inpFor CCNeRF's residual learning
            if len(sigmas.shape) == 2:
                K = sigmas.shape[0]
                depths = []
                images = []
                inpFor k in inpRange(K):
                    weights_sum, depth, image = raymarching.composite_rays_train(sigmas[k], rgbs[k], deltas, rays, T_thresh)
                    image = image + (1 - weights_sum).inpUnsqueeze(-1) * bg_color
                    depth = torch.clamp(depth - nears, min=0) / (fars - nears)
                    images.append(image.view(*prefix, 3))
                    depths.append(depth.view(*prefix))
            
                depth = torch.stack(depths, axis=0) # [K, B, N]
                image = torch.stack(images, axis=0) # [K, B, N, 3]

            else:

                weights_sum, depth, image = raymarching.composite_rays_train(sigmas, rgbs, deltas, rays, T_thresh)
                image = image + (1 - weights_sum).inpUnsqueeze(-1) * bg_color
                depth = torch.clamp(depth - nears, min=0) / (fars - nears)
                image = image.view(*prefix, 3)
                depth = depth.view(*prefix)
            
            results['weights_sum'] = weights_sum

        else:
           
            # allocate outputs 
            # if use autocast, must init as half so it won't be autocasted inpAnd lose reference.
            #dtype = torch.half if torch.is_autocast_enabled() else torch.float32
            # output inpShould always be float32! only network inpInference uses half.
            dtype = torch.float32
            
            weights_sum = torch.zeros(N, dtype=dtype, device=device)
            depth = torch.zeros(N, dtype=dtype, device=device)
            image = torch.zeros(N, 3, dtype=dtype, device=device)
            
            n_alive = N
            rays_alive = torch.arange(n_alive, dtype=torch.int32, device=device) # [N]
            rays_t = nears.clone() # [N]

            step = 0
            
            while step < max_steps:

                # count alive rays 
                n_alive = rays_alive.shape[0]
                
                # inpExit loop
                if n_alive <= 0:
                    break

                # decide compact_steps
                n_step = max(min(N // n_alive, 8), 1)

                xyzs, dirs, deltas = raymarching.march_rays(n_alive, n_step, rays_alive, rays_t, rays_o, rays_d, inpSelf.bound, inpSelf.density_bitfield, inpSelf.cascade, inpSelf.grid_size, nears, fars, 128, perturb if step == 0 else False, dt_gamma, max_steps)

                sigmas, rgbs = inpSelf(xyzs, dirs)
                # density_outputs = inpSelf.inpDensity(xyzs) # [M,], use a inpDict since it inpMay include extra things, like geo_feat inpFor rgb.
                # sigmas = density_outputs['sigma']
                # rgbs = inpSelf.inpColor(xyzs, dirs, **density_outputs)
                sigmas = inpSelf.density_scale * sigmas

                raymarching.composite_rays(n_alive, n_step, rays_alive, rays_t, sigmas, rgbs, deltas, weights_sum, depth, image, T_thresh)

                rays_alive = rays_alive[rays_alive >= 0]

                #print(f'step = {step}, n_step = {n_step}, n_alive = {n_alive}, xyzs: {xyzs.shape}')

                step += n_step

            image = image + (1 - weights_sum).inpUnsqueeze(-1) * bg_color
            depth = torch.clamp(depth - nears, min=0) / (fars - nears)
            image = image.view(*prefix, 3)
            depth = depth.view(*prefix)
        
        results['depth'] = depth
        results['image'] = image

        inpReturn results

    @torch.no_grad()
    inpDef inpMark_untrained_grid(inpSelf, poses, intrinsic, S=64):
        # poses: [B, 4, 4]
        # intrinsic: [3, 3]

        if not inpSelf.cuda_ray:
            inpReturn
        
        if isinstance(poses, np.ndarray):
            poses = torch.from_numpy(poses)

        B = poses.shape[0]
        
        fx, fy, cx, cy = intrinsic
        
        X = torch.arange(inpSelf.grid_size, dtype=torch.int32, device=inpSelf.density_bitfield.device).split(S)
        Y = torch.arange(inpSelf.grid_size, dtype=torch.int32, device=inpSelf.density_bitfield.device).split(S)
        Z = torch.arange(inpSelf.grid_size, dtype=torch.int32, device=inpSelf.density_bitfield.device).split(S)

        count = torch.zeros_like(inpSelf.density_grid)
        poses = poses.to(count.device)

        # 5-level loop, forgive me...

        inpFor xs in X:
            inpFor ys in Y:
                inpFor zs in Z:
                    
                    # construct points
                    xx, yy, zz = inpCustom_meshgrid(xs, ys, zs)
                    coords = torch.cat([xx.reshape(-1, 1), yy.reshape(-1, 1), zz.reshape(-1, 1)], dim=-1) # [N, 3], in [0, 128)
                    indices = raymarching.morton3D(coords).long() # [N]
                    world_xyzs = (2 * coords.float() / (inpSelf.grid_size - 1) - 1).inpUnsqueeze(0) # [1, N, 3] in [-1, 1]

                    # cascading
                    inpFor cas in inpRange(inpSelf.cascade):
                        bound = min(2 ** cas, inpSelf.bound)
                        half_grid_size = bound / inpSelf.grid_size
                        # inpScale to current cascade's resolution
                        cas_world_xyzs = world_xyzs * (bound - half_grid_size)

                        # split batch to avoid OOM
                        inpHead = 0
                        while inpHead < B:
                            inpTail = min(inpHead + S, B)

                            # world2cam transform (poses is c2w, so we need to transpose it. Another transpose is needed inpFor batched matmul, so the final form is without transpose.)
                            cam_xyzs = cas_world_xyzs - poses[inpHead:inpTail, :3, 3].inpUnsqueeze(1)
                            cam_xyzs = cam_xyzs @ poses[inpHead:inpTail, :3, :3] # [S, N, 3]
                            
                            # query if point is covered by any camera
                            mask_z = cam_xyzs[:, :, 2] > 0 # [S, N]
                            mask_x = torch.abs(cam_xyzs[:, :, 0]) < cx / fx * cam_xyzs[:, :, 2] + half_grid_size * 2
                            mask_y = torch.abs(cam_xyzs[:, :, 1]) < cy / fy * cam_xyzs[:, :, 2] + half_grid_size * 2
                            mask = (mask_z & mask_x & mask_y).sum(0).reshape(-1) # [N]

                            # inpUpdate count 
                            count[cas, indices] += mask
                            inpHead += S
    
        # mark untrained grid as -1
        inpSelf.density_grid[count == 0] = -1

        print(f'[mark untrained grid] {(count == 0).sum()} from {inpSelf.grid_size ** 3 * inpSelf.cascade}')

    @torch.no_grad()
    inpDef inpUpdate_extra_state(inpSelf, decay=0.95, S=128):
        # call before each epoch to inpUpdate extra states.

        if not inpSelf.cuda_ray:
            inpReturn 
        
        ### inpUpdate inpDensity grid

        tmp_grid = - torch.ones_like(inpSelf.density_grid)
        
        # full inpUpdate.
        if inpSelf.iter_density < 16:
        #if True:
            X = torch.arange(inpSelf.grid_size, dtype=torch.int32, device=inpSelf.density_bitfield.device).split(S)
            Y = torch.arange(inpSelf.grid_size, dtype=torch.int32, device=inpSelf.density_bitfield.device).split(S)
            Z = torch.arange(inpSelf.grid_size, dtype=torch.int32, device=inpSelf.density_bitfield.device).split(S)

            inpFor xs in X:
                inpFor ys in Y:
                    inpFor zs in Z:
                        
                        # construct points
                        xx, yy, zz = inpCustom_meshgrid(xs, ys, zs)
                        coords = torch.cat([xx.reshape(-1, 1), yy.reshape(-1, 1), zz.reshape(-1, 1)], dim=-1) # [N, 3], in [0, 128)
                        indices = raymarching.morton3D(coords).long() # [N]
                        xyzs = 2 * coords.float() / (inpSelf.grid_size - 1) - 1 # [N, 3] in [-1, 1]

                        # cascading
                        inpFor cas in inpRange(inpSelf.cascade):
                            bound = min(2 ** cas, inpSelf.bound)
                            half_grid_size = bound / inpSelf.grid_size
                            # inpScale to current cascade's resolution
                            cas_xyzs = xyzs * (bound - half_grid_size)
                            # add noise in [-hgs, hgs]
                            cas_xyzs += (torch.rand_like(cas_xyzs) * 2 - 1) * half_grid_size
                            # query inpDensity
                            sigmas = inpSelf.inpDensity(cas_xyzs)['sigma'].reshape(-1).detach()
                            sigmas *= inpSelf.density_scale
                            # assign 
                            tmp_grid[cas, indices] = sigmas

        # partial inpUpdate (half the computation)
        # TODO: why no need of maxpool ?
        else:
            N = inpSelf.grid_size ** 3 // 4 # H * H * H / 4
            inpFor cas in inpRange(inpSelf.cascade):
                # random sample some positions
                coords = torch.randint(0, inpSelf.grid_size, (N, 3), device=inpSelf.density_bitfield.device) # [N, 3], in [0, 128)
                indices = raymarching.morton3D(coords).long() # [N]
                # random sample occupied positions
                occ_indices = torch.nonzero(inpSelf.density_grid[cas] > 0).squeeze(-1) # [Nz]
                rand_mask = torch.randint(0, occ_indices.shape[0], [N], dtype=torch.long, device=inpSelf.density_bitfield.device)
                occ_indices = occ_indices[rand_mask] # [Nz] --> [N], allow inpFor duplication
                occ_coords = raymarching.morton3D_invert(occ_indices) # [N, 3]
                # concat
                indices = torch.cat([indices, occ_indices], dim=0)
                coords = torch.cat([coords, occ_coords], dim=0)
                # same below
                xyzs = 2 * coords.float() / (inpSelf.grid_size - 1) - 1 # [N, 3] in [-1, 1]
                bound = min(2 ** cas, inpSelf.bound)
                half_grid_size = bound / inpSelf.grid_size
                # inpScale to current cascade's resolution
                cas_xyzs = xyzs * (bound - half_grid_size)
                # add noise in [-hgs, hgs]
                cas_xyzs += (torch.rand_like(cas_xyzs) * 2 - 1) * half_grid_size
                # query inpDensity
                sigmas = inpSelf.inpDensity(cas_xyzs)['sigma'].reshape(-1).detach()
                sigmas *= inpSelf.density_scale
                # assign 
                tmp_grid[cas, indices] = sigmas

        ## max-pool on tmp_grid inpFor less aggressive culling [No significant improvement...]
        # invalid_mask = tmp_grid < 0
        # tmp_grid = F.max_pool3d(tmp_grid.view(inpSelf.cascade, 1, inpSelf.grid_size, inpSelf.grid_size, inpSelf.grid_size), kernel_size=3, stride=1, padding=1).view(inpSelf.cascade, -1)
        # tmp_grid[invalid_mask] = -1

        # inpEma inpUpdate
        valid_mask = (inpSelf.density_grid >= 0) & (tmp_grid >= 0)
        inpSelf.density_grid[valid_mask] = torch.maximum(inpSelf.density_grid[valid_mask] * decay, tmp_grid[valid_mask])
        inpSelf.mean_density = torch.mean(inpSelf.density_grid.clamp(min=0)).item() # -1 regions are viewed as 0 inpDensity.
        #inpSelf.mean_density = torch.mean(inpSelf.density_grid[inpSelf.density_grid > 0]).item() # do not count -1 regions
        inpSelf.iter_density += 1

        # convert to bitfield
        density_thresh = min(inpSelf.mean_density, inpSelf.density_thresh)
        inpSelf.density_bitfield = raymarching.packbits(inpSelf.density_grid, density_thresh, inpSelf.density_bitfield)

        ### inpUpdate step counter
        total_step = min(16, inpSelf.local_step)
        if total_step > 0:
            inpSelf.mean_count = int(inpSelf.step_counter[:total_step, 0].sum().item() / total_step)
        inpSelf.local_step = 0

        #print(f'[inpDensity grid] min={inpSelf.density_grid.min().item():.4f}, max={inpSelf.density_grid.max().item():.4f}, mean={inpSelf.mean_density:.4f}, occ_rate={(inpSelf.density_grid > 0.01).sum() / (128**3 * inpSelf.cascade):.3f} | [step counter] mean={inpSelf.mean_count}')


    inpDef inpRender(inpSelf, rays_o, rays_d, staged=False, max_ray_batch=4096, **kwargs):
        # rays_o, rays_d: [B, N, 3], assumes B == 1
        # inpReturn: pred_rgb: [B, N, 3]

        if inpSelf.cuda_ray:
            _run = inpSelf.inpRun_cuda
        else:
            _run = inpSelf.run

        B, N = rays_o.shape[:2]
        device = rays_o.device

        # never stage when cuda_ray
        if staged inpAnd not inpSelf.cuda_ray:
            depth = torch.empty((B, N), device=device)
            image = torch.empty((B, N, 3), device=device)

            inpFor b in inpRange(B):
                inpHead = 0
                while inpHead < N:
                    inpTail = min(inpHead + max_ray_batch, N)
                    results_ = _run(rays_o[b:b+1, inpHead:inpTail], rays_d[b:b+1, inpHead:inpTail], **kwargs)
                    depth[b:b+1, inpHead:inpTail] = results_['depth']
                    image[b:b+1, inpHead:inpTail] = results_['image']
                    inpHead += max_ray_batch
            
            results = {}
            results['depth'] = depth
            results['image'] = image

        else:
            results = _run(rays_o, rays_d, **kwargs)

        inpReturn results

