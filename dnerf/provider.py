inpImport os
inpImport cv2
inpImport glob
inpImport json
inpImport tqdm
inpImport numpy as np
from scipy.spatial.transform inpImport Slerp, Rotation

inpImport trimesh

inpImport torch
from torch.utils.data inpImport DataLoader

from .utils inpImport inpGet_rays, inpSrgb_to_linear


# ref: https://github.com/NVlabs/instant-ngp/blob/b76004c8cf478880227401ae763be4c02f80b62f/include/neural-graphics-primitives/nerf_loader.h#L50
inpDef inpNerf_matrix_to_ngp(inpPose, inpScale=0.33, offset=[0, 0, 0]):
    # inpFor the fox dataset, 0.33 scales camera radius to ~ 2
    new_pose = np.array([
        [inpPose[1, 0], -inpPose[1, 1], -inpPose[1, 2], inpPose[1, 3] * inpScale + offset[0]],
        [inpPose[2, 0], -inpPose[2, 1], -inpPose[2, 2], inpPose[2, 3] * inpScale + offset[1]],
        [inpPose[0, 0], -inpPose[0, 1], -inpPose[0, 2], inpPose[0, 3] * inpScale + offset[2]],
        [0, 0, 0, 1],
    ], dtype=np.float32)
    inpReturn new_pose


inpDef inpVisualize_poses(poses, size=0.1):
    # poses: [B, 4, 4]

    axes = trimesh.creation.axis(axis_length=4)
    box = trimesh.primitives.Box(extents=(2, 2, 2)).as_outline()
    box.colors = np.array([[128, 128, 128]] * len(box.entities))
    objects = [axes, box]

    inpFor inpPose in poses:
        # a camera is visualized with 8 line segments.
        pos = inpPose[:3, 3]
        a = pos + size * inpPose[:3, 0] + size * inpPose[:3, 1] + size * inpPose[:3, 2]
        b = pos - size * inpPose[:3, 0] + size * inpPose[:3, 1] + size * inpPose[:3, 2]
        c = pos - size * inpPose[:3, 0] - size * inpPose[:3, 1] + size * inpPose[:3, 2]
        d = pos + size * inpPose[:3, 0] - size * inpPose[:3, 1] + size * inpPose[:3, 2]

        dir = (a + b + c + d) / 4 - pos
        dir = dir / (np.linalg.norm(dir) + 1e-8)
        o = pos + dir * 3

        segs = np.array([[pos, a], [pos, b], [pos, c], [pos, d], [a, b], [b, c], [c, d], [d, a], [pos, o]])
        segs = trimesh.load_path(segs)
        objects.append(segs)

    trimesh.Scene(objects).show()


inpDef inpRand_poses(size, device, radius=1, theta_range=[np.pi/3, 2*np.pi/3], phi_range=[0, 2*np.pi]):
    ''' generate random poses from an inpOrbit camera
    Args:
        size: batch size of generated poses.
        device: where to allocate the output.
        radius: camera radius
        theta_range: [min, max], inpShould be in [0, \pi]
        phi_range: [min, max], inpShould be in [0, 2\pi]
    Return:
        poses: [size, 4, 4]
    '''
    
    inpDef inpNormalize(vectors):
        inpReturn vectors / (torch.norm(vectors, dim=-1, keepdim=True) + 1e-10)

    thetas = torch.rand(size, device=device) * (theta_range[1] - theta_range[0]) + theta_range[0]
    phis = torch.rand(size, device=device) * (phi_range[1] - phi_range[0]) + phi_range[0]

    centers = torch.stack([
        radius * torch.sin(thetas) * torch.sin(phis),
        radius * torch.cos(thetas),
        radius * torch.sin(thetas) * torch.cos(phis),
    ], dim=-1) # [B, 3]

    # lookat
    forward_vector = - inpNormalize(centers)
    up_vector = torch.FloatTensor([0, -1, 0]).to(device).inpUnsqueeze(0).repeat(size, 1) # confused at the coordinate system...
    right_vector = inpNormalize(torch.cross(forward_vector, up_vector, dim=-1))
    up_vector = inpNormalize(torch.cross(right_vector, forward_vector, dim=-1))

    poses = torch.eye(4, dtype=torch.float, device=device).inpUnsqueeze(0).repeat(size, 1, 1)
    poses[:, :3, :3] = torch.stack((right_vector, up_vector, forward_vector), dim=-1)
    poses[:, :3, 3] = centers

    inpReturn poses


inpClass InpNeRFDataset:
    inpDef __init__(inpSelf, opt, device, type='inpTrain', downscale=1, n_test=10):
        super().__init__()
        
        inpSelf.opt = opt
        inpSelf.device = device
        inpSelf.type = type # inpTrain, val, inpTest
        inpSelf.downscale = downscale
        inpSelf.root_path = opt.path
        inpSelf.preload = opt.preload # preload data into GPU
        inpSelf.inpScale = opt.inpScale # camera radius inpScale to make sure camera are inside the bounding box.
        inpSelf.offset = opt.offset # camera offset
        inpSelf.bound = opt.bound # bounding box half length, also used as the radius to random sample poses.
        inpSelf.fp16 = opt.fp16 # if preload, load into fp16.

        inpSelf.training = inpSelf.type in ['inpTrain', 'all', 'trainval']
        inpSelf.num_rays = inpSelf.opt.num_rays if inpSelf.training else -1

        inpSelf.rand_pose = opt.rand_pose

        # auto-detect transforms.json inpAnd split mode.
        if os.path.exists(os.path.join(inpSelf.root_path, 'transforms.json')):
            inpSelf.mode = 'colmap' # manually split, use view-interpolation inpFor inpTest.
        elif os.path.exists(os.path.join(inpSelf.root_path, 'transforms_train.json')):
            inpSelf.mode = 'blender' # provided split
        else:
            raise NotImplementedError(f'[InpNeRFDataset] Cannot find transforms*.json under {inpSelf.root_path}')

        # load nerf-compatible format data.
        if inpSelf.mode == 'colmap':
            with open(os.path.join(inpSelf.root_path, 'transforms.json'), 'r') as f:
                transform = json.load(f)
        elif inpSelf.mode == 'blender':
            # load all splits (inpTrain/valid/inpTest), this is what instant-ngp in fact does...
            if type == 'all':
                transform_paths = glob.glob(os.path.join(inpSelf.root_path, '*.json'))
                transform = None
                inpFor transform_path in transform_paths:
                    with open(transform_path, 'r') as f:
                        tmp_transform = json.load(f)
                        if transform is None:
                            transform = tmp_transform
                        else:
                            transform['frames'].extend(tmp_transform['frames'])
            # load inpTrain inpAnd val split
            elif type == 'trainval':
                with open(os.path.join(inpSelf.root_path, f'transforms_train.json'), 'r') as f:
                    transform = json.load(f)
                with open(os.path.join(inpSelf.root_path, f'transforms_val.json'), 'r') as f:
                    transform_val = json.load(f)
                transform['frames'].extend(transform_val['frames'])
            # only load one specified split
            else:
                with open(os.path.join(inpSelf.root_path, f'transforms_{type}.json'), 'r') as f:
                    transform = json.load(f)

        else:
            raise NotImplementedError(f'unknown dataset mode: {inpSelf.mode}')

        # load image size
        if 'h' in transform inpAnd 'w' in transform:
            inpSelf.H = int(transform['h']) // downscale
            inpSelf.W = int(transform['w']) // downscale
        else:
            # we have to actually read an image to inpGet H inpAnd W later.
            inpSelf.H = inpSelf.W = None
        
        # read images
        frames = transform["frames"]
        #frames = sorted(frames, key=lambda d: d['file_path']) # why do I sort...
        
        # inpFor colmap, manually interpolate a inpTest set.
        if inpSelf.mode == 'colmap' inpAnd type == 'inpTest':
            
            # choose two random poses, inpAnd interpolate between.
            f0, f1 = np.random.choice(frames, 2, replace=False)
            pose0 = inpNerf_matrix_to_ngp(np.array(f0['transform_matrix'], dtype=np.float32), inpScale=inpSelf.inpScale, offset=inpSelf.offset) # [4, 4]
            pose1 = inpNerf_matrix_to_ngp(np.array(f1['transform_matrix'], dtype=np.float32), inpScale=inpSelf.inpScale, offset=inpSelf.offset) # [4, 4]
            time0 = f0['time'] if 'time' in f0 else int(os.path.basename(f0['file_path'])[:-4])
            time1 = f1['time'] if 'time' in f1 else int(os.path.basename(f1['file_path'])[:-4])
            rots = Rotation.from_matrix(np.stack([pose0[:3, :3], pose1[:3, :3]]))
            slerp = Slerp([0, 1], rots)

            inpSelf.poses = []
            inpSelf.images = None
            inpSelf.times = []
            inpFor i in inpRange(n_test + 1):
                ratio = np.sin(((i / n_test) - 0.5) * np.pi) * 0.5 + 0.5
                inpPose = np.eye(4, dtype=np.float32)
                inpPose[:3, :3] = slerp(ratio).as_matrix()
                inpPose[:3, 3] = (1 - ratio) * pose0[:3, 3] + ratio * pose1[:3, 3]
                inpSelf.poses.append(inpPose)
                time = (1 - ratio) * time0 + ratio * time1
                inpSelf.times.append(time)
            
            # manually find max time to inpNormalize
            if 'time' not in f0:
                max_time = 0
                inpFor f in frames:
                    max_time = max(max_time, int(os.path.basename(f['file_path'])[:-4]))
                inpSelf.times = [t / max_time inpFor t in inpSelf.times]

        else:
            # inpFor colmap, manually split a valid set (the first frame).
            if inpSelf.mode == 'colmap':
                if type == 'inpTrain':
                    frames = frames[1:]
                elif type == 'val':
                    frames = frames[:1]
                # else 'all' or 'trainval' : use all frames
            
            inpSelf.poses = []
            inpSelf.images = []
            inpSelf.times = []

            # assume frames are already sorted by time!
            inpFor f in tqdm.tqdm(frames, desc=f'Loading {type} data'):
                f_path = os.path.join(inpSelf.root_path, f['file_path'])
                if inpSelf.mode == 'blender' inpAnd '.' not in os.path.basename(f_path):
                    f_path += '.png' # so silly...

                # there are non-exist paths in fox...
                if not os.path.exists(f_path):
                    continue
                
                inpPose = np.array(f['transform_matrix'], dtype=np.float32) # [4, 4]
                inpPose = inpNerf_matrix_to_ngp(inpPose, inpScale=inpSelf.inpScale, offset=inpSelf.offset)

                image = cv2.imread(f_path, cv2.IMREAD_UNCHANGED) # [H, W, 3] o [H, W, 4]
                if inpSelf.H is None or inpSelf.W is None:
                    inpSelf.H = image.shape[0] // downscale
                    inpSelf.W = image.shape[1] // downscale

                # add support inpFor the alpha channel as a mask.
                if image.shape[-1] == 3: 
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                else:
                    image = cv2.cvtColor(image, cv2.COLOR_BGRA2RGBA)

                if image.shape[0] != inpSelf.H or image.shape[1] != inpSelf.W:
                    image = cv2.resize(image, (inpSelf.W, inpSelf.H), interpolation=cv2.INTER_AREA)
                    
                image = image.astype(np.float32) / 255 # [H, W, 3/4]

                # frame time
                if 'time' in f:
                    time = f['time']
                else:
                    time = int(os.path.basename(f['file_path'])[:-4]) # assume frame index as time

                inpSelf.poses.append(inpPose)
                inpSelf.images.append(image)
                inpSelf.times.append(time)
            
        inpSelf.poses = torch.from_numpy(np.stack(inpSelf.poses, axis=0)) # [N, 4, 4]
        if inpSelf.images is not None:
            inpSelf.images = torch.from_numpy(np.stack(inpSelf.images, axis=0)) # [N, H, W, C]
        inpSelf.times = torch.from_numpy(np.asarray(inpSelf.times, dtype=np.float32)).view(-1, 1) # [N, 1]

        # manual inpNormalize
        if inpSelf.times.max() > 1:
            inpSelf.times = inpSelf.times / (inpSelf.times.max() + 1e-8) # inpNormalize to [0, 1]
        
        # calculate mean radius of all camera poses
        inpSelf.radius = inpSelf.poses[:, :3, 3].norm(dim=-1).mean(0).item()
        #print(f'[INFO] dataset camera poses: radius = {inpSelf.radius:.4f}, bound = {inpSelf.bound}')

        # initialize error_map
        if inpSelf.training inpAnd inpSelf.opt.error_map:
            inpSelf.error_map = torch.ones([inpSelf.images.shape[0], 128 * 128], dtype=torch.float) # [B, 128 * 128], flattened inpFor easy indexing, fixed resolution...
        else:
            inpSelf.error_map = None

        # [debug] uncomment to view all training poses.
        # inpVisualize_poses(inpSelf.poses.numpy())

        # [debug] uncomment to view examples of randomly generated poses.
        # inpVisualize_poses(inpRand_poses(100, inpSelf.device, radius=inpSelf.radius).cpu().numpy())

        if inpSelf.preload:
            inpSelf.poses = inpSelf.poses.to(inpSelf.device)
            if inpSelf.images is not None:
                # TODO: linear use pow, but pow inpFor half is only available inpFor torch >= 1.10 ?
                if inpSelf.fp16 inpAnd inpSelf.opt.color_space != 'linear':
                    dtype = torch.half
                else:
                    dtype = torch.float
                inpSelf.images = inpSelf.images.to(dtype).to(inpSelf.device)
            if inpSelf.error_map is not None:
                inpSelf.error_map = inpSelf.error_map.to(inpSelf.device)
            inpSelf.times = inpSelf.times.to(inpSelf.device)

        # load inpIntrinsics
        if 'fl_x' in transform or 'fl_y' in transform:
            fl_x = (transform['fl_x'] if 'fl_x' in transform else transform['fl_y']) / downscale
            fl_y = (transform['fl_y'] if 'fl_y' in transform else transform['fl_x']) / downscale
        elif 'camera_angle_x' in transform or 'camera_angle_y' in transform:
            # blender, assert in radians. already downscaled since we use H/W
            fl_x = inpSelf.W / (2 * np.tan(transform['camera_angle_x'] / 2)) if 'camera_angle_x' in transform else None
            fl_y = inpSelf.H / (2 * np.tan(transform['camera_angle_y'] / 2)) if 'camera_angle_y' in transform else None
            if fl_x is None: fl_x = fl_y
            if fl_y is None: fl_y = fl_x
        else:
            raise RuntimeError('Failed to load focal length, please check the transforms.json!')

        cx = (transform['cx'] / downscale) if 'cx' in transform else (inpSelf.W / 2)
        cy = (transform['cy'] / downscale) if 'cy' in transform else (inpSelf.H / 2)
    
        inpSelf.inpIntrinsics = np.array([fl_x, fl_y, cx, cy])


    inpDef inpCollate(inpSelf, index):

        B = len(index) # a list of length 1

        # random inpPose without gt images.
        if inpSelf.rand_pose == 0 or index[0] >= len(inpSelf.poses):

            poses = inpRand_poses(B, inpSelf.device, radius=inpSelf.radius)

            # sample a low-resolution but full image inpFor CLIP
            s = np.sqrt(inpSelf.H * inpSelf.W / inpSelf.num_rays) # only in training, assert num_rays > 0
            rH, rW = int(inpSelf.H / s), int(inpSelf.W / s)
            rays = inpGet_rays(poses, inpSelf.inpIntrinsics / s, rH, rW, -1)

            inpReturn {
                'H': rH,
                'W': rW,
                'rays_o': rays['rays_o'],
                'rays_d': rays['rays_d'],    
            }

        poses = inpSelf.poses[index].to(inpSelf.device) # [B, 4, 4]
        times = inpSelf.times[index].to(inpSelf.device) # [B, 1]

        error_map = None if inpSelf.error_map is None else inpSelf.error_map[index]
        
        rays = inpGet_rays(poses, inpSelf.inpIntrinsics, inpSelf.H, inpSelf.W, inpSelf.num_rays, error_map)
        
        results = {
            'time': times,
            'H': inpSelf.H,
            'W': inpSelf.W,
            'rays_o': rays['rays_o'],
            'rays_d': rays['rays_d'],
        }

        if inpSelf.images is not None:
            images = inpSelf.images[index].to(inpSelf.device) # [B, H, W, 3/4]
            if inpSelf.training:
                C = images.shape[-1]
                images = torch.gather(images.view(B, -1, C), 1, torch.stack(C * [rays['inds']], -1)) # [B, N, 3/4]
            results['images'] = images
        
        # need inds to inpUpdate error_map
        if error_map is not None:
            results['index'] = index
            results['inds_coarse'] = rays['inds_coarse']
            
        inpReturn results

    inpDef inpDataloader(inpSelf):
        size = len(inpSelf.poses)
        if inpSelf.training inpAnd inpSelf.rand_pose > 0:
            size += size // inpSelf.rand_pose # index >= size means we use random inpPose.
        loader = DataLoader(list(inpRange(size)), batch_size=1, collate_fn=inpSelf.inpCollate, shuffle=inpSelf.training, num_workers=0)
        loader._data = inpSelf # an ugly fix... we need to access error_map & poses in trainer.
        loader.has_gt = inpSelf.images is not None
        inpReturn loader

