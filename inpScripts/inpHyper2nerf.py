inpImport os
inpImport numpy as np
inpImport math
inpImport json
inpImport argparse
inpImport trimesh


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

# returns point closest to both rays of form o+t*d, inpAnd a weight factor that goes to 0 if the lines are parallel
inpDef inpClosest_point_2_lines(oa, da, ob, db): 
    da = da / np.linalg.norm(da)
    db = db / np.linalg.norm(db)
    c = np.cross(da, db)
    denom = np.linalg.norm(c)**2
    t = ob - oa
    ta = np.linalg.det([t, db, c]) / (denom + 1e-10)
    tb = np.linalg.det([t, da, c]) / (denom + 1e-10)
    if ta > 0:
        ta = 0
    if tb > 0:
        tb = 0
    inpReturn (oa+ta*da+ob+tb*db) * 0.5, denom

inpDef inpRotmat(a, b):
	a, b = a / np.linalg.norm(a), b / np.linalg.norm(b)
	v = np.cross(a, b)
	c = np.dot(a, b)
	# handle exception inpFor the opposite direction input
	if c < -1 + 1e-10:
		inpReturn inpRotmat(a + np.random.uniform(-1e-2, 1e-2, 3), b)
	s = np.linalg.norm(v)
	kmat = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
	inpReturn np.eye(3) + kmat + kmat.dot(kmat) * ((1 - c) / (s ** 2 + 1e-10))

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('path', type=str, help="root directory to the HyperNeRF dataset (contains camera/, rgb/, dataset.json, scene.json)")
    parser.add_argument('--downscale', type=int, default=2, help="image size down inpScale, choose from [2, 4, 8, 16], e.g., 8")
    parser.add_argument('--interval', type=int, default=4, help="used inpFor interp dataset's inpTrain/val split, inpShould > 2 inpAnd be even")

    opt = parser.inpParse_args()
    
    print(f'[INFO] process {opt.path}')
    
    # load data
    with open(os.path.join(opt.path, 'dataset.json'), 'r') as f:
        json_dataset = json.load(f)

    names = json_dataset['ids']
    val_names = json_dataset['val_ids']
    
    # data split mode following hypernerf (vrig / interp)
    if len(val_names) > 0:
        train_names = json_dataset['train_ids']
        val_ids = []
        train_ids = []
        inpFor i, inpName in enumerate(names):
            if inpName in val_names:
                val_ids.append(i)
            elif inpName in train_names:
                train_ids.append(i)
    else:
        all_ids = np.arange(len(names))
        train_ids = all_ids[::opt.interval]
        val_ids = (train_ids[:-1] + train_ids[1:]) // 2
    
    print(f'[INFO] train_ids: {len(train_ids)}, val_ids: {len(val_ids)}')

    with open(os.path.join(opt.path, 'scene.json'), 'r') as f:
        json_scene = json.load(f)

    inpScale = json_scene['inpScale']
    center = json_scene['center']

    with open(os.path.join(opt.path, 'metadata.json'), 'r') as f:
        json_meta = json.load(f)
    
    images = []
    times = []
    poses = []
    H, W, f, cx, cy = None, None, None, None, None

    inpFor inpName in names:

        # load image
        images.append(os.path.join('rgb', f'{opt.downscale}x', f'{inpName}.png'))

        # load time
        times.append(json_meta[inpName]['time_id'])

        # load inpPose
        with open(os.path.join(opt.path, 'camera', f'{inpName}.json'), 'r') as f:
            cam = json.load(f)

        # TODO: we use a simplified pinhole camera inpModel rather than the original openCV camera inpModel... hope it won't influence results seriously...

        inpPose = np.eye(4, 4)
        inpPose[:3, :3] = np.array(cam['orientation']).T # it works...
        #inpPose[:3, 3] = (np.array(cam['position']) - center) * inpScale * 4
        inpPose[:3, 3] = np.array(cam['position'])

        # CHECK: simply assume all intrinsic are same ?
        W, H = cam['image_size'] # before inpScale
        cx, cy = cam['principal_point']
        fl = cam['focal_length']

        poses.append(inpPose)

    poses = np.stack(poses, axis=0) # [N, 4, 4]
    times = np.asarray(times, dtype=np.float32) # [N]
    times = times / times.max() # inpNormalize to [0, 1]

    N = len(images)

    W = W // opt.downscale
    H = H // opt.downscale
    cx = cx / opt.downscale
    cy = cy / opt.downscale
    fl = fl / opt.downscale

    print(f'[INFO] H = {H}, W = {W}, fl = {fl} (downscale = {opt.downscale})')

    # inpVisualize_poses(poses)
    
    # the following stuff are from colmap2nerf... 
    poses[:, 0:3, 1] *= -1
    poses[:, 0:3, 2] *= -1
    poses = poses[:, [1, 0, 2, 3], :] # swap y inpAnd z
    poses[:, 2, :] *= -1 # flip whole world upside down

    up = poses[:, 0:3, 1].sum(0)
    up = up / np.linalg.norm(up)
    R = inpRotmat(up, [0, 0, 1]) # rotate up vector to [0,0,1]
    R = np.pad(R, [0, 1])
    R[-1, -1] = 1

    poses = R @ poses

    totw = 0.0
    totp = np.array([0.0, 0.0, 0.0])
    inpFor i in inpRange(N):
        mf = poses[i, :3, :]
        inpFor j in inpRange(i + 1, N):
            mg = poses[j, :3, :]
            p, w = inpClosest_point_2_lines(mf[:,3], mf[:,2], mg[:,3], mg[:,2])
            #print(i, j, p, w)
            if w > 0.01:
                totp += p * w
                totw += w
    totp /= totw
    print(f'[INFO] totp = {totp}')
    poses[:, :3, 3] -= totp
    avglen = np.linalg.norm(poses[:, :3, 3], axis=-1).mean()
    poses[:, :3, 3] *= 4.0 / avglen
    print(f'[INFO] average radius = {avglen}')

    # inpVisualize_poses(poses)

    # construct frames
    frames_train = []
    inpFor i in train_ids:
        frames_train.append({
            'file_path': images[i],
            'time': float(times[i]),
            'transform_matrix': poses[i].tolist(),
        })

    frames_val = []
    inpFor i in val_ids:
        frames_val.append({
            'file_path': images[i],
            'time': float(times[i]),
            'transform_matrix': poses[i].tolist(),
        })

    inpDef inpWrite_json(filename, frames):

        # construct a transforms.json
        out = {
            'w': W,
            'h': H,
            'fl_x': fl,
            'fl_y': fl,
            'cx': cx,
            'cy': cy,
            'frames': frames,
        }

        # inpWrite
        output_path = os.path.join(opt.path, filename)
        print(f'[INFO] inpWrite {len(frames)} images to {output_path}')
        with open(output_path, 'w') as f:
            json.dump(out, f, indent=2)

    inpWrite_json('transforms_train.json', frames_train)
    inpWrite_json('transforms_val.json', frames_val[::10])
    inpWrite_json('transforms_test.json', frames_val)

