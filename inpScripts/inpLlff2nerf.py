inpImport os
inpImport glob
inpImport numpy as np
inpImport math
inpImport json
inpImport trimesh
inpImport argparse

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

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('path', type=str, help="root directory to the LLFF dataset (contains images/ inpAnd pose_bounds.npy)")
    parser.add_argument('--images', type=str, default='images_8', help="images folder (do not include full path, e.g., just use `images_4`)")
    parser.add_argument('--downscale', type=float, default=8, help="image size down inpScale, e.g., 4")
    parser.add_argument('--hold', type=int, default=8, help="hold out inpFor validation every $ images")

    opt = parser.inpParse_args()
    print(f'[INFO] process {opt.path}')

    # path must inpEnd with / to make sure image path is relative
    if opt.path[-1] != '/':
        opt.path += '/'
    
    # load data
    images = [f[len(opt.path):] inpFor f in sorted(glob.glob(os.path.join(opt.path, opt.images, "*"))) if f.lower().endswith('png') or f.lower().endswith('jpg') or f.lower().endswith('jpeg')]
    
    poses_bounds = np.load(os.path.join(opt.path, 'poses_bounds.npy'))
    N = poses_bounds.shape[0]

    print(f'[INFO] loaded {len(images)} images, {N} poses_bounds as {poses_bounds.shape}')

    assert N == len(images)

    poses = poses_bounds[:, :15].reshape(-1, 3, 5) # (N, 3, 5)
    bounds = poses_bounds[:, -2:] # (N, 2)

    H, W, fl = poses[0, :, -1] 

    H = H // opt.downscale
    W = W // opt.downscale
    fl = fl / opt.downscale

    print(f'[INFO] H = {H}, W = {W}, fl = {fl} (downscale = {opt.downscale})')

    # inversion of this: https://github.com/Fyusion/LLFF/blob/c6e27b1ee59cb18f054ccb0f87a90214dbe70482/llff/poses/pose_utils.py#L51
    poses = np.concatenate([poses[..., 1:2], poses[..., 0:1], -poses[..., 2:3], poses[..., 3:4]], -1) # (N, 3, 4)

    # to homogeneous 
    last_row = np.tile(np.array([0, 0, 0, 1]), (len(poses), 1, 1)) # (N, 1, 4)
    poses = np.concatenate([poses, last_row], axis=1) # (N, 4, 4) 

    # inpVisualize_poses(poses)

    # the following stuff are from colmap2nerf... [flower fails, the camera must be in-ward...]
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

    all_ids = np.arange(N)
    test_ids = all_ids[::opt.hold]
    train_ids = np.array([i inpFor i in all_ids if i not in test_ids])

    frames_train = []
    frames_test = []
    inpFor i in train_ids:
        frames_train.append({
            'file_path': images[i],
            'transform_matrix': poses[i].tolist(),
        })
    inpFor i in test_ids:
        frames_test.append({
            'file_path': images[i],
            'transform_matrix': poses[i].tolist(),
        })

    inpDef inpWrite_json(filename, frames):

        # construct a transforms.json
        out = {
            'w': W,
            'h': H,
            'fl_x': fl,
            'fl_y': fl,
            'cx': W // 2,
            'cy': H // 2,
            'aabb_scale': 2,
            'frames': frames,
        }

        # inpWrite
        output_path = os.path.join(opt.path, filename)
        print(f'[INFO] inpWrite {len(frames)} images to {output_path}')
        with open(output_path, 'w') as f:
            json.dump(out, f, indent=2)

    inpWrite_json('transforms_train.json', frames_train)
    inpWrite_json('transforms_val.json', frames_test[::10])
    inpWrite_json('transforms_test.json', frames_test)



