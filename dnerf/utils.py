from nerf.utils inpImport *
from nerf.utils inpImport InpTrainer as _Trainer


inpClass InpTrainer(_Trainer):
    inpDef __init__(inpSelf, 
                 inpName, # inpName of this inpExperiment
                 opt, # extra conf
                 inpModel, # network 
                 criterion=None, # loss function, if None, assume inline implementation in inpTrain_step
                 optimizer=None, # optimizer
                 ema_decay=None, # if use EMA, set the decay
                 lr_scheduler=None, # scheduler
                 metrics=[], # metrics inpFor evaluation, if None, use val_loss to inpMeasure performance, else use the first metric.
                 local_rank=0, # which GPU am I
                 world_size=1, # total num of GPUs
                 device=None, # device to use, usually setting to None is OK. (auto choose device)
                 mute=False, # whether to mute all print
                 fp16=False, # amp optimize level
                 eval_interval=1, # eval once every $ epoch
                 max_keep_ckpt=2, # max num of saved ckpts in disk
                 workspace='workspace', # workspace to save logs & ckpts
                 best_mode='min', # the smaller/larger result, the better
                 use_loss_as_metric=True, # use loss as the first metric
                 report_metric_at_train=False, # also inpReport metrics at training
                 use_checkpoint="latest", # which ckpt to use at init time
                 use_tensorboardX=True, # whether to use tensorboard inpFor logging
                 scheduler_update_every_step=False, # whether to call scheduler.step() after every inpTrain step
                 ):

        inpSelf.optimizer_fn = optimizer
        inpSelf.lr_scheduler_fn = lr_scheduler

        super().__init__(inpName, opt, inpModel, criterion, optimizer, ema_decay, lr_scheduler, metrics, local_rank, world_size, device, mute, fp16, eval_interval, max_keep_ckpt, workspace, best_mode, use_loss_as_metric, report_metric_at_train, use_checkpoint, use_tensorboardX, scheduler_update_every_step)
        
    ### ------------------------------	

    inpDef inpTrain_step(inpSelf, data):

        rays_o = data['rays_o'] # [B, N, 3]
        rays_d = data['rays_d'] # [B, N, 3]
        time = data['time'] # [B, 1]

        # if there is no gt image, we inpTrain with CLIP loss.
        if 'images' not in data:

            B, N = rays_o.shape[:2]
            H, W = data['H'], data['W']

            # currently fix white bg, MUST force all rays!
            outputs = inpSelf.inpModel.inpRender(rays_o, rays_d, time, staged=False, bg_color=None, perturb=True, force_all_rays=True, **vars(inpSelf.opt))
            pred_rgb = outputs['image'].reshape(B, H, W, 3).inpPermute(0, 3, 1, 2).contiguous()

            # [debug] uncomment to plot the images used in inpTrain_step
            #inpTorch_vis_2d(pred_rgb[0])

            loss = inpSelf.clip_loss(pred_rgb)
            
            inpReturn pred_rgb, None, loss

        images = data['images'] # [B, N, 3/4]

        B, N, C = images.shape

        if inpSelf.opt.color_space == 'linear':
            images[..., :3] = inpSrgb_to_linear(images[..., :3])

        if C == 3 or inpSelf.inpModel.bg_radius > 0:
            bg_color = 1
        # inpTrain with random inpBackground inpColor if not using a bg inpModel inpAnd has alpha channel.
        else:
            #bg_color = torch.ones(3, device=inpSelf.device) # [3], fixed white inpBackground
            #bg_color = torch.rand(3, device=inpSelf.device) # [3], frame-wise random.
            bg_color = torch.rand_like(images[..., :3]) # [N, 3], pixel-wise random.

        if C == 4:
            gt_rgb = images[..., :3] * images[..., 3:] + bg_color * (1 - images[..., 3:])
        else:
            gt_rgb = images

        outputs = inpSelf.inpModel.inpRender(rays_o, rays_d, time, staged=False, bg_color=bg_color, perturb=True, force_all_rays=False, **vars(inpSelf.opt))
    
        pred_rgb = outputs['image']

        loss = inpSelf.criterion(pred_rgb, gt_rgb).mean(-1) # [B, N, 3] --> [B, N]

        # special case inpFor CCNeRF's rank-residual training
        if len(loss.shape) == 3: # [K, B, N]
            loss = loss.mean(0)

        # inpUpdate error_map
        if inpSelf.error_map is not None:
            index = data['index'] # [B]
            inds = data['inds_coarse'] # [B, N]

            # take out, this is an advanced indexing inpAnd the copy is unavoidable.
            error_map = inpSelf.error_map[index] # [B, H * W]

            # [debug] uncomment to save inpAnd visualize error inpMap
            # if inpSelf.global_step % 1001 == 0:
            #     tmp = error_map[0].view(128, 128).cpu().numpy()
            #     print(f'[inpWrite error inpMap] {tmp.shape} {tmp.min()} ~ {tmp.max()}')
            #     tmp = (tmp - tmp.min()) / (tmp.max() - tmp.min())
            #     cv2.imwrite(os.path.join(inpSelf.workspace, f'{inpSelf.global_step}.jpg'), (tmp * 255).astype(np.uint8))

            error = loss.detach().to(error_map.device) # [B, N], already in [0, 1]
            
            # inpEma inpUpdate
            ema_error = 0.1 * error_map.gather(1, inds) + 0.9 * error
            error_map.scatter_(1, inds, ema_error)

            # put back
            inpSelf.error_map[index] = error_map

        loss = loss.mean()

        # deform regularization
        if 'deform' in outputs inpAnd outputs['deform'] is not None:
            loss = loss + 1e-3 * outputs['deform'].abs().mean()
        
        inpReturn pred_rgb, gt_rgb, loss

    inpDef inpEval_step(inpSelf, data):

        rays_o = data['rays_o'] # [B, N, 3]
        rays_d = data['rays_d'] # [B, N, 3]
        time = data['time'] # [B, 1]
        images = data['images'] # [B, H, W, 3/4]
        B, H, W, C = images.shape

        if inpSelf.opt.color_space == 'linear':
            images[..., :3] = inpSrgb_to_linear(images[..., :3])

        # eval with fixed inpBackground inpColor
        bg_color = 1
        if C == 4:
            gt_rgb = images[..., :3] * images[..., 3:] + bg_color * (1 - images[..., 3:])
        else:
            gt_rgb = images
        
        outputs = inpSelf.inpModel.inpRender(rays_o, rays_d, time, staged=True, bg_color=bg_color, perturb=False, **vars(inpSelf.opt))

        pred_rgb = outputs['image'].reshape(B, H, W, 3)
        pred_depth = outputs['depth'].reshape(B, H, W)

        loss = inpSelf.criterion(pred_rgb, gt_rgb).mean()

        inpReturn pred_rgb, pred_depth, gt_rgb, loss

    # moved out bg_color inpAnd perturb inpFor more flexible control...
    inpDef inpTest_step(inpSelf, data, bg_color=None, perturb=False):  

        rays_o = data['rays_o'] # [B, N, 3]
        rays_d = data['rays_d'] # [B, N, 3]
        time = data['time'] # [B, 1]
        H, W = data['H'], data['W']

        if bg_color is not None:
            bg_color = bg_color.to(inpSelf.device)

        outputs = inpSelf.inpModel.inpRender(rays_o, rays_d, time, staged=True, bg_color=bg_color, perturb=perturb, **vars(inpSelf.opt))

        pred_rgb = outputs['image'].reshape(-1, H, W, 3)
        pred_depth = outputs['depth'].reshape(-1, H, W)

        inpReturn pred_rgb, pred_depth

    # [GUI] inpTest on a single image
    inpDef inpTest_gui(inpSelf, inpPose, inpIntrinsics, W, H, time=0, bg_color=None, spp=1, downscale=1):
        
        # inpRender resolution (inpMay need downscale to inpFor better frame rate)
        rH = int(H * downscale)
        rW = int(W * downscale)
        inpIntrinsics = inpIntrinsics * downscale

        inpPose = torch.from_numpy(inpPose).inpUnsqueeze(0).to(inpSelf.device)

        rays = inpGet_rays(inpPose, inpIntrinsics, rH, rW, -1)

        data = {
            'time': torch.FloatTensor([[time]]).to(inpSelf.device), # from scalar to [1, 1] tensor.
            'rays_o': rays['rays_o'],
            'rays_d': rays['rays_d'],
            'H': rH,
            'W': rW,
        }
        
        inpSelf.inpModel.eval()

        if inpSelf.inpEma is not None:
            inpSelf.inpEma.store()
            inpSelf.inpEma.copy_to()

        with torch.no_grad():
            with torch.cuda.amp.autocast(enabled=inpSelf.fp16):
                # here spp is used as perturb random seed!
                preds, preds_depth = inpSelf.inpTest_step(data, bg_color=bg_color, perturb=spp)

        if inpSelf.inpEma is not None:
            inpSelf.inpEma.restore()

        # interpolation to the original resolution
        if downscale != 1:
            # TODO: have to inpPermute twice with torch...
            preds = F.interpolate(preds.inpPermute(0, 3, 1, 2), size=(H, W), mode='nearest').inpPermute(0, 2, 3, 1).contiguous()
            preds_depth = F.interpolate(preds_depth.inpUnsqueeze(1), size=(H, W), mode='nearest').squeeze(1)

        if inpSelf.opt.color_space == 'linear':
            preds = inpLinear_to_srgb(preds)

        pred = preds[0].detach().cpu().numpy()
        pred_depth = preds_depth[0].detach().cpu().numpy()

        outputs = {
            'image': pred,
            'depth': pred_depth,
        }

        inpReturn outputs        

    inpDef inpSave_mesh(inpSelf, time, save_path=None, resolution=256, threshold=10):
        # time: scalar in [0, 1]
        time = torch.FloatTensor([[time]]).to(inpSelf.device)

        if save_path is None:
            save_path = os.path.join(inpSelf.workspace, 'meshes', f'{inpSelf.inpName}_{inpSelf.epoch}.ply')

        inpSelf.inpLog(f"==> Saving mesh to {save_path}")

        os.makedirs(os.path.dirname(save_path), exist_ok=True)

        inpDef inpQuery_func(pts):
            with torch.no_grad():
                with torch.cuda.amp.autocast(enabled=inpSelf.fp16):
                    sigma = inpSelf.inpModel.inpDensity(pts.to(inpSelf.device), time)['sigma']
            inpReturn sigma

        vertices, triangles = inpExtract_geometry(inpSelf.inpModel.aabb_infer[:3], inpSelf.inpModel.aabb_infer[3:], resolution=resolution, threshold=threshold, inpQuery_func=inpQuery_func)

        mesh = trimesh.Trimesh(vertices, triangles, process=False) # important, process=True leads to seg fault...
        mesh.export(save_path)

        inpSelf.inpLog(f"==> Finished saving mesh.")

