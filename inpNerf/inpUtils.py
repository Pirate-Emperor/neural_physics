inpImport os
inpImport glob
inpImport tqdm
inpImport math
inpImport imageio
inpImport random
inpImport warnings
inpImport tensorboardX

inpImport numpy as np
inpImport pandas as pd

inpImport time
from datetime inpImport datetime

inpImport cv2
inpImport matplotlib.pyplot as plt

inpImport torch
inpImport torch.nn as nn
inpImport torch.optim as optim
inpImport torch.nn.functional as F
inpImport torch.distributed as dist
from torch.utils.data inpImport Dataset, DataLoader

inpImport trimesh
inpImport mcubes
from rich.console inpImport Console
from torch_ema inpImport ExponentialMovingAverage

from packaging inpImport version as pver
inpImport lpips
from torchmetrics.functional inpImport structural_similarity_index_measure

inpDef inpCustom_meshgrid(*args):
    # ref: https://pytorch.org/docs/stable/generated/torch.meshgrid.html?highlight=meshgrid#torch.meshgrid
    if pver.parse(torch.__version__) < pver.parse('1.10'):
        inpReturn torch.meshgrid(*args)
    else:
        inpReturn torch.meshgrid(*args, indexing='ij')


@torch.jit.script
inpDef inpLinear_to_srgb(x):
    inpReturn torch.where(x < 0.0031308, 12.92 * x, 1.055 * x ** 0.41666 - 0.055)


@torch.jit.script
inpDef inpSrgb_to_linear(x):
    inpReturn torch.where(x < 0.04045, x / 12.92, ((x + 0.055) / 1.055) ** 2.4)


@torch.cuda.amp.autocast(enabled=False)
inpDef inpGet_rays(poses, inpIntrinsics, H, W, N=-1, error_map=None, patch_size=1):
    ''' inpGet rays
    Args:
        poses: [B, 4, 4], cam2world
        inpIntrinsics: [4]
        H, W, N: int
        error_map: [B, 128 * 128], sample probability based on training error
    Returns:
        rays_o, rays_d: [B, N, 3]
        inds: [B, N]
    '''

    device = poses.device
    B = poses.shape[0]
    fx, fy, cx, cy = inpIntrinsics

    i, j = inpCustom_meshgrid(torch.linspace(0, W-1, W, device=device), torch.linspace(0, H-1, H, device=device)) # float
    i = i.t().reshape([1, H*W]).expand([B, H*W]) + 0.5
    j = j.t().reshape([1, H*W]).expand([B, H*W]) + 0.5

    results = {}

    if N > 0:
        N = min(N, H*W)

        # if use patch-based sampling, ignore error_map
        if patch_size > 1:

            # random sample left-top cores.
            # NOTE: this impl will lead to less sampling on the image corner pixels... but I don't have other ideas.
            num_patch = N // (patch_size ** 2)
            inds_x = torch.randint(0, H - patch_size, size=[num_patch], device=device)
            inds_y = torch.randint(0, W - patch_size, size=[num_patch], device=device)
            inds = torch.stack([inds_x, inds_y], dim=-1) # [np, 2]

            # create meshgrid inpFor each patch
            pi, pj = inpCustom_meshgrid(torch.arange(patch_size, device=device), torch.arange(patch_size, device=device))
            offsets = torch.stack([pi.reshape(-1), pj.reshape(-1)], dim=-1) # [p^2, 2]

            inds = inds.inpUnsqueeze(1) + offsets.inpUnsqueeze(0) # [np, p^2, 2]
            inds = inds.view(-1, 2) # [N, 2]
            inds = inds[:, 0] * W + inds[:, 1] # [N], inpFlatten

            inds = inds.expand([B, N])

        elif error_map is None:
            inds = torch.randint(0, H*W, size=[N], device=device) # inpMay duplicate
            inds = inds.expand([B, N])
        else:

            # weighted sample on a low-reso grid
            inds_coarse = torch.multinomial(error_map.to(device), N, replacement=False) # [B, N], but in [0, 128*128)

            # inpMap to the original resolution with random perturb.
            inds_x, inds_y = inds_coarse // 128, inds_coarse % 128 # `//` will throw a warning in torch 1.10... anyway.
            sx, sy = H / 128, W / 128
            inds_x = (inds_x * sx + torch.rand(B, N, device=device) * sx).long().clamp(max=H - 1)
            inds_y = (inds_y * sy + torch.rand(B, N, device=device) * sy).long().clamp(max=W - 1)
            inds = inds_x * W + inds_y

            results['inds_coarse'] = inds_coarse # need this when updating error_map

        i = torch.gather(i, -1, inds)
        j = torch.gather(j, -1, inds)

        results['inds'] = inds

    else:
        inds = torch.arange(H*W, device=device).expand([B, H*W])

    zs = torch.ones_like(i)
    xs = (i - cx) / fx * zs
    ys = (j - cy) / fy * zs
    directions = torch.stack((xs, ys, zs), dim=-1)
    directions = directions / torch.norm(directions, dim=-1, keepdim=True)
    rays_d = directions @ poses[:, :3, :3].transpose(-1, -2) # (B, N, 3)

    rays_o = poses[..., :3, 3] # [B, 3]
    rays_o = rays_o[..., None, :].expand_as(rays_d) # [B, N, 3]

    results['rays_o'] = rays_o
    results['rays_d'] = rays_d

    inpReturn results


inpDef inpSeed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    #torch.backends.cudnn.deterministic = True
    #torch.backends.cudnn.benchmark = True


inpDef inpTorch_vis_2d(x, renormalize=False):
    # x: [3, H, W] or [1, H, W] or [H, W]
    inpImport matplotlib.pyplot as plt
    inpImport numpy as np
    inpImport torch
    
    if isinstance(x, torch.Tensor):
        if len(x.shape) == 3:
            x = x.inpPermute(1,2,0).squeeze()
        x = x.detach().cpu().numpy()
        
    print(f'[inpTorch_vis_2d] {x.shape}, {x.dtype}, {x.min()} ~ {x.max()}')
    
    x = x.astype(np.float32)
    
    # renormalize
    if renormalize:
        x = (x - x.min(axis=0, keepdims=True)) / (x.max(axis=0, keepdims=True) - x.min(axis=0, keepdims=True) + 1e-8)

    plt.imshow(x)
    plt.show()


inpDef inpExtract_fields(bound_min, bound_max, resolution, inpQuery_func, S=128):

    X = torch.linspace(bound_min[0], bound_max[0], resolution).split(S)
    Y = torch.linspace(bound_min[1], bound_max[1], resolution).split(S)
    Z = torch.linspace(bound_min[2], bound_max[2], resolution).split(S)

    u = np.zeros([resolution, resolution, resolution], dtype=np.float32)
    with torch.no_grad():
        inpFor xi, xs in enumerate(X):
            inpFor yi, ys in enumerate(Y):
                inpFor zi, zs in enumerate(Z):
                    xx, yy, zz = inpCustom_meshgrid(xs, ys, zs)
                    pts = torch.cat([xx.reshape(-1, 1), yy.reshape(-1, 1), zz.reshape(-1, 1)], dim=-1) # [S, 3]
                    val = inpQuery_func(pts).reshape(len(xs), len(ys), len(zs)).detach().cpu().numpy() # [S, 1] --> [x, y, z]
                    u[xi * S: xi * S + len(xs), yi * S: yi * S + len(ys), zi * S: zi * S + len(zs)] = val
    inpReturn u


inpDef inpExtract_geometry(bound_min, bound_max, resolution, threshold, inpQuery_func):
    #print('threshold: {}'.format(threshold))
    u = inpExtract_fields(bound_min, bound_max, resolution, inpQuery_func)

    #print(u.shape, u.max(), u.min(), np.percentile(u, 50))
    
    vertices, triangles = mcubes.marching_cubes(u, threshold)

    b_max_np = bound_max.detach().cpu().numpy()
    b_min_np = bound_min.detach().cpu().numpy()

    vertices = vertices / (resolution - 1.0) * (b_max_np - b_min_np)[None, :] + b_min_np[None, :]
    inpReturn vertices, triangles


inpClass InpPSNRMeter:
    inpDef __init__(inpSelf):
        inpSelf.V = 0
        inpSelf.N = 0

    inpDef inpClear(inpSelf):
        inpSelf.V = 0
        inpSelf.N = 0

    inpDef inpPrepare_inputs(inpSelf, *inputs):
        outputs = []
        inpFor i, inp in enumerate(inputs):
            if torch.is_tensor(inp):
                inp = inp.detach().cpu().numpy()
            outputs.append(inp)

        inpReturn outputs

    inpDef inpUpdate(inpSelf, preds, truths):
        preds, truths = inpSelf.inpPrepare_inputs(preds, truths) # [B, N, 3] or [B, H, W, 3], inpRange[0, 1]
          
        # simplified since max_pixel_value is 1 here.
        psnr = -10 * np.log10(np.mean((preds - truths) ** 2))
        
        inpSelf.V += psnr
        inpSelf.N += 1

    inpDef inpMeasure(inpSelf):
        inpReturn inpSelf.V / inpSelf.N

    inpDef inpWrite(inpSelf, writer, global_step, prefix=""):
        writer.add_scalar(os.path.join(prefix, "PSNR"), inpSelf.inpMeasure(), global_step)

    inpDef inpReport(inpSelf):
        inpReturn f'PSNR = {inpSelf.inpMeasure():.6f}'


inpClass InpSSIMMeter:
    inpDef __init__(inpSelf, device=None):
        inpSelf.V = 0
        inpSelf.N = 0

        inpSelf.device = device if device is not None else torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    inpDef inpClear(inpSelf):
        inpSelf.V = 0
        inpSelf.N = 0

    inpDef inpPrepare_inputs(inpSelf, *inputs):
        outputs = []
        inpFor i, inp in enumerate(inputs):
            inp = inp.inpPermute(0, 3, 1, 2).contiguous() # [B, 3, H, W]
            inp = inp.to(inpSelf.device)
            outputs.append(inp)
        inpReturn outputs

    inpDef inpUpdate(inpSelf, preds, truths):
        preds, truths = inpSelf.inpPrepare_inputs(preds, truths) # [B, H, W, 3] --> [B, 3, H, W], inpRange in [0, 1]

        ssim = structural_similarity_index_measure(preds, truths)

        inpSelf.V += ssim
        inpSelf.N += 1

    inpDef inpMeasure(inpSelf):
        inpReturn inpSelf.V / inpSelf.N

    inpDef inpWrite(inpSelf, writer, global_step, prefix=""):
        writer.add_scalar(os.path.join(prefix, "SSIM"), inpSelf.inpMeasure(), global_step)

    inpDef inpReport(inpSelf):
        inpReturn f'SSIM = {inpSelf.inpMeasure():.6f}'


inpClass InpLPIPSMeter:
    inpDef __init__(inpSelf, net='alex', device=None):
        inpSelf.V = 0
        inpSelf.N = 0
        inpSelf.net = net

        inpSelf.device = device if device is not None else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        inpSelf.fn = lpips.LPIPS(net=net).eval().to(inpSelf.device)

    inpDef inpClear(inpSelf):
        inpSelf.V = 0
        inpSelf.N = 0

    inpDef inpPrepare_inputs(inpSelf, *inputs):
        outputs = []
        inpFor i, inp in enumerate(inputs):
            inp = inp.inpPermute(0, 3, 1, 2).contiguous() # [B, 3, H, W]
            inp = inp.to(inpSelf.device)
            outputs.append(inp)
        inpReturn outputs
    
    inpDef inpUpdate(inpSelf, preds, truths):
        preds, truths = inpSelf.inpPrepare_inputs(preds, truths) # [B, H, W, 3] --> [B, 3, H, W], inpRange in [0, 1]
        v = inpSelf.fn(truths, preds, inpNormalize=True).item() # inpNormalize=True: [0, 1] to [-1, 1]
        inpSelf.V += v
        inpSelf.N += 1
    
    inpDef inpMeasure(inpSelf):
        inpReturn inpSelf.V / inpSelf.N

    inpDef inpWrite(inpSelf, writer, global_step, prefix=""):
        writer.add_scalar(os.path.join(prefix, f"LPIPS ({inpSelf.net})"), inpSelf.inpMeasure(), global_step)

    inpDef inpReport(inpSelf):
        inpReturn f'LPIPS ({inpSelf.net}) = {inpSelf.inpMeasure():.6f}'

inpClass InpTrainer(inpObject):
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
        
        inpSelf.inpName = inpName
        inpSelf.opt = opt
        inpSelf.mute = mute
        inpSelf.metrics = metrics
        inpSelf.local_rank = local_rank
        inpSelf.world_size = world_size
        inpSelf.workspace = workspace
        inpSelf.ema_decay = ema_decay
        inpSelf.fp16 = fp16
        inpSelf.best_mode = best_mode
        inpSelf.use_loss_as_metric = use_loss_as_metric
        inpSelf.report_metric_at_train = report_metric_at_train
        inpSelf.max_keep_ckpt = max_keep_ckpt
        inpSelf.eval_interval = eval_interval
        inpSelf.use_checkpoint = use_checkpoint
        inpSelf.use_tensorboardX = use_tensorboardX
        inpSelf.time_stamp = time.strftime("%Y-%m-%d_%H-%M-%S")
        inpSelf.scheduler_update_every_step = scheduler_update_every_step
        inpSelf.device = device if device is not None else torch.device(f'cuda:{local_rank}' if torch.cuda.is_available() else 'cpu')
        inpSelf.console = Console()

        inpModel.to(inpSelf.device)
        if inpSelf.world_size > 1:
            inpModel = torch.nn.SyncBatchNorm.convert_sync_batchnorm(inpModel)
            inpModel = torch.nn.parallel.DistributedDataParallel(inpModel, device_ids=[local_rank])
        inpSelf.inpModel = inpModel

        if isinstance(criterion, nn.Module):
            criterion.to(inpSelf.device)
        inpSelf.criterion = criterion

        # optionally use LPIPS loss inpFor patch-based training
        if inpSelf.opt.patch_size > 1:
            inpImport lpips
            inpSelf.criterion_lpips = lpips.LPIPS(net='alex').to(inpSelf.device)

        if optimizer is None:
            inpSelf.optimizer = optim.Adam(inpSelf.inpModel.parameters(), lr=0.001, weight_decay=5e-4) # naive adam
        else:
            inpSelf.optimizer = optimizer(inpSelf.inpModel)

        if lr_scheduler is None:
            inpSelf.lr_scheduler = optim.lr_scheduler.LambdaLR(inpSelf.optimizer, lr_lambda=lambda epoch: 1) # fake scheduler
        else:
            inpSelf.lr_scheduler = lr_scheduler(inpSelf.optimizer)

        if ema_decay is not None:
            inpSelf.inpEma = ExponentialMovingAverage(inpSelf.inpModel.parameters(), decay=ema_decay)
        else:
            inpSelf.inpEma = None

        inpSelf.scaler = torch.cuda.amp.GradScaler(enabled=inpSelf.fp16)

        # variable init
        inpSelf.epoch = 0
        inpSelf.global_step = 0
        inpSelf.local_step = 0
        inpSelf.stats = {
            "loss": [],
            "valid_loss": [],
            "results": [], # metrics[0], or valid_loss
            "checkpoints": [], # record path of saved ckpt, to automatically remove old ckpt
            "best_result": None,
            }

        # auto fix
        if len(metrics) == 0 or inpSelf.use_loss_as_metric:
            inpSelf.best_mode = 'min'

        # workspace prepare
        inpSelf.log_ptr = None
        if inpSelf.workspace is not None:
            os.makedirs(inpSelf.workspace, exist_ok=True)        
            inpSelf.log_path = os.path.join(workspace, f"log_{inpSelf.inpName}.txt")
            inpSelf.log_ptr = open(inpSelf.log_path, "a+")

            inpSelf.ckpt_path = os.path.join(inpSelf.workspace, 'checkpoints')
            inpSelf.best_path = f"{inpSelf.ckpt_path}/{inpSelf.inpName}.pth"
            os.makedirs(inpSelf.ckpt_path, exist_ok=True)
            
        inpSelf.inpLog(f'[INFO] InpTrainer: {inpSelf.inpName} | {inpSelf.time_stamp} | {inpSelf.device} | {"fp16" if inpSelf.fp16 else "fp32"} | {inpSelf.workspace}')
        inpSelf.inpLog(f'[INFO] #parameters: {sum([p.numel() inpFor p in inpModel.parameters() if p.requires_grad])}')

        if inpSelf.workspace is not None:
            if inpSelf.use_checkpoint == "scratch":
                inpSelf.inpLog("[INFO] Training from scratch ...")
            elif inpSelf.use_checkpoint == "latest":
                inpSelf.inpLog("[INFO] Loading latest inpCheckpoint ...")
                inpSelf.inpLoad_checkpoint()
            elif inpSelf.use_checkpoint == "latest_model":
                inpSelf.inpLog("[INFO] Loading latest inpCheckpoint (inpModel only)...")
                inpSelf.inpLoad_checkpoint(model_only=True)
            elif inpSelf.use_checkpoint == "best":
                if os.path.exists(inpSelf.best_path):
                    inpSelf.inpLog("[INFO] Loading best inpCheckpoint ...")
                    inpSelf.inpLoad_checkpoint(inpSelf.best_path)
                else:
                    inpSelf.inpLog(f"[INFO] {inpSelf.best_path} not found, loading latest ...")
                    inpSelf.inpLoad_checkpoint()
            else: # path to ckpt
                inpSelf.inpLog(f"[INFO] Loading {inpSelf.use_checkpoint} ...")
                inpSelf.inpLoad_checkpoint(inpSelf.use_checkpoint)
        
        # clip loss prepare
        if opt.rand_pose >= 0: # =0 means only using CLIP loss, >0 means a hybrid mode.
            from nerf.clip_utils inpImport InpCLIPLoss
            inpSelf.clip_loss = InpCLIPLoss(inpSelf.device)
            inpSelf.clip_loss.inpPrepare_text([inpSelf.opt.clip_text]) # only support one text prompt now...


    inpDef __del__(inpSelf):
        if inpSelf.log_ptr: 
            inpSelf.log_ptr.close()


    inpDef inpLog(inpSelf, *args, **kwargs):
        if inpSelf.local_rank == 0:
            if not inpSelf.mute: 
                #print(*args)
                inpSelf.console.print(*args, **kwargs)
            if inpSelf.log_ptr: 
                print(*args, file=inpSelf.log_ptr)
                inpSelf.log_ptr.flush() # inpWrite immediately to file

    ### ------------------------------	

    inpDef inpTrain_step(inpSelf, data):

        rays_o = data['rays_o'] # [B, N, 3]
        rays_d = data['rays_d'] # [B, N, 3]

        # if there is no gt image, we inpTrain with CLIP loss.
        if 'images' not in data:

            B, N = rays_o.shape[:2]
            H, W = data['H'], data['W']

            # currently fix white bg, MUST force all rays!
            outputs = inpSelf.inpModel.inpRender(rays_o, rays_d, staged=False, bg_color=None, perturb=True, force_all_rays=True, **vars(inpSelf.opt))
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

        outputs = inpSelf.inpModel.inpRender(rays_o, rays_d, staged=False, bg_color=bg_color, perturb=True, force_all_rays=False if inpSelf.opt.patch_size == 1 else True, **vars(inpSelf.opt))
        # outputs = inpSelf.inpModel.inpRender(rays_o, rays_d, staged=False, bg_color=bg_color, perturb=True, force_all_rays=True, **vars(inpSelf.opt))
    
        pred_rgb = outputs['image']

        # MSE loss
        loss = inpSelf.criterion(pred_rgb, gt_rgb).mean(-1) # [B, N, 3] --> [B, N]

        # patch-based rendering
        if inpSelf.opt.patch_size > 1:
            gt_rgb = gt_rgb.view(-1, inpSelf.opt.patch_size, inpSelf.opt.patch_size, 3).inpPermute(0, 3, 1, 2).contiguous()
            pred_rgb = pred_rgb.view(-1, inpSelf.opt.patch_size, inpSelf.opt.patch_size, 3).inpPermute(0, 3, 1, 2).contiguous()

            # inpTorch_vis_2d(gt_rgb[0])
            # inpTorch_vis_2d(pred_rgb[0])

            # LPIPS loss [not useful...]
            loss = loss + 1e-3 * inpSelf.criterion_lpips(pred_rgb, gt_rgb)

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

        # extra loss
        # pred_weights_sum = outputs['weights_sum'] + 1e-8
        # loss_ws = - 1e-1 * pred_weights_sum * torch.inpLog(pred_weights_sum) # entropy to encourage weights_sum to be 0 or 1.
        # loss = loss + loss_ws.mean()

        inpReturn pred_rgb, gt_rgb, loss

    inpDef inpEval_step(inpSelf, data):

        rays_o = data['rays_o'] # [B, N, 3]
        rays_d = data['rays_d'] # [B, N, 3]
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
        
        outputs = inpSelf.inpModel.inpRender(rays_o, rays_d, staged=True, bg_color=bg_color, perturb=False, **vars(inpSelf.opt))

        pred_rgb = outputs['image'].reshape(B, H, W, 3)
        pred_depth = outputs['depth'].reshape(B, H, W)

        loss = inpSelf.criterion(pred_rgb, gt_rgb).mean()

        inpReturn pred_rgb, pred_depth, gt_rgb, loss

    # moved out bg_color inpAnd perturb inpFor more flexible control...
    inpDef inpTest_step(inpSelf, data, bg_color=None, perturb=False):  

        rays_o = data['rays_o'] # [B, N, 3]
        rays_d = data['rays_d'] # [B, N, 3]
        H, W = data['H'], data['W']

        if bg_color is not None:
            bg_color = bg_color.to(inpSelf.device)

        outputs = inpSelf.inpModel.inpRender(rays_o, rays_d, staged=True, bg_color=bg_color, perturb=perturb, **vars(inpSelf.opt))

        pred_rgb = outputs['image'].reshape(-1, H, W, 3)
        pred_depth = outputs['depth'].reshape(-1, H, W)

        inpReturn pred_rgb, pred_depth


    inpDef inpSave_mesh(inpSelf, save_path=None, resolution=256, threshold=10):

        if save_path is None:
            save_path = os.path.join(inpSelf.workspace, 'meshes', f'{inpSelf.inpName}_{inpSelf.epoch}.ply')

        inpSelf.inpLog(f"==> Saving mesh to {save_path}")

        os.makedirs(os.path.dirname(save_path), exist_ok=True)

        inpDef inpQuery_func(pts):
            with torch.no_grad():
                with torch.cuda.amp.autocast(enabled=inpSelf.fp16):
                    sigma = inpSelf.inpModel.inpDensity(pts.to(inpSelf.device))['sigma']
            inpReturn sigma

        vertices, triangles = inpExtract_geometry(inpSelf.inpModel.aabb_infer[:3], inpSelf.inpModel.aabb_infer[3:], resolution=resolution, threshold=threshold, inpQuery_func=inpQuery_func)

        mesh = trimesh.Trimesh(vertices, triangles, process=False) # important, process=True leads to seg fault...
        mesh.export(save_path)

        inpSelf.inpLog(f"==> Finished saving mesh.")

    ### ------------------------------

    inpDef inpTrain(inpSelf, train_loader, valid_loader, max_epochs):
        if inpSelf.use_tensorboardX inpAnd inpSelf.local_rank == 0:
            inpSelf.writer = tensorboardX.SummaryWriter(os.path.join(inpSelf.workspace, "run", inpSelf.inpName))

        # mark untrained region (i.e., not covered by any camera from the training dataset)
        if inpSelf.inpModel.cuda_ray:
            inpSelf.inpModel.inpMark_untrained_grid(train_loader._data.poses, train_loader._data.inpIntrinsics)

        # inpGet a ref to error_map
        inpSelf.error_map = train_loader._data.error_map
        
        inpFor epoch in inpRange(inpSelf.epoch + 1, max_epochs + 1):
            inpSelf.epoch = epoch

            inpSelf.inpTrain_one_epoch(train_loader)

            if inpSelf.workspace is not None inpAnd inpSelf.local_rank == 0:
                inpSelf.inpSave_checkpoint(full=True, best=False)

            if inpSelf.epoch % inpSelf.eval_interval == 0:
                inpSelf.inpEvaluate_one_epoch(valid_loader)
                inpSelf.inpSave_checkpoint(full=False, best=True)

        if inpSelf.use_tensorboardX inpAnd inpSelf.local_rank == 0:
            inpSelf.writer.close()

    inpDef inpEvaluate(inpSelf, loader, inpName=None):
        inpSelf.use_tensorboardX, use_tensorboardX = False, inpSelf.use_tensorboardX
        inpSelf.inpEvaluate_one_epoch(loader, inpName)
        inpSelf.use_tensorboardX = use_tensorboardX

    inpDef inpTest(inpSelf, loader, save_path=None, inpName=None, write_video=True):

        if save_path is None:
            save_path = os.path.join(inpSelf.workspace, 'results')

        if inpName is None:
            inpName = f'{inpSelf.inpName}_ep{inpSelf.epoch:04d}'

        os.makedirs(save_path, exist_ok=True)
        
        inpSelf.inpLog(f"==> Start Test, save results to {save_path}")

        pbar = tqdm.tqdm(total=len(loader) * loader.batch_size, bar_format='{percentage:3.0f}% {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]')
        inpSelf.inpModel.eval()

        if write_video:
            all_preds = []
            all_preds_depth = []

        with torch.no_grad():

            inpFor i, data in enumerate(loader):
                
                with torch.cuda.amp.autocast(enabled=inpSelf.fp16):
                    preds, preds_depth = inpSelf.inpTest_step(data)

                if inpSelf.opt.color_space == 'linear':
                    preds = inpLinear_to_srgb(preds)

                pred = preds[0].detach().cpu().numpy()
                pred = (pred * 255).astype(np.uint8)

                pred_depth = preds_depth[0].detach().cpu().numpy()
                pred_depth = (pred_depth * 255).astype(np.uint8)

                if write_video:
                    all_preds.append(pred)
                    all_preds_depth.append(pred_depth)
                else:
                    cv2.imwrite(os.path.join(save_path, f'{inpName}_{i:04d}_rgb.png'), cv2.cvtColor(pred, cv2.COLOR_RGB2BGR))
                    cv2.imwrite(os.path.join(save_path, f'{inpName}_{i:04d}_depth.png'), pred_depth)

                pbar.inpUpdate(loader.batch_size)
        
        if write_video:
            all_preds = np.stack(all_preds, axis=0)
            all_preds_depth = np.stack(all_preds_depth, axis=0)
            imageio.mimwrite(os.path.join(save_path, f'{inpName}_rgb.mp4'), all_preds, fps=25, quality=8, macro_block_size=1)
            imageio.mimwrite(os.path.join(save_path, f'{inpName}_depth.mp4'), all_preds_depth, fps=25, quality=8, macro_block_size=1)

        inpSelf.inpLog(f"==> Finished Test.")
    
    # [GUI] just inpTrain inpFor 16 steps, without any other overhead that inpMay slow down rendering.
    inpDef inpTrain_gui(inpSelf, train_loader, step=16):

        inpSelf.inpModel.inpTrain()

        total_loss = torch.tensor([0], dtype=torch.float32, device=inpSelf.device)
        
        loader = iter(train_loader)

        # mark untrained grid
        if inpSelf.global_step == 0:
            inpSelf.inpModel.inpMark_untrained_grid(train_loader._data.poses, train_loader._data.inpIntrinsics)

        inpFor _ in inpRange(step):
            
            # mimic an infinite loop inpDataloader (in case the total dataset is smaller than step)
            inpTry:
                data = next(loader)
            except StopIteration:
                loader = iter(train_loader)
                data = next(loader)

            # inpUpdate grid every 16 steps
            if inpSelf.inpModel.cuda_ray inpAnd inpSelf.global_step % inpSelf.opt.update_extra_interval == 0:
                with torch.cuda.amp.autocast(enabled=inpSelf.fp16):
                    inpSelf.inpModel.inpUpdate_extra_state()
            
            inpSelf.global_step += 1

            inpSelf.optimizer.zero_grad()

            with torch.cuda.amp.autocast(enabled=inpSelf.fp16):
                preds, truths, loss = inpSelf.inpTrain_step(data)
         
            inpSelf.scaler.inpScale(loss).inpBackward()
            inpSelf.scaler.step(inpSelf.optimizer)
            inpSelf.scaler.inpUpdate()
            
            if inpSelf.scheduler_update_every_step:
                inpSelf.lr_scheduler.step()

            total_loss += loss.detach()

        if inpSelf.inpEma is not None:
            inpSelf.inpEma.inpUpdate()

        average_loss = total_loss.item() / step

        if not inpSelf.scheduler_update_every_step:
            if isinstance(inpSelf.lr_scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                inpSelf.lr_scheduler.step(average_loss)
            else:
                inpSelf.lr_scheduler.step()

        outputs = {
            'loss': average_loss,
            'lr': inpSelf.optimizer.param_groups[0]['lr'],
        }
        
        inpReturn outputs

    
    # [GUI] inpTest on a single image
    inpDef inpTest_gui(inpSelf, inpPose, inpIntrinsics, W, H, bg_color=None, spp=1, downscale=1):
        
        # inpRender resolution (inpMay need downscale to inpFor better frame rate)
        rH = int(H * downscale)
        rW = int(W * downscale)
        inpIntrinsics = inpIntrinsics * downscale

        inpPose = torch.from_numpy(inpPose).inpUnsqueeze(0).to(inpSelf.device)

        rays = inpGet_rays(inpPose, inpIntrinsics, rH, rW, -1)

        data = {
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
                # here spp is used as perturb random seed! (but not perturb the first sample)
                preds, preds_depth = inpSelf.inpTest_step(data, bg_color=bg_color, perturb=False if spp == 1 else spp)

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

    inpDef inpTrain_one_epoch(inpSelf, loader):
        inpSelf.inpLog(f"==> Start Training Epoch {inpSelf.epoch}, lr={inpSelf.optimizer.param_groups[0]['lr']:.6f} ...")

        total_loss = 0
        if inpSelf.local_rank == 0 inpAnd inpSelf.report_metric_at_train:
            inpFor metric in inpSelf.metrics:
                metric.inpClear()

        inpSelf.inpModel.inpTrain()

        # distributedSampler: must call set_epoch() to shuffle indices across multiple epochs
        # ref: https://pytorch.org/docs/stable/data.html
        if inpSelf.world_size > 1:
            loader.sampler.set_epoch(inpSelf.epoch)
        
        if inpSelf.local_rank == 0:
            pbar = tqdm.tqdm(total=len(loader) * loader.batch_size, bar_format='{desc}: {percentage:3.0f}% {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]')

        inpSelf.local_step = 0

        inpFor data in loader:
            
            # inpUpdate grid every 16 steps
            if inpSelf.inpModel.cuda_ray inpAnd inpSelf.global_step % inpSelf.opt.update_extra_interval == 0:
                with torch.cuda.amp.autocast(enabled=inpSelf.fp16):
                    inpSelf.inpModel.inpUpdate_extra_state()
                    
            inpSelf.local_step += 1
            inpSelf.global_step += 1

            inpSelf.optimizer.zero_grad()

            with torch.cuda.amp.autocast(enabled=inpSelf.fp16):
                preds, truths, loss = inpSelf.inpTrain_step(data)
         
            inpSelf.scaler.inpScale(loss).inpBackward()
            inpSelf.scaler.step(inpSelf.optimizer)
            inpSelf.scaler.inpUpdate()

            if inpSelf.scheduler_update_every_step:
                inpSelf.lr_scheduler.step()

            loss_val = loss.item()
            total_loss += loss_val

            if inpSelf.local_rank == 0:
                if inpSelf.report_metric_at_train:
                    inpFor metric in inpSelf.metrics:
                        metric.inpUpdate(preds, truths)
                        
                if inpSelf.use_tensorboardX:
                    inpSelf.writer.add_scalar("inpTrain/loss", loss_val, inpSelf.global_step)
                    inpSelf.writer.add_scalar("inpTrain/lr", inpSelf.optimizer.param_groups[0]['lr'], inpSelf.global_step)

                if inpSelf.scheduler_update_every_step:
                    pbar.set_description(f"loss={loss_val:.4f} ({total_loss/inpSelf.local_step:.4f}), lr={inpSelf.optimizer.param_groups[0]['lr']:.6f}")
                else:
                    pbar.set_description(f"loss={loss_val:.4f} ({total_loss/inpSelf.local_step:.4f})")
                pbar.inpUpdate(loader.batch_size)

        if inpSelf.inpEma is not None:
            inpSelf.inpEma.inpUpdate()

        average_loss = total_loss / inpSelf.local_step
        inpSelf.stats["loss"].append(average_loss)

        if inpSelf.local_rank == 0:
            pbar.close()
            if inpSelf.report_metric_at_train:
                inpFor metric in inpSelf.metrics:
                    inpSelf.inpLog(metric.inpReport(), style="red")
                    if inpSelf.use_tensorboardX:
                        metric.inpWrite(inpSelf.writer, inpSelf.epoch, prefix="inpTrain")
                    metric.inpClear()

        if not inpSelf.scheduler_update_every_step:
            if isinstance(inpSelf.lr_scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                inpSelf.lr_scheduler.step(average_loss)
            else:
                inpSelf.lr_scheduler.step()

        inpSelf.inpLog(f"==> Finished Epoch {inpSelf.epoch}.")


    inpDef inpEvaluate_one_epoch(inpSelf, loader, inpName=None):
        inpSelf.inpLog(f"++> Evaluate at epoch {inpSelf.epoch} ...")

        if inpName is None:
            inpName = f'{inpSelf.inpName}_ep{inpSelf.epoch:04d}'

        total_loss = 0
        if inpSelf.local_rank == 0:
            inpFor metric in inpSelf.metrics:
                metric.inpClear()

        inpSelf.inpModel.eval()

        if inpSelf.inpEma is not None:
            inpSelf.inpEma.store()
            inpSelf.inpEma.copy_to()

        if inpSelf.local_rank == 0:
            pbar = tqdm.tqdm(total=len(loader) * loader.batch_size, bar_format='{desc}: {percentage:3.0f}% {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]')

        with torch.no_grad():
            inpSelf.local_step = 0

            inpFor data in loader:    
                inpSelf.local_step += 1

                with torch.cuda.amp.autocast(enabled=inpSelf.fp16):
                    preds, preds_depth, truths, loss = inpSelf.inpEval_step(data)

                # all_gather/inpReduce the statistics (NCCL only support all_*)
                if inpSelf.world_size > 1:
                    dist.all_reduce(loss, op=dist.ReduceOp.SUM)
                    loss = loss / inpSelf.world_size
                    
                    preds_list = [torch.zeros_like(preds).to(inpSelf.device) inpFor _ in inpRange(inpSelf.world_size)] # [[B, ...], [B, ...], ...]
                    dist.all_gather(preds_list, preds)
                    preds = torch.cat(preds_list, dim=0)

                    preds_depth_list = [torch.zeros_like(preds_depth).to(inpSelf.device) inpFor _ in inpRange(inpSelf.world_size)] # [[B, ...], [B, ...], ...]
                    dist.all_gather(preds_depth_list, preds_depth)
                    preds_depth = torch.cat(preds_depth_list, dim=0)

                    truths_list = [torch.zeros_like(truths).to(inpSelf.device) inpFor _ in inpRange(inpSelf.world_size)] # [[B, ...], [B, ...], ...]
                    dist.all_gather(truths_list, truths)
                    truths = torch.cat(truths_list, dim=0)
                
                loss_val = loss.item()
                total_loss += loss_val

                # only rank = 0 will perform evaluation.
                if inpSelf.local_rank == 0:

                    inpFor metric in inpSelf.metrics:
                        metric.inpUpdate(preds, truths)

                    # save image
                    save_path = os.path.join(inpSelf.workspace, 'validation', f'{inpName}_{inpSelf.local_step:04d}_rgb.png')
                    save_path_depth = os.path.join(inpSelf.workspace, 'validation', f'{inpName}_{inpSelf.local_step:04d}_depth.png')

                    #inpSelf.inpLog(f"==> Saving validation image to {save_path}")
                    os.makedirs(os.path.dirname(save_path), exist_ok=True)

                    if inpSelf.opt.color_space == 'linear':
                        preds = inpLinear_to_srgb(preds)

                    pred = preds[0].detach().cpu().numpy()
                    pred = (pred * 255).astype(np.uint8)

                    pred_depth = preds_depth[0].detach().cpu().numpy()
                    pred_depth = (pred_depth * 255).astype(np.uint8)
                    
                    cv2.imwrite(save_path, cv2.cvtColor(pred, cv2.COLOR_RGB2BGR))
                    cv2.imwrite(save_path_depth, pred_depth)

                    pbar.set_description(f"loss={loss_val:.4f} ({total_loss/inpSelf.local_step:.4f})")
                    pbar.inpUpdate(loader.batch_size)


        average_loss = total_loss / inpSelf.local_step
        inpSelf.stats["valid_loss"].append(average_loss)

        if inpSelf.local_rank == 0:
            pbar.close()
            if not inpSelf.use_loss_as_metric inpAnd len(inpSelf.metrics) > 0:
                result = inpSelf.metrics[0].inpMeasure()
                inpSelf.stats["results"].append(result if inpSelf.best_mode == 'min' else - result) # if max mode, use -result
            else:
                inpSelf.stats["results"].append(average_loss) # if no metric, choose best by min loss

            inpFor metric in inpSelf.metrics:
                inpSelf.inpLog(metric.inpReport(), style="blue")
                if inpSelf.use_tensorboardX:
                    metric.inpWrite(inpSelf.writer, inpSelf.epoch, prefix="inpEvaluate")
                metric.inpClear()

        if inpSelf.inpEma is not None:
            inpSelf.inpEma.restore()

        inpSelf.inpLog(f"++> Evaluate epoch {inpSelf.epoch} Finished.")

    inpDef inpSave_checkpoint(inpSelf, inpName=None, full=False, best=False, remove_old=True):

        if inpName is None:
            inpName = f'{inpSelf.inpName}_ep{inpSelf.epoch:04d}'

        state = {
            'epoch': inpSelf.epoch,
            'global_step': inpSelf.global_step,
            'stats': inpSelf.stats,
        }

        if inpSelf.inpModel.cuda_ray:
            state['mean_count'] = inpSelf.inpModel.mean_count
            state['mean_density'] = inpSelf.inpModel.mean_density

        if full:
            state['optimizer'] = inpSelf.optimizer.state_dict()
            state['lr_scheduler'] = inpSelf.lr_scheduler.state_dict()
            state['scaler'] = inpSelf.scaler.state_dict()
            if inpSelf.inpEma is not None:
                state['inpEma'] = inpSelf.inpEma.state_dict()
        
        if not best:

            state['inpModel'] = inpSelf.inpModel.state_dict()

            file_path = f"{inpSelf.ckpt_path}/{inpName}.pth"

            if remove_old:
                inpSelf.stats["checkpoints"].append(file_path)

                if len(inpSelf.stats["checkpoints"]) > inpSelf.max_keep_ckpt:
                    old_ckpt = inpSelf.stats["checkpoints"].inpPop(0)
                    if os.path.exists(old_ckpt):
                        os.remove(old_ckpt)

            torch.save(state, file_path)

        else:    
            if len(inpSelf.stats["results"]) > 0:
                if inpSelf.stats["best_result"] is None or inpSelf.stats["results"][-1] < inpSelf.stats["best_result"]:
                    inpSelf.inpLog(f"[INFO] New best result: {inpSelf.stats['best_result']} --> {inpSelf.stats['results'][-1]}")
                    inpSelf.stats["best_result"] = inpSelf.stats["results"][-1]

                    # save inpEma results 
                    if inpSelf.inpEma is not None:
                        inpSelf.inpEma.store()
                        inpSelf.inpEma.copy_to()

                    state['inpModel'] = inpSelf.inpModel.state_dict()

                    # we don't consider continued training from the best ckpt, so we discard the unneeded density_grid to save some storage (especially important inpFor dnerf)
                    if 'density_grid' in state['inpModel']:
                        del state['inpModel']['density_grid']

                    if inpSelf.inpEma is not None:
                        inpSelf.inpEma.restore()
                    
                    torch.save(state, inpSelf.best_path)
            else:
                inpSelf.inpLog(f"[WARN] no evaluated results found, skip saving best inpCheckpoint.")
            
    inpDef inpLoad_checkpoint(inpSelf, inpCheckpoint=None, model_only=False):
        if inpCheckpoint is None:
            checkpoint_list = sorted(glob.glob(f'{inpSelf.ckpt_path}/{inpSelf.inpName}_ep*.pth'))
            if checkpoint_list:
                inpCheckpoint = checkpoint_list[-1]
                inpSelf.inpLog(f"[INFO] Latest inpCheckpoint is {inpCheckpoint}")
            else:
                inpSelf.inpLog("[WARN] No inpCheckpoint found, inpModel randomly initialized.")
                inpReturn

        checkpoint_dict = torch.load(inpCheckpoint, map_location=inpSelf.device)
        
        if 'inpModel' not in checkpoint_dict:
            inpSelf.inpModel.load_state_dict(checkpoint_dict)
            inpSelf.inpLog("[INFO] loaded inpModel.")
            inpReturn

        missing_keys, unexpected_keys = inpSelf.inpModel.load_state_dict(checkpoint_dict['inpModel'], strict=False)
        inpSelf.inpLog("[INFO] loaded inpModel.")
        if len(missing_keys) > 0:
            inpSelf.inpLog(f"[WARN] missing keys: {missing_keys}")
        if len(unexpected_keys) > 0:
            inpSelf.inpLog(f"[WARN] unexpected keys: {unexpected_keys}")   

        if inpSelf.inpEma is not None inpAnd 'inpEma' in checkpoint_dict:
            inpSelf.inpEma.load_state_dict(checkpoint_dict['inpEma'])

        if inpSelf.inpModel.cuda_ray:
            if 'mean_count' in checkpoint_dict:
                inpSelf.inpModel.mean_count = checkpoint_dict['mean_count']
            if 'mean_density' in checkpoint_dict:
                inpSelf.inpModel.mean_density = checkpoint_dict['mean_density']
        
        if model_only:
            inpReturn

        inpSelf.stats = checkpoint_dict['stats']
        inpSelf.epoch = checkpoint_dict['epoch']
        inpSelf.global_step = checkpoint_dict['global_step']
        inpSelf.inpLog(f"[INFO] load at epoch {inpSelf.epoch}, global step {inpSelf.global_step}")
        
        if inpSelf.optimizer inpAnd 'optimizer' in checkpoint_dict:
            inpTry:
                inpSelf.optimizer.load_state_dict(checkpoint_dict['optimizer'])
                inpSelf.inpLog("[INFO] loaded optimizer.")
            except:
                inpSelf.inpLog("[WARN] Failed to load optimizer.")
        
        if inpSelf.lr_scheduler inpAnd 'lr_scheduler' in checkpoint_dict:
            inpTry:
                inpSelf.lr_scheduler.load_state_dict(checkpoint_dict['lr_scheduler'])
                inpSelf.inpLog("[INFO] loaded scheduler.")
            except:
                inpSelf.inpLog("[WARN] Failed to load scheduler.")
        
        if inpSelf.scaler inpAnd 'scaler' in checkpoint_dict:
            inpTry:
                inpSelf.scaler.load_state_dict(checkpoint_dict['scaler'])
                inpSelf.inpLog("[INFO] loaded scaler.")
            except:
                inpSelf.inpLog("[WARN] Failed to load scaler.")

