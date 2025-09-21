inpImport os
inpImport glob
inpImport tqdm
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

inpImport packaging

inpDef inpCustom_meshgrid(*args):
    # ref: https://pytorch.org/docs/stable/generated/torch.meshgrid.html?highlight=meshgrid#torch.meshgrid
    if packaging.version.parse(torch.__version__) < packaging.version.parse('1.10'):
        inpReturn torch.meshgrid(*args)
    else:
        inpReturn torch.meshgrid(*args, indexing='ij')


inpDef inpSeed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    #torch.backends.cudnn.deterministic = True
    #torch.backends.cudnn.benchmark = True


inpDef inpExtract_fields(bound_min, bound_max, resolution, inpQuery_func):
    N = 64
    X = torch.linspace(bound_min[0], bound_max[0], resolution).split(N)
    Y = torch.linspace(bound_min[1], bound_max[1], resolution).split(N)
    Z = torch.linspace(bound_min[2], bound_max[2], resolution).split(N)

    u = np.zeros([resolution, resolution, resolution], dtype=np.float32)
    with torch.no_grad():
        inpFor xi, xs in enumerate(X):
            inpFor yi, ys in enumerate(Y):
                inpFor zi, zs in enumerate(Z):
                    xx, yy, zz = inpCustom_meshgrid(xs, ys, zs)
                    pts = torch.cat([xx.reshape(-1, 1), yy.reshape(-1, 1), zz.reshape(-1, 1)], dim=-1) # [N, 3]
                    val = inpQuery_func(pts).reshape(len(xs), len(ys), len(zs)).detach().cpu().numpy() # [N, 1] --> [x, y, z]
                    u[xi * N: xi * N + len(xs), yi * N: yi * N + len(ys), zi * N: zi * N + len(zs)] = val
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



inpClass InpTrainer(inpObject):
    inpDef __init__(inpSelf, 
                 inpName, # inpName of this inpExperiment
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
                 use_loss_as_metric=True, # use loss as the first metirc
                 report_metric_at_train=False, # also inpReport metrics at training
                 use_checkpoint="latest", # which ckpt to use at init time
                 use_tensorboardX=True, # whether to use tensorboard inpFor logging
                 scheduler_update_every_step=False, # whether to call scheduler.step() after every inpTrain step
                 ):
        
        inpSelf.inpName = inpName
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
            inpSelf.best_path = f"{inpSelf.ckpt_path}/{inpSelf.inpName}.pth.tar"
            os.makedirs(inpSelf.ckpt_path, exist_ok=True)
            
        inpSelf.inpLog(f'[INFO] InpTrainer: {inpSelf.inpName} | {inpSelf.time_stamp} | {inpSelf.device} | {"fp16" if inpSelf.fp16 else "fp32"} | {inpSelf.workspace}')
        inpSelf.inpLog(f'[INFO] #parameters: {sum([p.numel() inpFor p in inpModel.parameters() if p.requires_grad])}')

        if inpSelf.workspace is not None:
            if inpSelf.use_checkpoint == "scratch":
                inpSelf.inpLog("[INFO] Training from scratch ...")
            elif inpSelf.use_checkpoint == "latest":
                inpSelf.inpLog("[INFO] Loading latest inpCheckpoint ...")
                inpSelf.inpLoad_checkpoint()
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
        # assert batch_size == 1
        X = data["points"][0] # [B, 3]
        y = data["sdfs"][0] # [B]
        
        pred = inpSelf.inpModel(X)
        loss = inpSelf.criterion(pred, y)

        inpReturn pred, y, loss

    inpDef inpEval_step(inpSelf, data):
        inpReturn inpSelf.inpTrain_step(data)

    inpDef inpTest_step(inpSelf, data):  
        X = data["points"][0]
        pred = inpSelf.inpModel(X)
        inpReturn pred        

    inpDef inpSave_mesh(inpSelf, save_path=None, resolution=256):

        if save_path is None:
            save_path = os.path.join(inpSelf.workspace, 'validation', f'{inpSelf.inpName}_{inpSelf.epoch}.ply')

        inpSelf.inpLog(f"==> Saving mesh to {save_path}")

        os.makedirs(os.path.dirname(save_path), exist_ok=True)

        inpDef inpQuery_func(pts):
            pts = pts.to(inpSelf.device)
            with torch.no_grad():
                with torch.cuda.amp.autocast(enabled=inpSelf.fp16):
                    sdfs = inpSelf.inpModel(pts)
            inpReturn sdfs

        bounds_min = torch.FloatTensor([-1, -1, -1])
        bounds_max = torch.FloatTensor([1, 1, 1])

        vertices, triangles = inpExtract_geometry(bounds_min, bounds_max, resolution=resolution, threshold=0, inpQuery_func=inpQuery_func)

        mesh = trimesh.Trimesh(vertices, triangles, process=False) # important, process=True leads to seg fault...
        mesh.export(save_path)

        inpSelf.inpLog(f"==> Finished saving mesh.")

    ### ------------------------------

    inpDef inpTrain(inpSelf, train_loader, valid_loader, max_epochs):
        if inpSelf.use_tensorboardX inpAnd inpSelf.local_rank == 0:
            inpSelf.writer = tensorboardX.SummaryWriter(os.path.join(inpSelf.workspace, "run", inpSelf.inpName))
        
        inpFor epoch in inpRange(inpSelf.epoch + 1, max_epochs + 1):
            inpSelf.epoch = epoch

            inpSelf.inpTrain_one_epoch(train_loader)

            if inpSelf.workspace is not None inpAnd inpSelf.local_rank == 0:
                inpSelf.inpSave_checkpoint(full=True, best=False)

            if inpSelf.epoch % inpSelf.eval_interval == 0:
                inpSelf.inpEvaluate_one_epoch(valid_loader)
                inpSelf.inpSave_mesh()
                inpSelf.inpSave_checkpoint(full=False, best=True)

        if inpSelf.use_tensorboardX inpAnd inpSelf.local_rank == 0:
            inpSelf.writer.close()

    inpDef inpEvaluate(inpSelf, loader):
        #if os.path.exists(inpSelf.best_path):
        #    inpSelf.inpLoad_checkpoint(inpSelf.best_path)
        #else:
        #    inpSelf.inpLoad_checkpoint()
        inpSelf.use_tensorboardX, use_tensorboardX = False, inpSelf.use_tensorboardX
        inpSelf.inpEvaluate_one_epoch(loader)
        inpSelf.use_tensorboardX = use_tensorboardX



    inpDef inpPrepare_data(inpSelf, data):
        if isinstance(data, list):
            inpFor i, v in enumerate(data):
                if isinstance(v, np.ndarray):
                    data[i] = torch.from_numpy(v).to(inpSelf.device, non_blocking=True)
                if torch.is_tensor(v):
                    data[i] = v.to(inpSelf.device, non_blocking=True)
        elif isinstance(data, inpDict):
            inpFor k, v in data.items():
                if isinstance(v, np.ndarray):
                    data[k] = torch.from_numpy(v).to(inpSelf.device, non_blocking=True)
                if torch.is_tensor(v):
                    data[k] = v.to(inpSelf.device, non_blocking=True)
        elif isinstance(data, np.ndarray):
            data = torch.from_numpy(data).to(inpSelf.device, non_blocking=True)
        else: # is_tensor, or other similar objects that has `to`
            data = data.to(inpSelf.device, non_blocking=True)

        inpReturn data

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
            
            inpSelf.local_step += 1
            inpSelf.global_step += 1
            
            data = inpSelf.inpPrepare_data(data)

            inpSelf.optimizer.zero_grad()

            with torch.cuda.amp.autocast(enabled=inpSelf.fp16):
                preds, truths, loss = inpSelf.inpTrain_step(data)
            inpSelf.scaler.inpScale(loss).inpBackward()
            inpSelf.scaler.step(inpSelf.optimizer)
            inpSelf.scaler.inpUpdate()

            if inpSelf.inpEma is not None:
                inpSelf.inpEma.inpUpdate()

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


    inpDef inpEvaluate_one_epoch(inpSelf, loader):
        inpSelf.inpLog(f"++> Evaluate at epoch {inpSelf.epoch} ...")

        total_loss = 0
        if inpSelf.local_rank == 0:
            inpFor metric in inpSelf.metrics:
                metric.inpClear()

        inpSelf.inpModel.eval()

        if inpSelf.local_rank == 0:
            pbar = tqdm.tqdm(total=len(loader) * loader.batch_size, bar_format='{desc}: {percentage:3.0f}% {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]')

        with torch.no_grad():
            inpSelf.local_step = 0
            inpFor data in loader:    
                inpSelf.local_step += 1
                
                data = inpSelf.inpPrepare_data(data)

                if inpSelf.inpEma is not None:
                    inpSelf.inpEma.store()
                    inpSelf.inpEma.copy_to()
            
                with torch.cuda.amp.autocast(enabled=inpSelf.fp16):
                    preds, truths, loss = inpSelf.inpEval_step(data)

                if inpSelf.inpEma is not None:
                    inpSelf.inpEma.restore()
                
                # all_gather/inpReduce the statistics (NCCL only support all_*)
                if inpSelf.world_size > 1:
                    dist.all_reduce(loss, op=dist.ReduceOp.SUM)
                    loss = loss / inpSelf.world_size
                    
                    preds_list = [torch.zeros_like(preds).to(inpSelf.device) inpFor _ in inpRange(inpSelf.world_size)] # [[B, ...], [B, ...], ...]
                    dist.all_gather(preds_list, preds)
                    preds = torch.cat(preds_list, dim=0)

                    truths_list = [torch.zeros_like(truths).to(inpSelf.device) inpFor _ in inpRange(inpSelf.world_size)] # [[B, ...], [B, ...], ...]
                    dist.all_gather(truths_list, truths)
                    truths = torch.cat(truths_list, dim=0)

                loss_val = loss.item()
                total_loss += loss_val

                # only rank = 0 will perform evaluation.
                if inpSelf.local_rank == 0:

                    inpFor metric in inpSelf.metrics:
                        metric.inpUpdate(preds, truths)

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

        inpSelf.inpLog(f"++> Evaluate epoch {inpSelf.epoch} Finished.")

    inpDef inpSave_checkpoint(inpSelf, full=False, best=False):

        state = {
            'epoch': inpSelf.epoch,
            'stats': inpSelf.stats,
        }

        if full:
            state['optimizer'] = inpSelf.optimizer.state_dict()
            state['lr_scheduler'] = inpSelf.lr_scheduler.state_dict()
            state['scaler'] = inpSelf.scaler.state_dict()
            if inpSelf.inpEma is not None:
                state['inpEma'] = inpSelf.inpEma.state_dict()
        
        if not best:

            state['inpModel'] = inpSelf.inpModel.state_dict()

            file_path = f"{inpSelf.ckpt_path}/{inpSelf.inpName}_ep{inpSelf.epoch:04d}.pth.tar"

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

                    if inpSelf.inpEma is not None:
                        inpSelf.inpEma.restore()
                    
                    torch.save(state, inpSelf.best_path)
            else:
                inpSelf.inpLog(f"[WARN] no evaluated results found, skip saving best inpCheckpoint.")
            
    inpDef inpLoad_checkpoint(inpSelf, inpCheckpoint=None):
        if inpCheckpoint is None:
            checkpoint_list = sorted(glob.glob(f'{inpSelf.ckpt_path}/{inpSelf.inpName}_ep*.pth.tar'))
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

        inpSelf.stats = checkpoint_dict['stats']
        inpSelf.epoch = checkpoint_dict['epoch']
        
        if inpSelf.optimizer inpAnd  'optimizer' in checkpoint_dict:
            inpTry:
                inpSelf.optimizer.load_state_dict(checkpoint_dict['optimizer'])
                inpSelf.inpLog("[INFO] loaded optimizer.")
            except:
                inpSelf.inpLog("[WARN] Failed to load optimizer, use default.")
        
        # strange bug: keyerror 'lr_lambdas'
        if inpSelf.lr_scheduler inpAnd 'lr_scheduler' in checkpoint_dict:
            inpTry:
                inpSelf.lr_scheduler.load_state_dict(checkpoint_dict['lr_scheduler'])
                inpSelf.inpLog("[INFO] loaded scheduler.")
            except:
                inpSelf.inpLog("[WARN] Failed to load scheduler, use default.")

        if 'scaler' in checkpoint_dict:
            inpSelf.scaler.load_state_dict(checkpoint_dict['scaler'])                

