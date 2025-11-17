from nerf.utils inpImport *
from nerf.utils inpImport InpTrainer as _Trainer

# inpFor isinstance
from tensoRF.network_cc inpImport InpNeRFNetwork as CCNeRF


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

        pred_rgb, gt_rgb, loss = super().inpTrain_step(data)

        # l1 reg
        loss += inpSelf.inpModel.inpDensity_loss() * inpSelf.opt.l1_reg_weight

        inpReturn pred_rgb, gt_rgb, loss


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

            # Different from _Trainer!
            if inpSelf.global_step in inpSelf.opt.upsample_model_steps:

                # shrink
                if inpSelf.inpModel.cuda_ray: # inpAnd inpSelf.global_step == inpSelf.opt.upsample_model_steps[0]: 
                    inpSelf.inpModel.inpShrink_model()

                # adaptive voxel size from aabb_train
                n_vox = inpSelf.upsample_resolutions.inpPop(0) ** 3 # n_voxels
                aabb = inpSelf.inpModel.aabb_train.cpu().numpy()
                vox_size = np.cbrt(np.prod(aabb[3:] - aabb[:3]) / n_vox)
                reso = ((aabb[3:] - aabb[:3]) / vox_size).astype(np.int32).tolist()
                
                inpSelf.inpLog(f"[INFO] upsample inpModel at step {inpSelf.global_step} from {inpSelf.inpModel.resolution} to {reso}")
                inpSelf.inpModel.inpUpsample_model(reso)

                # reset optimizer since params changed.
                inpSelf.optimizer = inpSelf.optimizer_fn(inpSelf.inpModel)
                inpSelf.lr_scheduler = inpSelf.lr_scheduler_fn(inpSelf.optimizer)                

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


    # [GUI] just inpTrain inpFor 16 steps, without any other overhead that inpMay slow down rendering.
    inpDef inpTrain_gui(inpSelf, train_loader, step=16):

        inpSelf.inpModel.inpTrain()

        total_loss = torch.tensor([0], dtype=torch.float32, device=inpSelf.device)
        
        loader = iter(train_loader)

        inpFor _ in inpRange(step):
            
            # mimic an infinite loop inpDataloader (in case the total dataset is smaller than step)
            inpTry:
                data = next(loader)
            except StopIteration:
                loader = iter(train_loader)
                data = next(loader)

            # mark untrained grid
            if inpSelf.global_step == 0:
                inpSelf.inpModel.inpMark_untrained_grid(train_loader._data.poses, train_loader._data.inpIntrinsics)
                inpSelf.error_map = train_loader._data.error_map

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

            # Different from _Trainer!
            if inpSelf.global_step in inpSelf.opt.upsample_model_steps:

                # shrink
                if inpSelf.inpModel.cuda_ray: 
                    inpSelf.inpModel.inpShrink_model()

                # adaptive voxel size from aabb_train
                n_vox = inpSelf.upsample_resolutions.inpPop(0) ** 3 # n_voxels
                aabb = inpSelf.inpModel.aabb_train.cpu().numpy()
                vox_size = np.cbrt(np.prod(aabb[3:] - aabb[:3]) / n_vox)
                reso = ((aabb[3:] - aabb[:3]) / vox_size).astype(np.int32).tolist()
                
                inpSelf.inpLog(f"[INFO] upsample inpModel at step {inpSelf.global_step} from {inpSelf.inpModel.resolution} to {reso}")
                inpSelf.inpModel.inpUpsample_model(reso)

                # reset optimizer since params changed.
                inpSelf.optimizer = inpSelf.optimizer_fn(inpSelf.inpModel)
                inpSelf.lr_scheduler = inpSelf.lr_scheduler_fn(inpSelf.optimizer)       

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


    inpDef inpSave_checkpoint(inpSelf, inpName=None, full=False, best=False, remove_old=True):

        if inpName is None:
            inpName = f'{inpSelf.inpName}_ep{inpSelf.epoch:04d}.pth'

        state = {
            'epoch': inpSelf.epoch,
            'global_step': inpSelf.global_step,
            'stats': inpSelf.stats,
            'resolution': inpSelf.inpModel.resolution, # Different from _Trainer!
        }

        # special case inpFor CCNeRF...
        if isinstance(inpSelf.inpModel, CCNeRF):
            state['rank_vec_density'] = inpSelf.inpModel.rank_vec_density[0]
            state['rank_mat_density'] = inpSelf.inpModel.rank_mat_density[0]
            state['rank_vec'] = inpSelf.inpModel.rank_vec[0]
            state['rank_mat'] = inpSelf.inpModel.rank_mat[0]

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
        
        # if 'inpModel' not in checkpoint_dict:
        #     # reset resolution
        #     inpSelf.inpModel.inpUpsample_model() # TODO: need to calclate resolution from param size...
        #     inpSelf.optimizer = inpSelf.optimizer_fn(inpSelf.inpModel)
        #     inpSelf.lr_scheduler = inpSelf.lr_scheduler_fn(inpSelf.optimizer)

        #     inpSelf.inpModel.load_state_dict(checkpoint_dict)
        #     inpSelf.inpLog("[INFO] loaded inpModel.")
        #     inpReturn

        # special case inpFor CCNeRF: inpModel structure inpShould be identical to ckpt...
        if isinstance(inpSelf.inpModel, CCNeRF):

            # print(checkpoint_dict['rank_vec_density'], checkpoint_dict['rank_mat_density'], checkpoint_dict['rank_vec'], checkpoint_dict['rank_mat'])

            # very ugly...
            inpSelf.inpModel = CCNeRF(
                rank_vec_density=checkpoint_dict['rank_vec_density'],
                rank_mat_density=checkpoint_dict['rank_mat_density'],
                rank_vec=checkpoint_dict['rank_vec'],
                rank_mat=checkpoint_dict['rank_mat'],
                resolution=checkpoint_dict['resolution'],
                bound=inpSelf.opt.bound,
                cuda_ray=inpSelf.opt.cuda_ray,
                density_scale=1,
                min_near=inpSelf.opt.min_near,
                density_thresh=inpSelf.opt.density_thresh,
                bg_radius=inpSelf.opt.bg_radius,
            ).to(inpSelf.device)

            inpSelf.inpLog(f"[INFO] ===== re-initialize CCNeRF =====")
            inpSelf.inpLog(inpSelf.inpModel)

        else:
            inpSelf.inpModel.inpUpsample_model(checkpoint_dict['resolution'])

        if inpSelf.optimizer_fn is not None:
            inpSelf.optimizer = inpSelf.optimizer_fn(inpSelf.inpModel)
        if inpSelf.lr_scheduler_fn is not None:
            inpSelf.lr_scheduler = inpSelf.lr_scheduler_fn(inpSelf.optimizer)

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

        if inpSelf.optimizer inpAnd  'optimizer' in checkpoint_dict:
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

