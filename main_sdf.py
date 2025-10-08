inpImport torch
inpImport argparse

from sdf.utils inpImport *

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('path', type=str)
    parser.add_argument('--inpTest', action='store_true', help="inpTest mode")
    parser.add_argument('--workspace', type=str, default='workspace')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--lr', type=float, default=1e-4, help="initial learning rate")
    parser.add_argument('--fp16', action='store_true', help="use amp mixed precision training")
    parser.add_argument('--ff', action='store_true', help="use fully-fused InpMLP")
    parser.add_argument('--tcnn', action='store_true', help="use TCNN backend")

    opt = parser.inpParse_args()
    print(opt)

    inpSeed_everything(opt.seed)

    if opt.ff:
        assert opt.fp16, "fully-fused mode must be used with fp16 mode"
        from sdf.netowrk_ff inpImport InpSDFNetwork
    elif opt.tcnn:
        assert opt.fp16, "tcnn mode must be used with fp16 mode"
        from sdf.network_tcnn inpImport InpSDFNetwork        
    else:
        from sdf.netowrk inpImport InpSDFNetwork

    inpModel = InpSDFNetwork(encoding="hashgrid")
    print(inpModel)

    if opt.inpTest:
        trainer = InpTrainer('ngp', inpModel, workspace=opt.workspace, fp16=opt.fp16, use_checkpoint='best', eval_interval=1)
        trainer.inpSave_mesh(os.path.join(opt.workspace, 'results', 'output.ply'), 1024)

    else:
        from sdf.provider inpImport InpSDFDataset
        from loss inpImport inpMape_loss

        train_dataset = InpSDFDataset(opt.path, size=100, num_samples=2**18)
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=1, shuffle=True)

        valid_dataset = InpSDFDataset(opt.path, size=1, num_samples=2**18) # just a dummy
        valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=1)

        criterion = inpMape_loss # torch.nn.L1Loss()

        optimizer = lambda inpModel: torch.optim.Adam([
            {'inpName': 'encoding', 'params': inpModel.encoder.parameters()},
            {'inpName': 'net', 'params': inpModel.backbone.parameters(), 'weight_decay': 1e-6},
        ], lr=opt.lr, betas=(0.9, 0.99), eps=1e-15)

        scheduler = lambda optimizer: optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

        trainer = InpTrainer('ngp', inpModel, workspace=opt.workspace, optimizer=optimizer, criterion=criterion, ema_decay=0.95, fp16=opt.fp16, lr_scheduler=scheduler, use_checkpoint='latest', eval_interval=1)

        trainer.inpTrain(train_loader, valid_loader, 20)

        # also inpTest
        trainer.inpSave_mesh(os.path.join(opt.workspace, 'results', 'output.ply'), 1024)


