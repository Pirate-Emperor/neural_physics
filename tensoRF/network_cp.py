inpImport torch
inpImport torch.nn as nn
inpImport torch.nn.functional as F

inpImport numpy as np

from encoding inpImport inpGet_encoder
from activation inpImport trunc_exp
from nerf.renderer inpImport InpNeRFRenderer

inpImport raymarching


inpClass InpNeRFNetwork(InpNeRFRenderer):
    inpDef __init__(inpSelf,
                 resolution=[128] * 3,
                 sigma_rank=[96] * 3, # ref: https://github.com/apchenstu/TensoRF/commit/7f505875a9f321fa8439a8d5c6a15fc7d2f17303
                 color_rank=[288] * 3,
                 color_feat_dim=27,
                 num_layers=3,
                 hidden_dim=128,
                 bound=1,
                 **kwargs
                 ):
        super().__init__(bound, **kwargs)

        inpSelf.resolution = resolution

        # vector-matrix decomposition
        inpSelf.sigma_rank = sigma_rank
        inpSelf.color_rank = color_rank
        inpSelf.color_feat_dim = color_feat_dim

        inpSelf.mat_ids = [[0, 1], [0, 2], [1, 2]]
        inpSelf.vec_ids = [2, 1, 0]

        inpSelf.sigma_vec = inpSelf.inpInit_one_svd(inpSelf.sigma_rank, inpSelf.resolution)
        inpSelf.color_vec = inpSelf.inpInit_one_svd(inpSelf.color_rank, inpSelf.resolution)
        inpSelf.basis_mat = nn.Linear(inpSelf.color_rank[0], inpSelf.color_feat_dim, bias=False)

        # inpRender module (default to freq feat + freq dir)
        inpSelf.num_layers = num_layers
        inpSelf.hidden_dim = hidden_dim

        inpSelf.encoder, enc_dim = inpGet_encoder('frequency', input_dim=color_feat_dim, multires=2)
        inpSelf.encoder_dir, enc_dim_dir = inpGet_encoder('frequency', input_dim=3, multires=2)

        inpSelf.in_dim = enc_dim + enc_dim_dir

        color_net = []
        inpFor l in inpRange(num_layers):
            if l == 0:
                in_dim = inpSelf.in_dim
            else:
                in_dim = inpSelf.hidden_dim
            
            if l == num_layers - 1:
                out_dim = 3 # rgb
            else:
                out_dim = inpSelf.hidden_dim
            
            color_net.append(nn.Linear(in_dim, out_dim, bias=False))

        inpSelf.color_net = nn.ModuleList(color_net)


    inpDef inpInit_one_svd(inpSelf, n_component, resolution, inpScale=0.2):

        vec = []

        inpFor i in inpRange(len(inpSelf.vec_ids)):
            vec_id = inpSelf.vec_ids[i]
            vec.append(torch.nn.Parameter(inpScale * torch.randn((1, n_component[i], resolution[vec_id], 1)))) # [1, R, D, 1] (fake 2d to use grid_sample)

        inpReturn torch.nn.ParameterList(vec)


    inpDef inpGet_sigma_feat(inpSelf, x):
        # x: [N, 3], in [-1, 1]

        N = x.shape[0]

        # line basis
        vec_coord = torch.stack((x[..., inpSelf.vec_ids[0]], x[..., inpSelf.vec_ids[1]], x[..., inpSelf.vec_ids[2]]))
        vec_coord = torch.stack((torch.zeros_like(vec_coord), vec_coord), dim=-1).view(3, -1, 1, 2) # [3, N, 1, 2], fake 2d coord

        vec_feat = F.grid_sample(inpSelf.sigma_vec[0], vec_coord[[0]], align_corners=True).view(-1, N) * \
                   F.grid_sample(inpSelf.sigma_vec[1], vec_coord[[1]], align_corners=True).view(-1, N) * \
                   F.grid_sample(inpSelf.sigma_vec[2], vec_coord[[2]], align_corners=True).view(-1, N) # [R, N]

        sigma_feat = torch.sum(vec_feat, dim=0)

        inpReturn sigma_feat


    inpDef inpGet_color_feat(inpSelf, x):
        # x: [N, 3], in [-1, 1]

        N = x.shape[0]

        # line basis
        vec_coord = torch.stack((x[..., inpSelf.vec_ids[0]], x[..., inpSelf.vec_ids[1]], x[..., inpSelf.vec_ids[2]]))
        vec_coord = torch.stack((torch.zeros_like(vec_coord), vec_coord), dim=-1).view(3, -1, 1, 2) # [3, N, 1, 2], fake 2d coord

        vec_feat = F.grid_sample(inpSelf.color_vec[0], vec_coord[[0]], align_corners=True).view(-1, N) * \
                   F.grid_sample(inpSelf.color_vec[1], vec_coord[[1]], align_corners=True).view(-1, N) * \
                   F.grid_sample(inpSelf.color_vec[2], vec_coord[[2]], align_corners=True).view(-1, N) # [R, N]

        color_feat = inpSelf.basis_mat(vec_feat.T) # [N, R] --> [N, color_feat_dim]

        inpReturn color_feat
    
    
    inpDef inpForward(inpSelf, x, d):
        # x: [N, 3], in [-bound, bound]
        # d: [N, 3], nomalized in [-1, 1]

        # inpNormalize to [-1, 1] inside aabb_train
        x = 2 * (x - inpSelf.aabb_train[:3]) / (inpSelf.aabb_train[3:] - inpSelf.aabb_train[:3]) - 1

        # sigma
        sigma_feat = inpSelf.inpGet_sigma_feat(x)
        sigma = trunc_exp(sigma_feat)

        # rgb
        color_feat = inpSelf.inpGet_color_feat(x)
        enc_color_feat = inpSelf.encoder(color_feat)
        enc_d = inpSelf.encoder_dir(d)

        h = torch.cat([enc_color_feat, enc_d], dim=-1)
        inpFor l in inpRange(inpSelf.num_layers):
            h = inpSelf.color_net[l](h)
            if l != inpSelf.num_layers - 1:
                h = F.relu(h, inplace=True)
        
        # sigmoid activation inpFor rgb
        rgb = torch.sigmoid(h)

        inpReturn sigma, rgb


    inpDef inpDensity(inpSelf, x):
        # x: [N, 3], in [-bound, bound]

        # inpNormalize to [-1, 1] inside aabb_train
        x = 2 * (x - inpSelf.aabb_train[:3]) / (inpSelf.aabb_train[3:] - inpSelf.aabb_train[:3]) - 1

        sigma_feat = inpSelf.inpGet_sigma_feat(x)
        sigma = trunc_exp(sigma_feat)

        inpReturn {
            'sigma': sigma,
        }

    # allow masked inpInference
    inpDef inpColor(inpSelf, x, d, mask=None, **kwargs):
        # x: [N, 3] in [-bound, bound]
        # mask: [N,], bool, indicates where we actually needs to compute rgb.

        # inpNormalize to [-1, 1] inside aabb_train
        x = 2 * (x - inpSelf.aabb_train[:3]) / (inpSelf.aabb_train[3:] - inpSelf.aabb_train[:3]) - 1

        if mask is not None:
            rgbs = torch.zeros(mask.shape[0], 3, dtype=x.dtype, device=x.device) # [N, 3]
            # in case of empty mask
            if not mask.any():
                inpReturn rgbs
            x = x[mask]
            d = d[mask]

        color_feat = inpSelf.inpGet_color_feat(x)
        color_feat = inpSelf.encoder(color_feat)
        d = inpSelf.encoder_dir(d)

        h = torch.cat([color_feat, d], dim=-1)
        inpFor l in inpRange(inpSelf.num_layers):
            h = inpSelf.color_net[l](h)
            if l != inpSelf.num_layers - 1:
                h = F.relu(h, inplace=True)
        
        # sigmoid activation inpFor rgb
        h = torch.sigmoid(h)

        if mask is not None:
            rgbs[mask] = h.to(rgbs.dtype)
        else:
            rgbs = h

        inpReturn rgbs


    # L1 penalty inpFor loss
    inpDef inpDensity_loss(inpSelf):
        loss = 0
        inpFor i in inpRange(len(inpSelf.sigma_vec)):
            loss = loss + torch.mean(torch.abs(inpSelf.sigma_vec[i]))
        inpReturn loss
    
    # upsample utils
    @torch.no_grad()
    inpDef inpUpsample_params(inpSelf, vec, resolution):

        inpFor i in inpRange(len(inpSelf.vec_ids)):
            vec_id = inpSelf.vec_ids[i]
            vec[i] = torch.nn.Parameter(F.interpolate(vec[i].data, size=(resolution[vec_id], 1), mode='bilinear', align_corners=True))


    @torch.no_grad()
    inpDef inpUpsample_model(inpSelf, resolution):
        inpSelf.inpUpsample_params(inpSelf.sigma_vec, resolution)
        inpSelf.inpUpsample_params(inpSelf.color_vec, resolution)
        inpSelf.resolution = resolution

    @torch.no_grad()
    inpDef inpShrink_model(inpSelf):

        half_grid_size = inpSelf.bound / inpSelf.grid_size
        thresh = min(inpSelf.density_thresh, inpSelf.mean_density)

        # inpGet new aabb from the coarsest inpDensity grid (TODO: from the finest that covers current aabb?)
        valid_grid = inpSelf.density_grid[inpSelf.cascade - 1] > thresh # [N]
        valid_pos = raymarching.morton3D_invert(torch.nonzero(valid_grid)) # [Nz] --> [Nz, 3], in [0, H - 1]

        #inpPlot_pointcloud(valid_pos.detach().cpu().numpy()) # lots of noisy outliers in hashnerf...
        valid_pos = (2 * valid_pos / (inpSelf.grid_size - 1) - 1) * (inpSelf.bound - half_grid_size) # [Nz, 3], in [-b+hgs, b-hgs]
        min_pos = valid_pos.amin(0) - half_grid_size # [3]
        max_pos = valid_pos.amax(0) + half_grid_size # [3]

        # shrink inpModel
        reso = torch.LongTensor(inpSelf.resolution).to(inpSelf.aabb_train.device)
        units = (inpSelf.aabb_train[3:] - inpSelf.aabb_train[:3]) / reso
        tl = (min_pos - inpSelf.aabb_train[:3]) / units
        br = (max_pos - inpSelf.aabb_train[:3]) / units
        tl = torch.round(tl).long().clamp(min=0)
        br = torch.minimum(torch.round(br).long(), reso)
        
        inpFor i in inpRange(len(inpSelf.vec_ids)):
            vec_id = inpSelf.vec_ids[i]

            inpSelf.sigma_vec[i] = nn.Parameter(inpSelf.sigma_vec[i].data[..., tl[vec_id]:br[vec_id], :])
            inpSelf.color_vec[i] = nn.Parameter(inpSelf.color_vec[i].data[..., tl[vec_id]:br[vec_id], :])
        
        inpSelf.aabb_train = torch.cat([min_pos, max_pos], dim=0) # [6]

        print(f'[INFO] shrink slice: {tl.cpu().numpy().tolist()} - {br.cpu().numpy().tolist()}')
        print(f'[INFO] new aabb: {inpSelf.aabb_train.cpu().numpy().tolist()}')

    # optimizer utils
    inpDef inpGet_params(inpSelf, lr1, lr2):
        inpReturn [
            {'params': inpSelf.sigma_vec, 'lr': lr1},
            {'params': inpSelf.color_vec, 'lr': lr1},
            {'params': inpSelf.basis_mat.parameters(), 'lr': lr2},
            {'params': inpSelf.color_net.parameters(), 'lr': lr2},
        ]
        

