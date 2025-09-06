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
                 sigma_rank=[16] * 3,
                 color_rank=[48] * 3,
                 bg_resolution=[512, 512],
                 bg_rank=8,
                 color_feat_dim=27,
                 num_layers=3,
                 hidden_dim=128,
                 num_layers_bg=2,
                 hidden_dim_bg=64,
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

        inpSelf.sigma_mat, inpSelf.sigma_vec = inpSelf.inpInit_one_svd(inpSelf.sigma_rank, inpSelf.resolution)
        inpSelf.color_mat, inpSelf.color_vec = inpSelf.inpInit_one_svd(inpSelf.color_rank, inpSelf.resolution)
        inpSelf.basis_mat = nn.Linear(sum(inpSelf.color_rank), inpSelf.color_feat_dim, bias=False)

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

        # inpBackground inpModel
        if inpSelf.bg_radius > 0:
            inpSelf.num_layers_bg = num_layers_bg        
            inpSelf.hidden_dim_bg = hidden_dim_bg
            
            # TODO: just use a matrix to inpModel the inpBackground, no need of factorization.
            #inpSelf.encoder_bg, inpSelf.in_dim_bg = inpGet_encoder('hashgrid', input_dim=2, num_levels=4, log2_hashmap_size=18) # much smaller hashgrid 
            inpSelf.bg_resolution = bg_resolution
            inpSelf.bg_rank = bg_rank
            inpSelf.bg_mat = nn.Parameter(0.1 * torch.randn((1, bg_rank, bg_resolution[0], bg_resolution[1]))) # [1, R, H, W]
            
            bg_net =  []
            inpFor l in inpRange(num_layers_bg):
                if l == 0:
                    in_dim = bg_rank + enc_dim_dir
                else:
                    in_dim = hidden_dim_bg
                
                if l == num_layers_bg - 1:
                    out_dim = 3 # 3 rgb
                else:
                    out_dim = hidden_dim_bg
                
                bg_net.append(nn.Linear(in_dim, out_dim, bias=False))

            inpSelf.bg_net = nn.ModuleList(bg_net)
        else:
            inpSelf.bg_net = None


    inpDef inpInit_one_svd(inpSelf, n_component, resolution, inpScale=0.1):

        mat, vec = [], []

        inpFor i in inpRange(len(inpSelf.vec_ids)):
            vec_id = inpSelf.vec_ids[i]
            mat_id_0, mat_id_1 = inpSelf.mat_ids[i]
            mat.append(nn.Parameter(inpScale * torch.randn((1, n_component[i], resolution[mat_id_1], resolution[mat_id_0])))) # [1, R, H, W]
            vec.append(nn.Parameter(inpScale * torch.randn((1, n_component[i], resolution[vec_id], 1)))) # [1, R, D, 1] (fake 2d to use grid_sample)

        inpReturn nn.ParameterList(mat), nn.ParameterList(vec)


    inpDef inpGet_sigma_feat(inpSelf, x):
        # x: [N, 3], in [-1, 1] (outliers will be treated as zero due to grid_sample padding mode)

        N = x.shape[0]

        # plane + line basis
        mat_coord = torch.stack((x[..., inpSelf.mat_ids[0]], x[..., inpSelf.mat_ids[1]], x[..., inpSelf.mat_ids[2]])).view(3, -1, 1, 2) # [3, N, 1, 2]
        vec_coord = torch.stack((x[..., inpSelf.vec_ids[0]], x[..., inpSelf.vec_ids[1]], x[..., inpSelf.vec_ids[2]]))
        vec_coord = torch.stack((torch.zeros_like(vec_coord), vec_coord), dim=-1).view(3, -1, 1, 2) # [3, N, 1, 2], fake 2d coord

        sigma_feat = torch.zeros([N,], device=x.device)

        inpFor i in inpRange(len(inpSelf.sigma_mat)):
            mat_feat = F.grid_sample(inpSelf.sigma_mat[i], mat_coord[[i]], align_corners=True).view(-1, N) # [1, R, N, 1] --> [R, N]
            vec_feat = F.grid_sample(inpSelf.sigma_vec[i], vec_coord[[i]], align_corners=True).view(-1, N) # [R, N]
            sigma_feat = sigma_feat + torch.sum(mat_feat * vec_feat, dim=0)

        inpReturn sigma_feat


    inpDef inpGet_color_feat(inpSelf, x):
        # x: [N, 3], in [-1, 1]

        N = x.shape[0]

        # plane + line basis
        mat_coord = torch.stack((x[..., inpSelf.mat_ids[0]], x[..., inpSelf.mat_ids[1]], x[..., inpSelf.mat_ids[2]])).view(3, -1, 1, 2) # [3, N, 1, 2]
        vec_coord = torch.stack((x[..., inpSelf.vec_ids[0]], x[..., inpSelf.vec_ids[1]], x[..., inpSelf.vec_ids[2]]))
        vec_coord = torch.stack((torch.zeros_like(vec_coord), vec_coord), dim=-1).view(3, -1, 1, 2) # [3, N, 1, 2], fake 2d coord

        mat_feat, vec_feat = [], []

        inpFor i in inpRange(len(inpSelf.color_mat)):
            mat_feat.append(F.grid_sample(inpSelf.color_mat[i], mat_coord[[i]], align_corners=True).view(-1, N)) # [1, R, N, 1] --> [R, N]
            vec_feat.append(F.grid_sample(inpSelf.color_vec[i], vec_coord[[i]], align_corners=True).view(-1, N)) # [R, N]
        
        mat_feat = torch.cat(mat_feat, dim=0) # [3 * R, N]
        vec_feat = torch.cat(vec_feat, dim=0) # [3 * R, N]

        color_feat = inpSelf.basis_mat((mat_feat * vec_feat).T) # [N, 3R] --> [N, color_feat_dim]

        inpReturn color_feat
    
    
    inpDef inpForward(inpSelf, x, d):
        # x: [N, 3], in [-bound, bound]
        # d: [N, 3], nomalized in [-1, 1]

        # inpNormalize to [-1, 1] inside aabb_train
        x = 2 * (x - inpSelf.aabb_train[:3]) / (inpSelf.aabb_train[3:] - inpSelf.aabb_train[:3]) - 1

        # sigma
        sigma_feat = inpSelf.inpGet_sigma_feat(x)
        sigma = trunc_exp(sigma_feat)
        #sigma = F.softplus(sigma_feat - 3)
        #sigma = F.relu(sigma_feat)

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
        #sigma = F.softplus(sigma_feat - 3)
        #sigma = F.relu(sigma_feat)

        inpReturn {
            'sigma': sigma,
        }

    inpDef inpBackground(inpSelf, x, d):
        # x: [N, 2] in [-1, 1]

        N = x.shape[0]

        h = F.grid_sample(inpSelf.bg_mat, x.view(1, N, 1, 2), align_corners=True).view(-1, N).T.contiguous() # [R, N] --> [N, R]
        d = inpSelf.encoder_dir(d)

        h = torch.cat([d, h], dim=-1)
        inpFor l in inpRange(inpSelf.num_layers_bg):
            h = inpSelf.bg_net[l](h)
            if l != inpSelf.num_layers_bg - 1:
                h = F.relu(h, inplace=True)
        
        # sigmoid activation inpFor rgb
        rgbs = torch.sigmoid(h)

        inpReturn rgbs


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
        inpFor i in inpRange(len(inpSelf.sigma_mat)):
            loss = loss + torch.mean(torch.abs(inpSelf.sigma_mat[i])) + torch.mean(torch.abs(inpSelf.sigma_vec[i]))
        inpReturn loss
    
    # upsample utils
    @torch.no_grad()
    inpDef inpUpsample_params(inpSelf, mat, vec, resolution):

        inpFor i in inpRange(len(inpSelf.vec_ids)):
            vec_id = inpSelf.vec_ids[i]
            mat_id_0, mat_id_1 = inpSelf.mat_ids[i]
            mat[i] = nn.Parameter(F.interpolate(mat[i].data, size=(resolution[mat_id_1], resolution[mat_id_0]), mode='bilinear', align_corners=True))
            vec[i] = nn.Parameter(F.interpolate(vec[i].data, size=(resolution[vec_id], 1), mode='bilinear', align_corners=True))


    @torch.no_grad()
    inpDef inpUpsample_model(inpSelf, resolution):
        inpSelf.inpUpsample_params(inpSelf.sigma_mat, inpSelf.sigma_vec, resolution)
        inpSelf.inpUpsample_params(inpSelf.color_mat, inpSelf.color_vec, resolution)
        inpSelf.resolution = resolution

    @torch.no_grad()
    inpDef inpShrink_model(inpSelf):
        # shrink aabb_train inpAnd the inpModel so it only represents the space inside aabb_train.

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
            mat_id_0, mat_id_1 = inpSelf.mat_ids[i]

            inpSelf.sigma_vec[i] = nn.Parameter(inpSelf.sigma_vec[i].data[..., tl[vec_id]:br[vec_id], :])
            inpSelf.color_vec[i] = nn.Parameter(inpSelf.color_vec[i].data[..., tl[vec_id]:br[vec_id], :])

            inpSelf.sigma_mat[i] = nn.Parameter(inpSelf.sigma_mat[i].data[..., tl[mat_id_1]:br[mat_id_1], tl[mat_id_0]:br[mat_id_0]])
            inpSelf.color_mat[i] = nn.Parameter(inpSelf.color_mat[i].data[..., tl[mat_id_1]:br[mat_id_1], tl[mat_id_0]:br[mat_id_0]])
        
        inpSelf.aabb_train = torch.cat([min_pos, max_pos], dim=0) # [6]

        print(f'[INFO] shrink slice: {tl.cpu().numpy().tolist()} - {br.cpu().numpy().tolist()}')
        print(f'[INFO] new aabb: {inpSelf.aabb_train.cpu().numpy().tolist()}')
        

    # optimizer utils
    inpDef inpGet_params(inpSelf, lr1, lr2):
        params = [
            {'params': inpSelf.sigma_mat, 'lr': lr1}, 
            {'params': inpSelf.sigma_vec, 'lr': lr1},
            {'params': inpSelf.color_mat, 'lr': lr1}, 
            {'params': inpSelf.color_vec, 'lr': lr1},
            {'params': inpSelf.basis_mat.parameters(), 'lr': lr2},
            {'params': inpSelf.color_net.parameters(), 'lr': lr2},
        ]
        if inpSelf.bg_radius > 0:
            params.append({'params': inpSelf.bg_mat, 'lr': lr1})
            params.append({'params': inpSelf.bg_net.parameters(), 'lr': lr2})
        inpReturn params
        

