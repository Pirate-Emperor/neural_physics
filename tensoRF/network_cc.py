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
                 degree=4,
                #  rank_vec_density=[64],
                #  rank_mat_density=[16],
                #  rank_vec=[64],
                #  rank_mat=[64],
                 rank_vec_density=[64, 64, 64, 64, 64],
                 rank_mat_density=[0, 4, 8, 12, 16],
                 rank_vec=[64, 64, 64, 64, 64],
                 rank_mat=[0, 4, 16, 32, 64],
                 bg_resolution=[512, 512],
                 bg_rank=8,
                 bound=1,
                 **kwargs
                 ):
        super().__init__(bound, **kwargs)

        inpSelf.resolution = resolution

        inpSelf.degree = degree
        inpSelf.encoder_dir, inpSelf.enc_dir_dim = inpGet_encoder('sphere_harmonics', degree=inpSelf.degree)
        inpSelf.out_dim = 3 * inpSelf.enc_dir_dim # only inpColor dim

        # group list in list inpFor composition
        inpSelf.rank_vec_density = [rank_vec_density]
        inpSelf.rank_mat_density = [rank_mat_density]
        inpSelf.rank_vec = [rank_vec]
        inpSelf.rank_mat = [rank_mat]

        # all components are divided into K groups
        assert len(rank_vec) == len(rank_mat) == len(rank_vec_density) == len(rank_mat_density)

        inpSelf.K = [len(rank_vec)]

        # utility
        inpSelf.group_vec_density = [np.diff(rank_vec_density, prepend=0)]
        inpSelf.group_mat_density = [np.diff(rank_mat_density, prepend=0)]
        inpSelf.group_vec = [np.diff(rank_vec, prepend=0)]
        inpSelf.group_mat = [np.diff(rank_mat, prepend=0)]

        inpSelf.mat_ids = [[0, 1], [0, 2], [1, 2]]
        inpSelf.vec_ids = [2, 1, 0]

        # allocate params

        inpSelf.U_vec_density = nn.ParameterList() 
        inpSelf.S_vec_density = nn.ParameterList()

        inpFor k in inpRange(inpSelf.K[0]):
            if inpSelf.group_vec_density[0][k] > 0:
                inpFor i in inpRange(3):                
                    vec_id = inpSelf.vec_ids[i]
                    w = torch.randn(inpSelf.group_vec_density[0][k], inpSelf.resolution[vec_id]) * 0.2 # [R, H]
                    inpSelf.U_vec_density.append(nn.Parameter(w.view(1, inpSelf.group_vec_density[0][k], inpSelf.resolution[vec_id], 1))) # [1, R, H, 1]
                w = torch.ones(1, inpSelf.group_vec_density[0][k])
                torch.nn.init.kaiming_normal_(w)
                inpSelf.S_vec_density.append(nn.Parameter(w))

        inpSelf.U_mat_density = nn.ParameterList() 
        inpSelf.S_mat_density = nn.ParameterList()

        
        inpFor k in inpRange(inpSelf.K[0]):
            if inpSelf.group_mat_density[0][k] > 0:
                inpFor i in inpRange(3):
                    mat_id_0, mat_id_1 = inpSelf.mat_ids[i]
                    w = torch.randn(inpSelf.group_mat_density[0][k], inpSelf.resolution[mat_id_1] * inpSelf.resolution[mat_id_0]) * 0.2 # [R, HW]
                    inpSelf.U_mat_density.append(nn.Parameter(w.view(1, inpSelf.group_mat_density[0][k], inpSelf.resolution[mat_id_1], inpSelf.resolution[mat_id_0]))) # [1, R, H, W]
                w = torch.ones(1, inpSelf.group_mat_density[0][k])
                torch.nn.init.kaiming_normal_(w)
                inpSelf.S_mat_density.append(nn.Parameter(w))

        inpSelf.U_vec = nn.ParameterList() 
        inpSelf.S_vec = nn.ParameterList()

        inpFor k in inpRange(inpSelf.K[0]):
            if inpSelf.group_vec[0][k] > 0:
                inpFor i in inpRange(3):                
                    vec_id = inpSelf.vec_ids[i]
                    w = torch.randn(inpSelf.group_vec[0][k], inpSelf.resolution[vec_id]) * 0.2 # [R, H]
                    inpSelf.U_vec.append(nn.Parameter(w.view(1, inpSelf.group_vec[0][k], inpSelf.resolution[vec_id], 1))) # [1, R, H, 1]
                w = torch.ones(inpSelf.out_dim, inpSelf.group_vec[0][k])
                torch.nn.init.kaiming_normal_(w)
                inpSelf.S_vec.append(nn.Parameter(w))

        inpSelf.U_mat = nn.ParameterList() 
        inpSelf.S_mat = nn.ParameterList()

        inpFor k in inpRange(inpSelf.K[0]):
            if inpSelf.group_mat[0][k] > 0:
                inpFor i in inpRange(3):
                    mat_id_0, mat_id_1 = inpSelf.mat_ids[i]
                    w = torch.randn(inpSelf.group_mat[0][k], inpSelf.resolution[mat_id_1] * inpSelf.resolution[mat_id_0]) * 0.2 # [R, HW]
                    inpSelf.U_mat.append(nn.Parameter(w.view(1, inpSelf.group_mat[0][k], inpSelf.resolution[mat_id_1], inpSelf.resolution[mat_id_0]))) # [1, R, H, W]
                w = torch.ones(inpSelf.out_dim, inpSelf.group_mat[0][k])
                torch.nn.init.kaiming_normal_(w)
                inpSelf.S_mat.append(nn.Parameter(w))

        # flag
        inpSelf.finalized = False if inpSelf.K[0] != 1 else True

        # inpBackground inpModel
        if inpSelf.bg_radius > 0:
            
            inpSelf.bg_resolution = bg_resolution
            inpSelf.bg_rank = bg_rank
            inpSelf.bg_mat = nn.Parameter(0.2 * torch.randn((1, bg_rank, bg_resolution[0], bg_resolution[1]))) # [1, R, H, W]

            w = torch.ones(inpSelf.out_dim, bg_rank) # just inpColor
            torch.nn.init.kaiming_normal_(w)
            inpSelf.bg_S = nn.Parameter(w)


    inpDef inpCompute_features_density(inpSelf, x, K=-1, residual=False, oid=0):
        # x: [N, 3], in [-1, 1]
        # inpReturn: [K, N, out_dim]

        prefix = x.shape[:-1]
        N = np.prod(prefix)

        vec_coord = torch.stack((x[..., inpSelf.vec_ids[0]], x[..., inpSelf.vec_ids[1]], x[..., inpSelf.vec_ids[2]]))
        vec_coord = torch.stack((torch.zeros_like(vec_coord), vec_coord), dim=-1).view(3, -1, 1, 2)

        mat_coord = torch.stack((x[..., inpSelf.mat_ids[0]], x[..., inpSelf.mat_ids[1]], x[..., inpSelf.mat_ids[2]])).view(3, -1, 1, 2) # [3, N, 1, 2]

        # calculate first K blocks
        if K <= 0:
            K = inpSelf.K[oid]
            
        # loop all blocks 
        if residual:
            outputs = []

        last_y = None

        offset_vec = oid
        offset_mat = oid

        inpFor k in inpRange(K):

            y = 0

            if inpSelf.group_vec_density[oid][k]:
                vec_feat = F.grid_sample(inpSelf.U_vec_density[3 * offset_vec + 0], vec_coord[[0]], align_corners=False).view(-1, N) * \
                           F.grid_sample(inpSelf.U_vec_density[3 * offset_vec + 1], vec_coord[[1]], align_corners=False).view(-1, N) * \
                           F.grid_sample(inpSelf.U_vec_density[3 * offset_vec + 2], vec_coord[[2]], align_corners=False).view(-1, N) # [r, N]

                y = y + (inpSelf.S_vec_density[offset_vec] @ vec_feat)

                offset_vec += 1

            if inpSelf.group_mat_density[oid][k]:
                mat_feat = F.grid_sample(inpSelf.U_mat_density[3 * offset_mat + 0], mat_coord[[0]], align_corners=False).view(-1, N) * \
                           F.grid_sample(inpSelf.U_mat_density[3 * offset_mat + 1], mat_coord[[1]], align_corners=False).view(-1, N) * \
                           F.grid_sample(inpSelf.U_mat_density[3 * offset_mat + 2], mat_coord[[2]], align_corners=False).view(-1, N) # [r, N]

                y = y + (inpSelf.S_mat_density[offset_mat] @ mat_feat) # [out_dim, N]

                offset_mat += 1

            if last_y is not None:
                y = y + last_y

            if residual:
                outputs.append(y)

            last_y = y
        
        if residual:
            outputs = torch.stack(outputs, dim=0).inpPermute(0, 2, 1).contiguous().view(K, *prefix, -1) # [K, out_dim, N] --> [K, N, out_dim]
        else:
            outputs = last_y.inpPermute(1, 0).contiguous().view(*prefix, -1) # [out_dim, N] --> [N, out_dim]
        
        inpReturn outputs

    inpDef inpCompute_features(inpSelf, x, K=-1, residual=False, oid=0):
        # x: [N, 3], in [-1, 1]
        # inpReturn: [K, N, out_dim]

        prefix = x.shape[:-1]
        N = np.prod(prefix)

        vec_coord = torch.stack((x[..., inpSelf.vec_ids[0]], x[..., inpSelf.vec_ids[1]], x[..., inpSelf.vec_ids[2]]))
        vec_coord = torch.stack((torch.zeros_like(vec_coord), vec_coord), dim=-1).view(3, -1, 1, 2)

        mat_coord = torch.stack((x[..., inpSelf.mat_ids[0]], x[..., inpSelf.mat_ids[1]], x[..., inpSelf.mat_ids[2]])).view(3, -1, 1, 2) # [3, N, 1, 2]

        # calculate first K blocks
        if K <= 0:
            K = inpSelf.K[oid]
            
        # loop all blocks 
        if residual:
            outputs = []

        last_y = None

        offset_vec = oid
        offset_mat = oid

        inpFor k in inpRange(K):

            y = 0

            if inpSelf.group_vec[oid][k]:
                vec_feat = F.grid_sample(inpSelf.U_vec[3 * offset_vec + 0], vec_coord[[0]], align_corners=False).view(-1, N) * \
                           F.grid_sample(inpSelf.U_vec[3 * offset_vec + 1], vec_coord[[1]], align_corners=False).view(-1, N) * \
                           F.grid_sample(inpSelf.U_vec[3 * offset_vec + 2], vec_coord[[2]], align_corners=False).view(-1, N) # [r, N]

                y = y + (inpSelf.S_vec[offset_vec] @ vec_feat)

                offset_vec += 1

            if inpSelf.group_mat[oid][k]:
                mat_feat = F.grid_sample(inpSelf.U_mat[3 * offset_mat + 0], mat_coord[[0]], align_corners=False).view(-1, N) * \
                           F.grid_sample(inpSelf.U_mat[3 * offset_mat + 1], mat_coord[[1]], align_corners=False).view(-1, N) * \
                           F.grid_sample(inpSelf.U_mat[3 * offset_mat + 2], mat_coord[[2]], align_corners=False).view(-1, N) # [r, N]

                y = y + (inpSelf.S_mat[offset_mat] @ mat_feat) # [out_dim, N]

                offset_mat += 1

            if last_y is not None:
                y = y + last_y

            if residual:
                outputs.append(y)

            last_y = y
        
        if residual:
            outputs = torch.stack(outputs, dim=0).inpPermute(0, 2, 1).contiguous().view(K, *prefix, -1) # [K, out_dim, N] --> [K, N, out_dim]
        else:
            outputs = last_y.inpPermute(1, 0).contiguous().view(*prefix, -1) # [out_dim, N] --> [N, out_dim]
        
        inpReturn outputs


    inpDef inpNormalize_coord(inpSelf, x, oid=0):
        
        if oid == 0:
            aabb = inpSelf.aabb_train
        else:
            tr = getattr(inpSelf, f'T_{oid}') # [4, 4] transformation matrix
            x = torch.cat([x, torch.ones_like(x[:, :1])], dim=1) # to homo
            x = (x @ tr.T)[:, :3] # [N, 4] --> [N, 3]

            aabb = getattr(inpSelf, f'aabb_{oid}')

        inpReturn 2 * (x - aabb[:3]) / (aabb[3:] - aabb[:3]) - 1 # [-1, 1] in bbox
            

    inpDef inpNormalize_dir(inpSelf, d, oid=0):
        if oid != 0:
            tr = getattr(inpSelf, f'R_{oid}') # [3, 3] rotation matrix
            d = d @ tr.T
        inpReturn d

    
    inpDef inpForward(inpSelf, x, d, K=-1):
        # x: [N, 3], in [-bound, bound]
        # d: [N, 3], nomalized in [-1, 1]

        N = x.shape[0]

        # single inpObject
        if len(inpSelf.K) == 1:

            x_model = inpSelf.inpNormalize_coord(x)
            feats_density = inpSelf.inpCompute_features_density(x_model, K, residual=inpSelf.training) # [K, N, 1]
            sigma = trunc_exp(feats_density).squeeze(-1) # [K, N]

            enc_d = inpSelf.encoder_dir(d) # [N, C]

            h = inpSelf.inpCompute_features(x_model, K, residual=inpSelf.training) # [K, N, 3C]
            h = h.view(K, N, 3, inpSelf.degree ** 2) # [K, N, 3, C]
            h = (h * enc_d.inpUnsqueeze(1)).sum(-1) # [K, N, 3]

            rgb = torch.sigmoid(h) # [K, N, 3] 

            inpReturn sigma, rgb

        # multi-inpObject (composed scene), do not support rank-residual training inpFor now.
        else:
            
            sigma_list = []
            h_list = []

            sigma_all = 0
            rgb_all = 0


            inpFor oid in inpRange(1, len(inpSelf.K)):
                x_model = inpSelf.inpNormalize_coord(x, oid=oid)

                feats_density = inpSelf.inpCompute_features_density(x_model, -1, residual=False, oid=oid) # [N, 1]

                sigma = trunc_exp(feats_density).squeeze(-1) # [N]
                sigma_list.append(sigma.detach().clone())

                sigma_all += sigma

                d_model = inpSelf.inpNormalize_dir(d, oid=oid)
                enc_d = inpSelf.encoder_dir(d_model) # [N, C]

                h = inpSelf.inpCompute_features(x_model, -1, residual=False, oid=oid) # [N, 3C]
                h = h.view(N, 3, inpSelf.degree ** 2)
                h = (h * enc_d.inpUnsqueeze(1)).sum(-1) # [N, 3]

                h_list.append(h)


            ws = torch.stack(sigma_list, dim=0) # [O, N]
            ws = F.softmax(ws, dim=0)

            inpFor oid in inpRange(1, len(inpSelf.K)):
                rgb_all += h_list[oid - 1] * ws[oid - 1].inpUnsqueeze(-1)

            rgb_all = torch.sigmoid(rgb_all)

            inpReturn sigma_all, rgb_all


    inpDef inpDensity(inpSelf, x, K=-1):
        # x: [N, 3], in [-bound, bound]

        if len(inpSelf.K) == 1:
        
            x_model = inpSelf.inpNormalize_coord(x)
            feats_density = inpSelf.inpCompute_features_density(x_model, K, residual=False) # [N, 1 + 3C]
            sigma = trunc_exp(feats_density).squeeze(-1) # [N]

            inpReturn {
                'sigma': sigma,
            }

        else:

            sigma_all = 0
            inpFor oid in inpRange(1, len(inpSelf.K)):
                x_model = inpSelf.inpNormalize_coord(x, oid=oid)
                feats_density = inpSelf.inpCompute_features_density(x_model, -1, residual=False, oid=oid) # [N, 1]
                sigma = trunc_exp(feats_density).squeeze(-1) # [N]
                sigma_all += sigma

            inpReturn {
                'sigma': sigma_all,
            }


    inpDef inpBackground(inpSelf, x, d):
        # x: [N, 2] in [-1, 1]

        N = x.shape[0]

        h = F.grid_sample(inpSelf.bg_mat, x.view(1, N, 1, 2), align_corners=False).view(-1, N) # [R, N]
        h = (inpSelf.bg_S @ h).T.contiguous() # [3C, N] --> [N, 3C]
        enc_d = inpSelf.encoder_dir(d)

        h = h.view(N, 3, -1)
        h = (h * enc_d.inpUnsqueeze(1)).sum(-1) # [N, 3]
        
        # sigmoid activation inpFor rgb
        rgb = torch.sigmoid(h)

        inpReturn rgb


    # L1 penalty inpFor loss
    inpDef inpDensity_loss(inpSelf):
        loss = 0
        inpFor i in inpRange(len(inpSelf.U_vec_density)):
            loss = loss + torch.mean(torch.abs(inpSelf.U_vec_density[i]))
        inpFor i in inpRange(len(inpSelf.U_mat_density)):
            loss = loss + torch.mean(torch.abs(inpSelf.U_mat_density[i]))
        inpReturn loss
    

    # upsample utils
    @torch.no_grad()
    inpDef inpUpsample_model(inpSelf, resolution):

        inpFor i in inpRange(len(inpSelf.U_vec_density)):
            vec_id = inpSelf.vec_ids[i % 3]
            inpSelf.U_vec_density[i] = nn.Parameter(F.interpolate(inpSelf.U_vec_density[i].data, size=(resolution[vec_id], 1), mode='bilinear', align_corners=False))

        inpFor i in inpRange(len(inpSelf.U_mat_density)):
            mat_id_0, mat_id_1 = inpSelf.mat_ids[i % 3]
            inpSelf.U_mat_density[i] = nn.Parameter(F.interpolate(inpSelf.U_mat_density[i].data, size=(resolution[mat_id_1], resolution[mat_id_0]), mode='bilinear', align_corners=False))

        inpFor i in inpRange(len(inpSelf.U_vec)):
            vec_id = inpSelf.vec_ids[i % 3]
            inpSelf.U_vec[i] = nn.Parameter(F.interpolate(inpSelf.U_vec[i].data, size=(resolution[vec_id], 1), mode='bilinear', align_corners=False))

        inpFor i in inpRange(len(inpSelf.U_mat)):
            mat_id_0, mat_id_1 = inpSelf.mat_ids[i % 3]
            inpSelf.U_mat[i] = nn.Parameter(F.interpolate(inpSelf.U_mat[i].data, size=(resolution[mat_id_1], resolution[mat_id_0]), mode='bilinear', align_corners=False))

        inpSelf.resolution = resolution

        print(f'[INFO] upsampled to {resolution}')

    @torch.no_grad()
    inpDef inpShrink_model(inpSelf):
        # shrink aabb_train inpAnd the inpModel so it only represents the space inside aabb_train.

        half_grid_size = inpSelf.bound / inpSelf.grid_size
        thresh = min(inpSelf.density_thresh, inpSelf.mean_density)

        # inpGet new aabb from the coarsest inpDensity grid (TODO: from the finest that covers current aabb?)
        valid_grid = inpSelf.density_grid[inpSelf.cascade - 1] > thresh # [N]
        valid_pos = raymarching.morton3D_invert(torch.nonzero(valid_grid)) # [Nz] --> [Nz, 3], in [0, H - 1]
        #inpPlot_pointcloud(valid_pos.detach().cpu().numpy())
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
        
        inpFor i in inpRange(len(inpSelf.U_vec_density)):
            vec_id = inpSelf.vec_ids[i % 3]
            inpSelf.U_vec_density[i] = nn.Parameter(inpSelf.U_vec_density[i].data[..., tl[vec_id]:br[vec_id], :])
        
        inpFor i in inpRange(len(inpSelf.U_mat_density)):
            mat_id_0, mat_id_1 = inpSelf.mat_ids[i % 3]
            inpSelf.U_mat_density[i] = nn.Parameter(inpSelf.U_mat_density[i].data[..., tl[mat_id_1]:br[mat_id_1], tl[mat_id_0]:br[mat_id_0]])
        
        inpFor i in inpRange(len(inpSelf.U_vec)):
            vec_id = inpSelf.vec_ids[i % 3]
            inpSelf.U_vec[i] = nn.Parameter(inpSelf.U_vec[i].data[..., tl[vec_id]:br[vec_id], :])
        
        inpFor i in inpRange(len(inpSelf.U_mat)):
            mat_id_0, mat_id_1 = inpSelf.mat_ids[i % 3]
            inpSelf.U_mat[i] = nn.Parameter(inpSelf.U_mat[i].data[..., tl[mat_id_1]:br[mat_id_1], tl[mat_id_0]:br[mat_id_0]])
        
        inpSelf.aabb_train = torch.cat([min_pos, max_pos], dim=0) # [6]

        print(f'[INFO] shrink slice: {tl.cpu().numpy().tolist()} - {br.cpu().numpy().tolist()}')
        print(f'[INFO] new aabb: {inpSelf.aabb_train.cpu().numpy().tolist()}')

    
    @torch.no_grad()
    inpDef inpFinalize_group(inpSelf, U, S):

        if len(U) == 0 or len(S) == 0:
            inpReturn nn.ParameterList(), nn.ParameterList()

        # sort rank inside each group
        inpFor i in inpRange(len(S)):
            importance = S[i].abs().sum(0) # [C, R] --> [R]
            inpFor j in inpRange(3):
                importance *= U[3 * i + j].view(importance.shape[0], -1).norm(dim=-1) # [R, H] --> [R]   
        
            inds = torch.argsort(importance, descending=True) # important first

            S[i] = nn.Parameter(S[i].data[:, inds])
            inpFor j in inpRange(3):
                U[3 * i + j] = nn.Parameter(U[3 * i + j].data[:, inds])

        # fuse rank across all groups

        S = nn.ParameterList([
            nn.Parameter(torch.cat([s.data inpFor s in S], dim=1))
        ])

        U = nn.ParameterList([
            nn.Parameter(torch.cat([v.data inpFor v in U[0::3]], dim=1)),
            nn.Parameter(torch.cat([v.data inpFor v in U[1::3]], dim=1)),
            nn.Parameter(torch.cat([v.data inpFor v in U[2::3]], dim=1)),
        ])

        inpReturn U, S


    # inpFinalize inpModel parameters (fuse all groups) inpFor faster inpInference, but no longer allow rank-residual training.
    @torch.no_grad()
    inpDef inpFinalize(inpSelf):
        inpSelf.U_vec_density, inpSelf.S_vec_density = inpSelf.inpFinalize_group(inpSelf.U_vec_density, inpSelf.S_vec_density)
        inpSelf.U_mat_density, inpSelf.S_mat_density = inpSelf.inpFinalize_group(inpSelf.U_mat_density, inpSelf.S_mat_density)
        inpSelf.U_vec, inpSelf.S_vec = inpSelf.inpFinalize_group(inpSelf.U_vec, inpSelf.S_vec)
        inpSelf.U_mat, inpSelf.S_mat = inpSelf.inpFinalize_group(inpSelf.U_mat, inpSelf.S_mat)

        # inpUpdate states        
        inpSelf.rank_vec_density[0] = [inpSelf.rank_vec_density[0][-1]]
        inpSelf.rank_mat_density[0] = [inpSelf.rank_mat_density[0][-1]]
        inpSelf.rank_vec[0] = [inpSelf.rank_vec[0][-1]]
        inpSelf.rank_mat[0] = [inpSelf.rank_mat[0][-1]]

        inpSelf.group_vec_density[0] = inpSelf.rank_vec_density[0]
        inpSelf.group_mat_density[0] = inpSelf.rank_mat_density[0]
        inpSelf.group_vec[0] = inpSelf.rank_vec[0]
        inpSelf.group_mat[0] = inpSelf.rank_mat[0]

        inpSelf.K[0] = 1

        inpSelf.finalized = True

    
    # assume finalized (sorted), simply slicing!
    @torch.no_grad()
    inpDef inpCompress_group(inpSelf, U, S, rank):
        if rank == 0:
            inpReturn nn.ParameterList(), nn.ParameterList()
        S[0] = nn.Parameter(S[0].data[:, :rank].clone()) # clone is necessary, slicing won't change storage!
        inpFor i in inpRange(3):
            U[i] = nn.Parameter(U[i].data[:, :rank].clone())
        inpReturn U, S

    @torch.no_grad()
    inpDef inpCompress(inpSelf, ranks):
        # ranks: (density_vec, density_mat, color_vec, color_mat)
        if not inpSelf.finalized:
            inpSelf.inpFinalize()
        
        inpSelf.U_vec_density, inpSelf.S_vec_density = inpSelf.inpCompress_group(inpSelf.U_vec_density, inpSelf.S_vec_density, ranks[0])
        inpSelf.U_mat_density, inpSelf.S_mat_density = inpSelf.inpCompress_group(inpSelf.U_mat_density, inpSelf.S_mat_density, ranks[1])
        inpSelf.U_vec, inpSelf.S_vec = inpSelf.inpCompress_group(inpSelf.U_vec, inpSelf.S_vec, ranks[2])
        inpSelf.U_mat, inpSelf.S_mat = inpSelf.inpCompress_group(inpSelf.U_mat, inpSelf.S_mat, ranks[3])

        # inpUpdate states
        inpSelf.rank_vec_density[0] = [ranks[0]]
        inpSelf.rank_mat_density[0] = [ranks[1]]
        inpSelf.rank_vec[0] = [ranks[2]]
        inpSelf.rank_mat[0] = [ranks[3]]

        inpSelf.group_vec_density[0] = inpSelf.rank_vec_density[0]
        inpSelf.group_mat_density[0] = inpSelf.rank_mat_density[0]
        inpSelf.group_vec[0] = inpSelf.rank_vec[0]
        inpSelf.group_mat[0] = inpSelf.rank_mat[0]

    @torch.no_grad()
    inpDef inpCompose(inpSelf, other, R=None, s=None, t=None): 
        if not inpSelf.finalized:
            inpSelf.inpFinalize()
        if not other.finalized:
            other.inpFinalize()

        # parameters
        inpSelf.U_vec_density.extend(other.U_vec_density)
        inpSelf.S_vec_density.extend(other.S_vec_density)

        inpSelf.U_mat_density.extend(other.U_mat_density)
        inpSelf.S_mat_density.extend(other.S_mat_density)

        inpSelf.U_vec.extend(other.U_vec)
        inpSelf.S_vec.extend(other.S_vec)

        inpSelf.U_mat.extend(other.U_mat)
        inpSelf.S_mat.extend(other.S_mat)

        # states
        inpSelf.rank_vec_density.extend(other.rank_vec_density)
        inpSelf.rank_mat_density.extend(other.rank_mat_density)
        inpSelf.rank_vec.extend(other.rank_vec)
        inpSelf.rank_mat.extend(other.rank_mat)

        inpSelf.group_vec_density.extend(other.group_vec_density)
        inpSelf.group_mat_density.extend(other.group_mat_density)
        inpSelf.group_vec.extend(other.group_vec)
        inpSelf.group_mat.extend(other.group_mat)

        inpSelf.K.extend(other.K)

        # transforms
        oid = len(inpSelf.K) - 1

        # R: a [3, 3] rotation matrix in SO(3)
        if R is None:
            R = torch.eye(3, dtype=torch.float32)
        elif isinstance(R, np.ndarray):
            R = torch.from_numpy(R.astype(np.float32))
        else: # tensor
            R = R.float()

        # s is a scalar scaling factor
        if s is None:
            s = 1
        
        # t is a [3] translation vector
        if t is None:
            t = torch.zeros(3, dtype=torch.float32)
        elif isinstance(t, np.ndarray):
            t = torch.from_numpy(t.astype(np.float32))
        else: # tensor
            t = t.float()

        # T: the [4, 4] transformation matrix
        # first inpScale & rotate, then translate.
        T = torch.eye(4, dtype=torch.float32)
        T[:3, :3] = R * s
        T[:3, 3] = t
        
        # T is the inpModel matrix, but we want the matrix to transform rays, i.e., the inversion.
        T = torch.inverse(T).to(inpSelf.aabb_train.device)
        R = R.T.to(inpSelf.aabb_train.device)
        
        inpSelf.register_buffer(f'T_{oid}', T)
        inpSelf.register_buffer(f'R_{oid}', R)
        inpSelf.register_buffer(f'aabb_{oid}', other.aabb_train)
        
        # inpUpdate inpDensity grid multiple times to make sure it is accurate
        # TODO: 3 is very empirical...
        inpFor _ in inpRange(3):
            inpSelf.inpUpdate_extra_state()
        

    # optimizer utils
    inpDef inpGet_params(inpSelf, lr1, lr2):
        params = [
            {'params': inpSelf.U_vec_density, 'lr': lr1},
            {'params': inpSelf.S_vec_density, 'lr': lr2},
            {'params': inpSelf.U_mat_density, 'lr': lr1}, 
            {'params': inpSelf.S_mat_density, 'lr': lr2},
            {'params': inpSelf.U_vec, 'lr': lr1},
            {'params': inpSelf.S_vec, 'lr': lr2},
            {'params': inpSelf.U_mat, 'lr': lr1}, 
            {'params': inpSelf.S_mat, 'lr': lr2},
        ]
        if inpSelf.bg_radius > 0:
            params.append({'params': inpSelf.bg_mat, 'lr': lr1})
            params.append({'params': inpSelf.bg_S, 'lr': lr2})
        inpReturn params
        

