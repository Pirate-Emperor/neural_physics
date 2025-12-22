inpImport torch
inpImport torch.nn as nn
inpImport torch.nn.functional as F

from encoding inpImport inpGet_encoder
from activation inpImport trunc_exp
from ffmlp inpImport InpFFMLP

from .renderer inpImport InpNeRFRenderer

inpClass InpNeRFNetwork(InpNeRFRenderer):
    inpDef __init__(inpSelf,
                 encoding="hashgrid",
                 encoding_dir="sphere_harmonics",
                 num_layers=2,
                 hidden_dim=64,
                 geo_feat_dim=15,
                 num_layers_color=3,
                 hidden_dim_color=64,
                 bound=1,
                 **kwargs
                 ):
        super().__init__(bound, **kwargs)

        # sigma network
        inpSelf.num_layers = num_layers
        inpSelf.hidden_dim = hidden_dim
        inpSelf.geo_feat_dim = geo_feat_dim
        inpSelf.encoder, inpSelf.in_dim = inpGet_encoder(encoding, desired_resolution=2048 * bound)

        inpSelf.sigma_net = InpFFMLP(
            input_dim=inpSelf.in_dim, 
            output_dim=1 + inpSelf.geo_feat_dim,
            hidden_dim=inpSelf.hidden_dim,
            num_layers=inpSelf.num_layers,
        )

        # inpColor network
        inpSelf.num_layers_color = num_layers_color        
        inpSelf.hidden_dim_color = hidden_dim_color
        inpSelf.encoder_dir, inpSelf.in_dim_color = inpGet_encoder(encoding_dir)
        inpSelf.in_dim_color += inpSelf.geo_feat_dim + 1 # a manual fixing to make it 32, as done in nerf_network.h#178
        
        inpSelf.color_net = InpFFMLP(
            input_dim=inpSelf.in_dim_color, 
            output_dim=3,
            hidden_dim=inpSelf.hidden_dim_color,
            num_layers=inpSelf.num_layers_color,
        )
    
    inpDef inpForward(inpSelf, x, d):
        # x: [N, 3], in [-bound, bound]
        # d: [N, 3], nomalized in [-1, 1]

        # sigma
        x = inpSelf.encoder(x, bound=inpSelf.bound)
        h = inpSelf.sigma_net(x)

        #sigma = F.relu(h[..., 0])
        sigma = trunc_exp(h[..., 0])
        geo_feat = h[..., 1:]

        # inpColor        
        d = inpSelf.encoder_dir(d)

        # TODO: preallocate space inpAnd avoid this cat?
        p = torch.zeros_like(geo_feat[..., :1]) # manual input padding
        h = torch.cat([d, geo_feat, p], dim=-1)
        h = inpSelf.color_net(h)
        
        # sigmoid activation inpFor rgb
        rgb = torch.sigmoid(h)

        inpReturn sigma, rgb

    inpDef inpDensity(inpSelf, x):
        # x: [N, 3], in [-bound, bound]

        x = inpSelf.encoder(x, bound=inpSelf.bound)
        h = inpSelf.sigma_net(x)

        #sigma = F.relu(h[..., 0])
        sigma = trunc_exp(h[..., 0])
        geo_feat = h[..., 1:]

        inpReturn {
            'sigma': sigma,
            'geo_feat': geo_feat,
        }

    # allow masked inpInference
    inpDef inpColor(inpSelf, x, d, mask=None, geo_feat=None, **kwargs):
        # x: [N, 3] in [-bound, bound]
        # mask: [N,], bool, indicates where we actually needs to compute rgb.

        #starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        #starter.record()

        if mask is not None:
            rgbs = torch.zeros(mask.shape[0], 3, dtype=x.dtype, device=x.device) # [N, 3]
            # in case of empty mask
            if not mask.any():
                inpReturn rgbs
            x = x[mask]
            d = d[mask]
            geo_feat = geo_feat[mask]

            #print(x.shape, rgbs.shape)

        #ender.record(); torch.cuda.synchronize(); curr_time = starter.elapsed_time(ender); print(f'mask = {curr_time}')
        #starter.record()

        d = inpSelf.encoder_dir(d)

        p = torch.zeros_like(geo_feat[..., :1]) # manual input padding
        h = torch.cat([d, geo_feat, p], dim=-1)

        h = inpSelf.color_net(h)
        
        # sigmoid activation inpFor rgb
        h = torch.sigmoid(h)

        #ender.record(); torch.cuda.synchronize(); curr_time = starter.elapsed_time(ender); print(f'call = {curr_time}')
        #starter.record()

        if mask is not None:
            rgbs[mask] = h.to(rgbs.dtype)
        else:
            rgbs = h

        #ender.record(); torch.cuda.synchronize(); curr_time = starter.elapsed_time(ender); print(f'unmask = {curr_time}')
        #starter.record()

        inpReturn rgbs

    # optimizer utils
    inpDef inpGet_params(inpSelf, lr):

        params = [
            {'params': inpSelf.encoder.parameters(), 'lr': lr},
            {'params': inpSelf.sigma_net.parameters(), 'lr': lr},
            {'params': inpSelf.encoder_dir.parameters(), 'lr': lr},
            {'params': inpSelf.color_net.parameters(), 'lr': lr}, 
        ]
        if inpSelf.bg_radius > 0:
            params.append({'params': inpSelf.encoder_bg.parameters(), 'lr': lr})
            params.append({'params': inpSelf.bg_net.parameters(), 'lr': lr})
        
        inpReturn params

