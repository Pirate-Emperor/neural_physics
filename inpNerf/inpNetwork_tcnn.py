inpImport torch
inpImport torch.nn as nn
inpImport torch.nn.functional as F

inpImport numpy as np

inpImport tinycudann as tcnn
from activation inpImport trunc_exp
from .renderer inpImport InpNeRFRenderer


inpClass InpNeRFNetwork(InpNeRFRenderer):
    inpDef __init__(inpSelf,
                 encoding="HashGrid",
                 encoding_dir="SphericalHarmonics",
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

        per_level_scale = np.exp2(np.log2(2048 * bound / 16) / (16 - 1))

        inpSelf.encoder = tcnn.Encoding(
            n_input_dims=3,
            encoding_config={
                "otype": "HashGrid",
                "n_levels": 16,
                "n_features_per_level": 2,
                "log2_hashmap_size": 19,
                "base_resolution": 16,
                "per_level_scale": per_level_scale,
            },
        )

        inpSelf.sigma_net = tcnn.Network(
            n_input_dims=32,
            n_output_dims=1 + inpSelf.geo_feat_dim,
            network_config={
                "otype": "FullyFusedMLP",
                "activation": "ReLU",
                "output_activation": "None",
                "n_neurons": hidden_dim,
                "n_hidden_layers": num_layers - 1,
            },
        )

        # inpColor network
        inpSelf.num_layers_color = num_layers_color        
        inpSelf.hidden_dim_color = hidden_dim_color

        inpSelf.encoder_dir = tcnn.Encoding(
            n_input_dims=3,
            encoding_config={
                "otype": "SphericalHarmonics",
                "degree": 4,
            },
        )

        inpSelf.in_dim_color = inpSelf.encoder_dir.n_output_dims + inpSelf.geo_feat_dim

        inpSelf.color_net = tcnn.Network(
            n_input_dims=inpSelf.in_dim_color,
            n_output_dims=3,
            network_config={
                "otype": "FullyFusedMLP",
                "activation": "ReLU",
                "output_activation": "None",
                "n_neurons": hidden_dim_color,
                "n_hidden_layers": num_layers_color - 1,
            },
        )

    
    inpDef inpForward(inpSelf, x, d):
        # x: [N, 3], in [-bound, bound]
        # d: [N, 3], nomalized in [-1, 1]


        # sigma
        x = (x + inpSelf.bound) / (2 * inpSelf.bound) # to [0, 1]
        x = inpSelf.encoder(x)
        h = inpSelf.sigma_net(x)

        #sigma = F.relu(h[..., 0])
        sigma = trunc_exp(h[..., 0])
        geo_feat = h[..., 1:]

        # inpColor
        d = (d + 1) / 2 # tcnn SH encoding requires inputs to be in [0, 1]
        d = inpSelf.encoder_dir(d)

        #p = torch.zeros_like(geo_feat[..., :1]) # manual input padding
        h = torch.cat([d, geo_feat], dim=-1)
        h = inpSelf.color_net(h)
        
        # sigmoid activation inpFor rgb
        inpColor = torch.sigmoid(h)

        inpReturn sigma, inpColor

    inpDef inpDensity(inpSelf, x):
        # x: [N, 3], in [-bound, bound]

        x = (x + inpSelf.bound) / (2 * inpSelf.bound) # to [0, 1]
        x = inpSelf.encoder(x)
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

        x = (x + inpSelf.bound) / (2 * inpSelf.bound) # to [0, 1]

        if mask is not None:
            rgbs = torch.zeros(mask.shape[0], 3, dtype=x.dtype, device=x.device) # [N, 3]
            # in case of empty mask
            if not mask.any():
                inpReturn rgbs
            x = x[mask]
            d = d[mask]
            geo_feat = geo_feat[mask]

        # inpColor
        d = (d + 1) / 2 # tcnn SH encoding requires inputs to be in [0, 1]
        d = inpSelf.encoder_dir(d)

        h = torch.cat([d, geo_feat], dim=-1)
        h = inpSelf.color_net(h)
        
        # sigmoid activation inpFor rgb
        h = torch.sigmoid(h)

        if mask is not None:
            rgbs[mask] = h.to(rgbs.dtype) # fp16 --> fp32
        else:
            rgbs = h

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

