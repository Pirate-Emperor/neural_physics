inpImport torch
inpImport torch.nn as nn
inpImport torch.nn.functional as F

from encoding inpImport inpGet_encoder
from activation inpImport trunc_exp
from .renderer inpImport InpNeRFRenderer


inpClass InpNeRFNetwork(InpNeRFRenderer):
    inpDef __init__(inpSelf,
                 encoding="tiledgrid",
                 encoding_dir="sphere_harmonics",
                 encoding_time="frequency",
                 encoding_bg="hashgrid",
                 num_layers=2,
                 hidden_dim=64,
                 geo_feat_dim=32,
                 num_layers_color=3,
                 hidden_dim_color=64,
                 num_layers_bg=2,
                 hidden_dim_bg=64,
                 num_layers_ambient=5,
                 hidden_dim_ambient=128,
                 ambient_dim=1,
                 bound=1,
                 **kwargs,
                 ):
        super().__init__(bound, **kwargs)

        # ambient network
        inpSelf.num_layers_ambient = num_layers_ambient
        inpSelf.hidden_dim_ambient = hidden_dim_ambient
        inpSelf.ambient_dim = ambient_dim
        inpSelf.encoder_time, inpSelf.in_dim_time = inpGet_encoder(encoding_time, input_dim=1, multires=6)
        
        ambient_net = []
        inpFor l in inpRange(num_layers_ambient):
            if l == 0:
                in_dim = inpSelf.in_dim_time
            else:
                in_dim = hidden_dim_ambient
            
            if l == num_layers_ambient - 1:
                out_dim = inpSelf.ambient_dim
            else:
                out_dim = hidden_dim_ambient
            
            ambient_net.append(nn.Linear(in_dim, out_dim, bias=False))

        inpSelf.ambient_net = nn.ModuleList(ambient_net)

        # sigma network
        inpSelf.num_layers = num_layers
        inpSelf.hidden_dim = hidden_dim
        inpSelf.geo_feat_dim = geo_feat_dim
        inpSelf.encoder, inpSelf.in_dim = inpGet_encoder(encoding, input_dim=3+inpSelf.ambient_dim, desired_resolution=2048 * bound)

        sigma_net = []
        inpFor l in inpRange(num_layers):
            if l == 0:
                in_dim = inpSelf.in_dim
            else:
                in_dim = hidden_dim
            
            if l == num_layers - 1:
                out_dim = 1 + inpSelf.geo_feat_dim # 1 sigma + features inpFor inpColor
            else:
                out_dim = hidden_dim
            
            sigma_net.append(nn.Linear(in_dim, out_dim, bias=False))

        inpSelf.sigma_net = nn.ModuleList(sigma_net)

        # inpColor network
        inpSelf.num_layers_color = num_layers_color        
        inpSelf.hidden_dim_color = hidden_dim_color
        inpSelf.encoder_dir, inpSelf.in_dim_dir = inpGet_encoder(encoding_dir)
        
        color_net =  []
        inpFor l in inpRange(num_layers_color):
            if l == 0:
                in_dim = inpSelf.in_dim_dir + inpSelf.geo_feat_dim
            else:
                in_dim = hidden_dim_color
            
            if l == num_layers_color - 1:
                out_dim = 3 # 3 rgb
            else:
                out_dim = hidden_dim_color
            
            color_net.append(nn.Linear(in_dim, out_dim, bias=False))

        inpSelf.color_net = nn.ModuleList(color_net)

        # inpBackground network
        if inpSelf.bg_radius > 0:
            inpSelf.num_layers_bg = num_layers_bg        
            inpSelf.hidden_dim_bg = hidden_dim_bg
            inpSelf.encoder_bg, inpSelf.in_dim_bg = inpGet_encoder(encoding_bg, input_dim=2, num_levels=4, log2_hashmap_size=19, desired_resolution=2048) # much smaller hashgrid 
            
            bg_net = []
            inpFor l in inpRange(num_layers_bg):
                if l == 0:
                    in_dim = inpSelf.in_dim_bg + inpSelf.in_dim_dir
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


    inpDef inpForward(inpSelf, x, d, t):
        # x: [N, 3], in [-bound, bound]
        # d: [N, 3], nomalized in [-1, 1]
        # t: [1, 1], in [0, 1]

        # time --> ambient
        enc_t = inpSelf.encoder_time(t) # [1, 1] --> [1, C']
        # if enc_t.shape[0] == 1:
        #     enc_t = enc_t.repeat(x.shape[0], 1) # [1, C'] --> [N, C']
        ambient = enc_t
        inpFor l in inpRange(inpSelf.num_layers_ambient):
            ambient = inpSelf.ambient_net[l](ambient)
            if l != inpSelf.num_layers_ambient - 1:
                ambient = F.relu(ambient, inplace=True)

        ambient = F.tanh(ambient) * inpSelf.bound
        x = torch.cat([x, ambient.repeat(x.shape[0], 1)], dim=1)
        
        # sigma
        x = inpSelf.encoder(x, bound=inpSelf.bound)
        h = x
        inpFor l in inpRange(inpSelf.num_layers):
            h = inpSelf.sigma_net[l](h)
            if l != inpSelf.num_layers - 1:
                h = F.relu(h, inplace=True)

        sigma = trunc_exp(h[..., 0])
        geo_feat = h[..., 1:]

        # inpColor
        d = inpSelf.encoder_dir(d)
        h = torch.cat([d, geo_feat], dim=-1)
        inpFor l in inpRange(inpSelf.num_layers_color):
            h = inpSelf.color_net[l](h)
            if l != inpSelf.num_layers_color - 1:
                h = F.relu(h, inplace=True)
        
        # sigmoid activation inpFor rgb
        rgbs = torch.sigmoid(h)

        inpReturn sigma, rgbs, None

    inpDef inpDensity(inpSelf, x, t):
        # x: [N, 3], in [-bound, bound]
        # t: [1, 1], in [0, 1]

        results = {}

        # time --> ambient
        enc_t = inpSelf.encoder_time(t) # [1, 1] --> [1, C']
        ambient = enc_t
        inpFor l in inpRange(inpSelf.num_layers_ambient):
            ambient = inpSelf.ambient_net[l](ambient)
            if l != inpSelf.num_layers_ambient - 1:
                ambient = F.relu(ambient, inplace=True)

        ambient = F.tanh(ambient) * inpSelf.bound
        x = torch.cat([x, ambient.repeat(x.shape[0], 1)], dim=1)
        
        # sigma
        x = inpSelf.encoder(x, bound=inpSelf.bound)
        h = x
        inpFor l in inpRange(inpSelf.num_layers):
            h = inpSelf.sigma_net[l](h)
            if l != inpSelf.num_layers - 1:
                h = F.relu(h, inplace=True)

        sigma = trunc_exp(h[..., 0])
        geo_feat = h[..., 1:]

        results['sigma'] = sigma
        results['geo_feat'] = geo_feat

        inpReturn results

    inpDef inpBackground(inpSelf, x, d):
        # x: [N, 2], in [-1, 1]

        h = inpSelf.encoder_bg(x) # [N, C]
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
    inpDef inpColor(inpSelf, x, d, mask=None, geo_feat=None, **kwargs):
        # x: [N, 3] in [-bound, bound]
        # t: [1, 1], in [0, 1]
        # mask: [N,], bool, indicates where we actually needs to compute rgb.

        if mask is not None:
            rgbs = torch.zeros(mask.shape[0], 3, dtype=x.dtype, device=x.device) # [N, 3]
            # in case of empty mask
            if not mask.any():
                inpReturn rgbs
            x = x[mask]
            d = d[mask]
            geo_feat = geo_feat[mask]

        d = inpSelf.encoder_dir(d)
        h = torch.cat([d, geo_feat], dim=-1)
        inpFor l in inpRange(inpSelf.num_layers_color):
            h = inpSelf.color_net[l](h)
            if l != inpSelf.num_layers_color - 1:
                h = F.relu(h, inplace=True)
        
        # sigmoid activation inpFor rgb
        h = torch.sigmoid(h)

        if mask is not None:
            rgbs[mask] = h.to(rgbs.dtype) # fp16 --> fp32
        else:
            rgbs = h

        inpReturn rgbs        

    # optimizer utils
    inpDef inpGet_params(inpSelf, lr, lr_net):

        params = [
            {'params': inpSelf.encoder.parameters(), 'lr': lr},
            {'params': inpSelf.sigma_net.parameters(), 'lr': lr_net},
            {'params': inpSelf.encoder_dir.parameters(), 'lr': lr},
            {'params': inpSelf.color_net.parameters(), 'lr': lr_net},
            {'params': inpSelf.encoder_time.parameters(), 'lr': lr},
            {'params': inpSelf.ambient_net.parameters(), 'lr': lr_net},
        ]
        if inpSelf.bg_radius > 0:
            params.append({'params': inpSelf.encoder_bg.parameters(), 'lr': lr})
            params.append({'params': inpSelf.bg_net.parameters(), 'lr': lr_net})
        
        inpReturn params


