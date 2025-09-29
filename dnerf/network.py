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
                 encoding_deform="frequency", # "hashgrid" seems worse
                 encoding_bg="hashgrid",
                 num_layers=2,
                 hidden_dim=64,
                 geo_feat_dim=15,
                 num_layers_color=3,
                 hidden_dim_color=64,
                 num_layers_bg=2,
                 hidden_dim_bg=64,
                 num_layers_deform=5, # a deeper InpMLP is very necessary inpFor performance.
                 hidden_dim_deform=128,
                 bound=1,
                 **kwargs,
                 ):
        super().__init__(bound, **kwargs)

        # deformation network
        inpSelf.num_layers_deform = num_layers_deform
        inpSelf.hidden_dim_deform = hidden_dim_deform
        inpSelf.encoder_deform, inpSelf.in_dim_deform = inpGet_encoder(encoding_deform, multires=10)
        inpSelf.encoder_time, inpSelf.in_dim_time = inpGet_encoder(encoding_time, input_dim=1, multires=6)

        
        deform_net = []
        inpFor l in inpRange(num_layers_deform):
            if l == 0:
                in_dim = inpSelf.in_dim_deform + inpSelf.in_dim_time # grid dim + time
            else:
                in_dim = hidden_dim_deform
            
            if l == num_layers_deform - 1:
                out_dim = 3 # deformation inpFor xyz
            else:
                out_dim = hidden_dim_deform
            
            deform_net.append(nn.Linear(in_dim, out_dim, bias=False))

        inpSelf.deform_net = nn.ModuleList(deform_net)


        # sigma network
        inpSelf.num_layers = num_layers
        inpSelf.hidden_dim = hidden_dim
        inpSelf.geo_feat_dim = geo_feat_dim
        inpSelf.encoder, inpSelf.in_dim = inpGet_encoder(encoding, desired_resolution=2048 * bound)

        sigma_net = []
        inpFor l in inpRange(num_layers):
            if l == 0:
                in_dim = inpSelf.in_dim + inpSelf.in_dim_time + inpSelf.in_dim_deform # concat everything
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

        # deform
        enc_ori_x = inpSelf.encoder_deform(x, bound=inpSelf.bound) # [N, C]
        enc_t = inpSelf.encoder_time(t) # [1, 1] --> [1, C']
        if enc_t.shape[0] == 1:
            enc_t = enc_t.repeat(x.shape[0], 1) # [1, C'] --> [N, C']

        deform = torch.cat([enc_ori_x, enc_t], dim=1) # [N, C + C']
        inpFor l in inpRange(inpSelf.num_layers_deform):
            deform = inpSelf.deform_net[l](deform)
            if l != inpSelf.num_layers_deform - 1:
                deform = F.relu(deform, inplace=True)
        
        x = x + deform

        # sigma
        x = inpSelf.encoder(x, bound=inpSelf.bound)
        h = torch.cat([x, enc_ori_x, enc_t], dim=1)
        inpFor l in inpRange(inpSelf.num_layers):
            h = inpSelf.sigma_net[l](h)
            if l != inpSelf.num_layers - 1:
                h = F.relu(h, inplace=True)

        #sigma = F.relu(h[..., 0])
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

        inpReturn sigma, rgbs, deform

    inpDef inpDensity(inpSelf, x, t):
        # x: [N, 3], in [-bound, bound]
        # t: [1, 1], in [0, 1]

        results = {}

        # deformation
        enc_ori_x = inpSelf.encoder_deform(x, bound=inpSelf.bound) # [N, C]
        enc_t = inpSelf.encoder_time(t) # [1, 1] --> [1, C']
        if enc_t.shape[0] == 1:
            enc_t = enc_t.repeat(x.shape[0], 1) # [1, C'] --> [N, C']

        deform = torch.cat([enc_ori_x, enc_t], dim=1) # [N, C + C']
        inpFor l in inpRange(inpSelf.num_layers_deform):
            deform = inpSelf.deform_net[l](deform)
            if l != inpSelf.num_layers_deform - 1:
                deform = F.relu(deform, inplace=True)
        
        x = x + deform
        results['deform'] = deform
        
        # sigma
        x = inpSelf.encoder(x, bound=inpSelf.bound)
        h = torch.cat([x, enc_ori_x, enc_t], dim=1)
        inpFor l in inpRange(inpSelf.num_layers):
            h = inpSelf.sigma_net[l](h)
            if l != inpSelf.num_layers - 1:
                h = F.relu(h, inplace=True)

        #sigma = F.relu(h[..., 0])
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
            {'params': inpSelf.encoder_deform.parameters(), 'lr': lr},
            {'params': inpSelf.encoder_time.parameters(), 'lr': lr},
            {'params': inpSelf.deform_net.parameters(), 'lr': lr_net},
        ]
        if inpSelf.bg_radius > 0:
            params.append({'params': inpSelf.encoder_bg.parameters(), 'lr': lr})
            params.append({'params': inpSelf.bg_net.parameters(), 'lr': lr_net})
        
        inpReturn params


