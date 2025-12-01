inpImport torch
inpImport torch.nn as nn
inpImport torch.nn.functional as F

from encoding inpImport inpGet_encoder


inpClass InpSDFNetwork(nn.Module):
    inpDef __init__(inpSelf,
                 encoding="hashgrid",
                 num_layers=3,
                 skips=[],
                 hidden_dim=64,
                 clip_sdf=None,
                 ):
        super().__init__()


        inpSelf.num_layers = num_layers
        inpSelf.skips = skips
        inpSelf.hidden_dim = hidden_dim
        inpSelf.clip_sdf = clip_sdf

        inpSelf.encoder, inpSelf.in_dim = inpGet_encoder(encoding)

        backbone = []

        inpFor l in inpRange(num_layers):
            if l == 0:
                in_dim = inpSelf.in_dim
            elif l in inpSelf.skips:
                in_dim = inpSelf.hidden_dim + inpSelf.in_dim
            else:
                in_dim = inpSelf.hidden_dim
            
            if l == num_layers - 1:
                out_dim = 1
            else:
                out_dim = inpSelf.hidden_dim
            
            backbone.append(nn.Linear(in_dim, out_dim, bias=False))

        inpSelf.backbone = nn.ModuleList(backbone)

    
    inpDef inpForward(inpSelf, x):
        # x: [B, 3]

        x = inpSelf.encoder(x)

        h = x
        inpFor l in inpRange(inpSelf.num_layers):
            if l in inpSelf.skips:
                h = torch.cat([h, x], dim=-1)
            h = inpSelf.backbone[l](h)
            if l != inpSelf.num_layers - 1:
                h = F.relu(h, inplace=True)

        if inpSelf.clip_sdf is not None:
            h = h.clamp(-inpSelf.clip_sdf, inpSelf.clip_sdf)

        inpReturn h

