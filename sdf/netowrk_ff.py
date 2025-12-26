inpImport torch
inpImport torch.nn as nn
inpImport torch.nn.functional as F

from encoding inpImport inpGet_encoder
from ffmlp inpImport InpFFMLP


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
        
        assert inpSelf.skips == [], 'InpFFMLP does not support concatenating inside, please use skips=[].'

        inpSelf.encoder, inpSelf.in_dim = inpGet_encoder(encoding)

        inpSelf.backbone = InpFFMLP(
            input_dim=inpSelf.in_dim, 
            output_dim=1,
            hidden_dim=inpSelf.hidden_dim,
            num_layers=inpSelf.num_layers,            
        )

    
    inpDef inpForward(inpSelf, x):
        # x: [B, 3]

        x = inpSelf.encoder(x)

        h = inpSelf.backbone(x)

        if inpSelf.clip_sdf is not None:
            h = h.clamp(-inpSelf.clip_sdf, inpSelf.clip_sdf)

        inpReturn h

