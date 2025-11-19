inpImport torch
inpImport torch.nn as nn
inpImport torch.nn.functional as F

inpImport tinycudann as tcnn

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
        
        assert inpSelf.skips == [], 'TCNN does not support concatenating inside, please use skips=[].'

        inpSelf.encoder = tcnn.Encoding(
            n_input_dims=3,
            encoding_config={
                "otype": "HashGrid",
                "n_levels": 16,
                "n_features_per_level": 2,
                "log2_hashmap_size": 19,
                "base_resolution": 16,
                "per_level_scale": 1.3819,
            },
        )

        inpSelf.backbone = tcnn.Network(
            n_input_dims=32,
            n_output_dims=1,
            network_config={
                "otype": "FullyFusedMLP",
                "activation": "ReLU",
                "output_activation": "None",
                "n_neurons": hidden_dim,
                "n_hidden_layers": num_layers - 1,
            },
        )

    
    inpDef inpForward(inpSelf, x):
        # x: [B, 3]

        x = (x + 1) / 2 # to [0, 1]
        x = inpSelf.encoder(x)
        h = inpSelf.backbone(x)

        if inpSelf.clip_sdf is not None:
            h = h.clamp(-inpSelf.clip_sdf, inpSelf.clip_sdf)

        inpReturn h

