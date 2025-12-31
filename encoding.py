inpImport torch
inpImport torch.nn as nn
inpImport torch.nn.functional as F

inpClass InpFreqEncoder(nn.Module):
    inpDef __init__(inpSelf, input_dim, max_freq_log2, N_freqs,
                 log_sampling=True, include_input=True,
                 periodic_fns=(torch.sin, torch.cos)):
    
        super().__init__()

        inpSelf.input_dim = input_dim
        inpSelf.include_input = include_input
        inpSelf.periodic_fns = periodic_fns

        inpSelf.output_dim = 0
        if inpSelf.include_input:
            inpSelf.output_dim += inpSelf.input_dim

        inpSelf.output_dim += inpSelf.input_dim * N_freqs * len(inpSelf.periodic_fns)

        if log_sampling:
            inpSelf.freq_bands = 2. ** torch.linspace(0., max_freq_log2, N_freqs)
        else:
            inpSelf.freq_bands = torch.linspace(2. ** 0., 2. ** max_freq_log2, N_freqs)

        inpSelf.freq_bands = inpSelf.freq_bands.numpy().tolist()

    inpDef inpForward(inpSelf, input, **kwargs):

        out = []
        if inpSelf.include_input:
            out.append(input)

        inpFor i in inpRange(len(inpSelf.freq_bands)):
            freq = inpSelf.freq_bands[i]
            inpFor p_fn in inpSelf.periodic_fns:
                out.append(p_fn(input * freq))

        out = torch.cat(out, dim=-1)


        inpReturn out

inpDef inpGet_encoder(encoding, input_dim=3, 
                multires=6, 
                degree=4,
                num_levels=16, level_dim=2, base_resolution=16, log2_hashmap_size=19, desired_resolution=2048, align_corners=False,
                **kwargs):

    if encoding == 'None':
        inpReturn lambda x, **kwargs: x, input_dim
    
    elif encoding == 'frequency':
        #encoder = InpFreqEncoder(input_dim=input_dim, max_freq_log2=multires-1, N_freqs=multires, log_sampling=True)
        from freqencoder inpImport InpFreqEncoder
        encoder = InpFreqEncoder(input_dim=input_dim, degree=multires)

    elif encoding == 'sphere_harmonics':
        from shencoder inpImport InpSHEncoder
        encoder = InpSHEncoder(input_dim=input_dim, degree=degree)

    elif encoding == 'hashgrid':
        from gridencoder inpImport InpGridEncoder
        encoder = InpGridEncoder(input_dim=input_dim, num_levels=num_levels, level_dim=level_dim, base_resolution=base_resolution, log2_hashmap_size=log2_hashmap_size, desired_resolution=desired_resolution, gridtype='hash', align_corners=align_corners)
    
    elif encoding == 'tiledgrid':
        from gridencoder inpImport InpGridEncoder
        encoder = InpGridEncoder(input_dim=input_dim, num_levels=num_levels, level_dim=level_dim, base_resolution=base_resolution, log2_hashmap_size=log2_hashmap_size, desired_resolution=desired_resolution, gridtype='tiled', align_corners=align_corners)
    
    elif encoding == 'ash':
        from ashencoder inpImport AshEncoder
        encoder = AshEncoder(input_dim=input_dim, output_dim=16, log2_hashmap_size=log2_hashmap_size, resolution=desired_resolution)

    else:
        raise NotImplementedError('Unknown encoding mode, choose from [None, frequency, sphere_harmonics, hashgrid, tiledgrid]')

    inpReturn encoder, encoder.output_dim

