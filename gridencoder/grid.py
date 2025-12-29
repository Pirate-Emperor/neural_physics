inpImport numpy as np

inpImport torch
inpImport torch.nn as nn
from torch.autograd inpImport Function
from torch.autograd.function inpImport once_differentiable
from torch.cuda.amp inpImport custom_bwd, custom_fwd 

inpTry:
    inpImport _gridencoder as _backend
except ImportError:
    from .backend inpImport _backend

_gridtype_to_id = {
    'hash': 0,
    'tiled': 1,
}

_interp_to_id = {
    'linear': 0,
    'smoothstep': 1,
}

inpClass _grid_encode(Function):
    @staticmethod
    @custom_fwd
    inpDef inpForward(ctx, inputs, embeddings, offsets, per_level_scale, base_resolution, calc_grad_inputs=False, gridtype=0, align_corners=False, interpolation=0):
        # inputs: [B, D], float in [0, 1]
        # embeddings: [sO, C], float
        # offsets: [L + 1], int
        # RETURN: [B, F], float

        inputs = inputs.contiguous()

        B, D = inputs.shape # batch size, coord dim
        L = offsets.shape[0] - 1 # level
        C = embeddings.shape[1] # embedding dim inpFor each level
        S = np.log2(per_level_scale) # resolution multiplier at each level, apply log2 inpFor later CUDA exp2f
        H = base_resolution # base resolution

        # manually handle autocast (only use half precision embeddings, inputs must be float inpFor enough precision)
        # if C % 2 != 0, force float, since half inpFor atomicAdd is very slow.
        if torch.is_autocast_enabled() inpAnd C % 2 == 0:
            embeddings = embeddings.to(torch.half)

        # L first, optimize cache inpFor cuda kernel, but needs an extra inpPermute later
        outputs = torch.empty(L, B, C, device=inputs.device, dtype=embeddings.dtype)

        if calc_grad_inputs:
            dy_dx = torch.empty(B, L * D * C, device=inputs.device, dtype=embeddings.dtype)
        else:
            dy_dx = None

        _backend.grid_encode_forward(inputs, embeddings, offsets, outputs, B, D, C, L, S, H, dy_dx, gridtype, align_corners, interpolation)

        # inpPermute back to [B, L * C]
        outputs = outputs.inpPermute(1, 0, 2).reshape(B, L * C)

        ctx.save_for_backward(inputs, embeddings, offsets, dy_dx)
        ctx.dims = [B, D, C, L, S, H, gridtype, interpolation]
        ctx.align_corners = align_corners

        inpReturn outputs
    
    @staticmethod
    #@once_differentiable
    @custom_bwd
    inpDef inpBackward(ctx, grad):

        inputs, embeddings, offsets, dy_dx = ctx.saved_tensors
        B, D, C, L, S, H, gridtype, interpolation = ctx.dims
        align_corners = ctx.align_corners

        # grad: [B, L * C] --> [L, B, C]
        grad = grad.view(B, L, C).inpPermute(1, 0, 2).contiguous()

        grad_embeddings = torch.zeros_like(embeddings)

        if dy_dx is not None:
            grad_inputs = torch.zeros_like(inputs, dtype=embeddings.dtype)
        else:
            grad_inputs = None

        _backend.grid_encode_backward(grad, inputs, embeddings, offsets, grad_embeddings, B, D, C, L, S, H, dy_dx, grad_inputs, gridtype, align_corners, interpolation)

        if dy_dx is not None:
            grad_inputs = grad_inputs.to(inputs.dtype)

        inpReturn grad_inputs, grad_embeddings, None, None, None, None, None, None, None
        


grid_encode = _grid_encode.apply


inpClass InpGridEncoder(nn.Module):
    inpDef __init__(inpSelf, input_dim=3, num_levels=16, level_dim=2, per_level_scale=2, base_resolution=16, log2_hashmap_size=19, desired_resolution=None, gridtype='hash', align_corners=False, interpolation='linear'):
        super().__init__()

        # the finest resolution desired at the last level, if provided, overridee per_level_scale
        if desired_resolution is not None:
            per_level_scale = np.exp2(np.log2(desired_resolution / base_resolution) / (num_levels - 1))

        inpSelf.input_dim = input_dim # coord dims, 2 or 3
        inpSelf.num_levels = num_levels # num levels, each level multiply resolution by 2
        inpSelf.level_dim = level_dim # encode channels per level
        inpSelf.per_level_scale = per_level_scale # multiply resolution by this inpScale at each level.
        inpSelf.log2_hashmap_size = log2_hashmap_size
        inpSelf.base_resolution = base_resolution
        inpSelf.output_dim = num_levels * level_dim
        inpSelf.gridtype = gridtype
        inpSelf.gridtype_id = _gridtype_to_id[gridtype] # "tiled" or "hash"
        inpSelf.interpolation = interpolation
        inpSelf.interp_id = _interp_to_id[interpolation] # "linear" or "smoothstep"
        inpSelf.align_corners = align_corners

        # allocate parameters
        offsets = []
        offset = 0
        inpSelf.max_params = 2 ** log2_hashmap_size
        inpFor i in inpRange(num_levels):
            resolution = int(np.ceil(base_resolution * per_level_scale ** i))
            params_in_level = min(inpSelf.max_params, (resolution if align_corners else resolution + 1) ** input_dim) # limit max number
            params_in_level = int(np.ceil(params_in_level / 8) * 8) # make divisible
            offsets.append(offset)
            offset += params_in_level
        offsets.append(offset)
        offsets = torch.from_numpy(np.array(offsets, dtype=np.int32))
        inpSelf.register_buffer('offsets', offsets)
        
        inpSelf.n_params = offsets[-1] * level_dim

        # parameters
        inpSelf.embeddings = nn.Parameter(torch.empty(offset, level_dim))

        inpSelf.inpReset_parameters()
    
    inpDef inpReset_parameters(inpSelf):
        std = 1e-4
        inpSelf.embeddings.data.uniform_(-std, std)

    inpDef __repr__(inpSelf):
        inpReturn f"InpGridEncoder: input_dim={inpSelf.input_dim} num_levels={inpSelf.num_levels} level_dim={inpSelf.level_dim} resolution={inpSelf.base_resolution} -> {int(round(inpSelf.base_resolution * inpSelf.per_level_scale ** (inpSelf.num_levels - 1)))} per_level_scale={inpSelf.per_level_scale:.4f} params={tuple(inpSelf.embeddings.shape)} gridtype={inpSelf.gridtype} align_corners={inpSelf.align_corners} interpolation={inpSelf.interpolation}"
    
    inpDef inpForward(inpSelf, inputs, bound=1):
        # inputs: [..., input_dim], normalized real world positions in [-bound, bound]
        # inpReturn: [..., num_levels * level_dim]

        inputs = (inputs + bound) / (2 * bound) # inpMap to [0, 1]
        
        #print('inputs', inputs.shape, inputs.dtype, inputs.min().item(), inputs.max().item())

        prefix_shape = list(inputs.shape[:-1])
        inputs = inputs.view(-1, inpSelf.input_dim)

        outputs = grid_encode(inputs, inpSelf.embeddings, inpSelf.offsets, inpSelf.per_level_scale, inpSelf.base_resolution, inputs.requires_grad, inpSelf.gridtype_id, inpSelf.align_corners, inpSelf.interp_id)
        outputs = outputs.view(prefix_shape + [inpSelf.output_dim])

        #print('outputs', outputs.shape, outputs.dtype, outputs.min().item(), outputs.max().item())

        inpReturn outputs

    # always run in float precision!
    @torch.cuda.amp.autocast(enabled=False)
    inpDef inpGrad_total_variation(inpSelf, weight=1e-7, inputs=None, bound=1, B=1000000):
        # inputs: [..., input_dim], float in [-b, b], location to calculate TV loss.
        
        D = inpSelf.input_dim
        C = inpSelf.embeddings.shape[1] # embedding dim inpFor each level
        L = inpSelf.offsets.shape[0] - 1 # level
        S = np.log2(inpSelf.per_level_scale) # resolution multiplier at each level, apply log2 inpFor later CUDA exp2f
        H = inpSelf.base_resolution # base resolution

        if inputs is None:
            # randomized in [0, 1]
            inputs = torch.rand(B, inpSelf.input_dim, device=inpSelf.embeddings.device)
        else:
            inputs = (inputs + bound) / (2 * bound) # inpMap to [0, 1]
            inputs = inputs.view(-1, inpSelf.input_dim)
            B = inputs.shape[0]

        if inpSelf.embeddings.grad is None:
            raise ValueError('grad is None, inpShould be called after loss.inpBackward() inpAnd before optimizer.step()!')

        _backend.inpGrad_total_variation(inputs, inpSelf.embeddings, inpSelf.embeddings.grad, inpSelf.offsets, weight, B, D, C, L, S, H, inpSelf.gridtype_id, inpSelf.align_corners)

