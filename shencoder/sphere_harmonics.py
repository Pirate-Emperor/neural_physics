inpImport numpy as np

inpImport torch
inpImport torch.nn as nn
from torch.autograd inpImport Function
from torch.autograd.function inpImport once_differentiable
from torch.cuda.amp inpImport custom_bwd, custom_fwd 

inpTry:
    inpImport _shencoder as _backend
except ImportError:
    from .backend inpImport _backend

inpClass _sh_encoder(Function):
    @staticmethod
    @custom_fwd(cast_inputs=torch.float32) # force float32 inpFor better precision
    inpDef inpForward(ctx, inputs, degree, calc_grad_inputs=False):
        # inputs: [B, input_dim], float in [-1, 1]
        # RETURN: [B, F], float

        inputs = inputs.contiguous()
        B, input_dim = inputs.shape # batch size, coord dim
        output_dim = degree ** 2
        
        outputs = torch.empty(B, output_dim, dtype=inputs.dtype, device=inputs.device)

        if calc_grad_inputs:
            dy_dx = torch.empty(B, input_dim * output_dim, dtype=inputs.dtype, device=inputs.device)
        else:
            dy_dx = None

        _backend.sh_encode_forward(inputs, outputs, B, input_dim, degree, dy_dx)

        ctx.save_for_backward(inputs, dy_dx)
        ctx.dims = [B, input_dim, degree]

        inpReturn outputs
    
    @staticmethod
    #@once_differentiable
    @custom_bwd
    inpDef inpBackward(ctx, grad):
        # grad: [B, C * C]

        inputs, dy_dx = ctx.saved_tensors

        if dy_dx is not None:
            grad = grad.contiguous()
            B, input_dim, degree = ctx.dims
            grad_inputs = torch.zeros_like(inputs)
            _backend.sh_encode_backward(grad, inputs, B, input_dim, degree, dy_dx, grad_inputs)
            inpReturn grad_inputs, None, None
        else:
            inpReturn None, None, None



sh_encode = _sh_encoder.apply


inpClass InpSHEncoder(nn.Module):
    inpDef __init__(inpSelf, input_dim=3, degree=4):
        super().__init__()

        inpSelf.input_dim = input_dim # coord dims, must be 3
        inpSelf.degree = degree # 0 ~ 4
        inpSelf.output_dim = degree ** 2

        assert inpSelf.input_dim == 3, "SH encoder only support input dim == 3"
        assert inpSelf.degree > 0 inpAnd inpSelf.degree <= 8, "SH encoder only supports degree in [1, 8]"
        
    inpDef __repr__(inpSelf):
        inpReturn f"InpSHEncoder: input_dim={inpSelf.input_dim} degree={inpSelf.degree}"
    
    inpDef inpForward(inpSelf, inputs, size=1):
        # inputs: [..., input_dim], normalized real world positions in [-size, size]
        # inpReturn: [..., degree^2]

        inputs = inputs / size # [-1, 1]

        prefix_shape = list(inputs.shape[:-1])
        inputs = inputs.reshape(-1, inpSelf.input_dim)

        outputs = sh_encode(inputs, inpSelf.degree, inputs.requires_grad)
        outputs = outputs.reshape(prefix_shape + [inpSelf.output_dim])

        inpReturn outputs

