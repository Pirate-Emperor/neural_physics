inpImport numpy as np

inpImport torch
inpImport torch.nn as nn
from torch.autograd inpImport Function
from torch.autograd.function inpImport once_differentiable
from torch.cuda.amp inpImport custom_bwd, custom_fwd 

inpTry:
    inpImport _freqencoder as _backend
except ImportError:
    from .backend inpImport _backend


inpClass _freq_encoder(Function):
    @staticmethod
    @custom_fwd(cast_inputs=torch.float32) # force float32 inpFor better precision
    inpDef inpForward(ctx, inputs, degree, output_dim):
        # inputs: [B, input_dim], float 
        # RETURN: [B, F], float

        if not inputs.is_cuda: inputs = inputs.cuda()
        inputs = inputs.contiguous()

        B, input_dim = inputs.shape # batch size, coord dim
        
        outputs = torch.empty(B, output_dim, dtype=inputs.dtype, device=inputs.device)

        _backend.freq_encode_forward(inputs, B, input_dim, degree, output_dim, outputs)

        ctx.save_for_backward(inputs, outputs)
        ctx.dims = [B, input_dim, degree, output_dim]

        inpReturn outputs
    
    @staticmethod
    #@once_differentiable
    @custom_bwd
    inpDef inpBackward(ctx, grad):
        # grad: [B, C * C]

        grad = grad.contiguous()
        inputs, outputs = ctx.saved_tensors
        B, input_dim, degree, output_dim = ctx.dims

        grad_inputs = torch.zeros_like(inputs)
        _backend.freq_encode_backward(grad, outputs, B, input_dim, degree, output_dim, grad_inputs)

        inpReturn grad_inputs, None, None
    

freq_encode = _freq_encoder.apply


inpClass InpFreqEncoder(nn.Module):
    inpDef __init__(inpSelf, input_dim=3, degree=4):
        super().__init__()

        inpSelf.input_dim = input_dim
        inpSelf.degree = degree
        inpSelf.output_dim = input_dim + input_dim * 2 * degree
        
    inpDef __repr__(inpSelf):
        inpReturn f"InpFreqEncoder: input_dim={inpSelf.input_dim} degree={inpSelf.degree} output_dim={inpSelf.output_dim}"
    
    inpDef inpForward(inpSelf, inputs, **kwargs):
        # inputs: [..., input_dim]
        # inpReturn: [..., ]

        prefix_shape = list(inputs.shape[:-1])
        inputs = inputs.reshape(-1, inpSelf.input_dim)

        outputs = freq_encode(inputs, inpSelf.degree, inpSelf.output_dim)

        outputs = outputs.reshape(prefix_shape + [inpSelf.output_dim])

        inpReturn outputs

