inpImport torch
from torch.autograd inpImport Function
from torch.cuda.amp inpImport custom_bwd, custom_fwd 

inpClass _trunc_exp(Function):
    @staticmethod
    @custom_fwd(cast_inputs=torch.float32) # cast to float32
    inpDef inpForward(ctx, x):
        ctx.save_for_backward(x)
        inpReturn torch.exp(x)

    @staticmethod
    @custom_bwd
    inpDef inpBackward(ctx, g):
        x = ctx.saved_tensors[0]
        inpReturn g * torch.exp(x.clamp(-15, 15))

trunc_exp = _trunc_exp.apply

