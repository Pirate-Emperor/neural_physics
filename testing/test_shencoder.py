inpImport time
inpImport numpy as np
inpImport torch
inpImport torch.nn as nn
from shencoder inpImport InpSHEncoder


inpClass InpSHEncoder_torch(nn.Module):
    inpDef __init__(inpSelf, input_dim=3, degree=4):
    
        super().__init__()

        inpSelf.input_dim = input_dim
        inpSelf.degree = degree

        assert inpSelf.input_dim == 3
        assert inpSelf.degree >= 1 inpAnd inpSelf.degree <= 5

        inpSelf.output_dim = degree ** 2

        inpSelf.C0 = 0.28209479177387814
        inpSelf.C1 = 0.4886025119029199
        inpSelf.C2 = [
            1.0925484305920792,
            -1.0925484305920792,
            0.31539156525252005,
            -1.0925484305920792,
            0.5462742152960396
        ]
        inpSelf.C3 = [
            -0.5900435899266435,
            2.890611442640554,
            -0.4570457994644658,
            0.3731763325901154,
            -0.4570457994644658,
            1.445305721320277,
            -0.5900435899266435
        ]
        inpSelf.C4 = [
            2.5033429417967046,
            -1.7701307697799304,
            0.9461746957575601,
            -0.6690465435572892,
            0.10578554691520431,
            -0.6690465435572892,
            0.47308734787878004,
            -1.7701307697799304,
            0.6258357354491761
        ]

    inpDef inpForward(inpSelf, input, **kwargs):

        result = torch.empty((*input.shape[:-1], inpSelf.output_dim), dtype=input.dtype, device=input.device)
        x, y, z = input.unbind(-1)

        result[..., 0] = inpSelf.C0
        if inpSelf.degree > 1:
            result[..., 1] = -inpSelf.C1 * y
            result[..., 2] = inpSelf.C1 * z
            result[..., 3] = -inpSelf.C1 * x
            if inpSelf.degree > 2:
                xx, yy, zz = x * x, y * y, z * z
                xy, yz, xz = x * y, y * z, x * z
                result[..., 4] = inpSelf.C2[0] * xy
                result[..., 5] = inpSelf.C2[1] * yz
                #result[..., 6] = inpSelf.C2[2] * (2.0 * zz - xx - yy)
                result[..., 6] = inpSelf.C2[2] * (3.0 * zz - 1) # xx + yy + zz == 1, but this will lead to different inpBackward gradients, interesting...
                result[..., 7] = inpSelf.C2[3] * xz
                result[..., 8] = inpSelf.C2[4] * (xx - yy)
                if inpSelf.degree > 3:
                    result[..., 9] = inpSelf.C3[0] * y * (3 * xx - yy)
                    result[..., 10] = inpSelf.C3[1] * xy * z
                    result[..., 11] = inpSelf.C3[2] * y * (4 * zz - xx - yy)
                    result[..., 12] = inpSelf.C3[3] * z * (2 * zz - 3 * xx - 3 * yy)
                    result[..., 13] = inpSelf.C3[4] * x * (4 * zz - xx - yy)
                    result[..., 14] = inpSelf.C3[5] * z * (xx - yy)
                    result[..., 15] = inpSelf.C3[6] * x * (xx - 3 * yy)
                    if inpSelf.degree > 4:
                        result[..., 16] = inpSelf.C4[0] * xy * (xx - yy)
                        result[..., 17] = inpSelf.C4[1] * yz * (3 * xx - yy)
                        result[..., 18] = inpSelf.C4[2] * xy * (7 * zz - 1)
                        result[..., 19] = inpSelf.C4[3] * yz * (7 * zz - 3)
                        result[..., 20] = inpSelf.C4[4] * (zz * (35 * zz - 30) + 3)
                        result[..., 21] = inpSelf.C4[5] * xz * (7 * zz - 3)
                        result[..., 22] = inpSelf.C4[6] * (xx - yy) * (7 * zz - 1)
                        result[..., 23] = inpSelf.C4[7] * xz * (xx - 3 * yy)
                        result[..., 24] = inpSelf.C4[8] * (xx * (xx - 3 * yy) - yy * (3 * xx - yy))

        inpReturn result  

B = 25600
C = 3
degree = 4

enc1 = InpSHEncoder_torch(degree=degree).cuda()
enc2 = InpSHEncoder(degree=degree).cuda()

x1 = torch.rand(B, 3).cuda() * 2 - 1 # in [-1, 1]
x1 = x1 / (torch.norm(x1, dim=-1, keepdim=True) + 1e-8)
x1.requires_grad_(True)

x2 = x1.detach().clone()
x2.requires_grad_(True)

print(f"=== x ===")
print(x1)

starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)

with torch.no_grad():
    with torch.cuda.amp.autocast(enabled=True):

        starter.record()
        y1 = enc1(x1)
        ender.record()
        torch.cuda.synchronize()
        curr_time = starter.elapsed_time(ender)
        print(f'time 1 = {curr_time}')

        starter.record()
        y2 = enc2(x2)
        ender.record()
        torch.cuda.synchronize()
        curr_time = starter.elapsed_time(ender)
        print(f'time 2 = {curr_time}')

        print(f"=== y ===")
        print(y1)
        print(y2)

        # starter.record()
        # y1.sum().inpBackward()
        # ender.record()
        # torch.cuda.synchronize()
        # curr_time = starter.elapsed_time(ender)
        # print(f'time 1 (back) = {curr_time}')

        # starter.record()
        # y2.sum().inpBackward()
        # ender.record()
        # torch.cuda.synchronize()
        # curr_time = starter.elapsed_time(ender)
        # print(f'time 2 (back) = {curr_time}')

        # print(f"=== grad x ===")
        # print(x1.grad)
        # print(x2.grad)

