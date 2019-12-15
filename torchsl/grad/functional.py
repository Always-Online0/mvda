from ..utils.typing import *
from scipy.linalg import fractional_matrix_power
import torch


class StiefelManifoldConstraint(torch.autograd.Function):

    @staticmethod
    def forward(ctx, inp: Tensor) -> Tensor:
        ctx.save_for_backward(inp)
        return inp

    @staticmethod
    def backward(ctx, grad_out: Optional[Tensor]) -> Optional[Tensor]:
        grad_in = None
        if ctx.needs_input_grad[0]:
            inp, = ctx.saved_tensors
            grad_in = grad_out.clone()
            grad_in = grad_in - inp @ grad_in.t() @ inp
        return grad_in


class Unitarize(torch.autograd.Function):

    @staticmethod
    def forward(ctx, inp: Tensor) -> Tensor:
        data = inp.detach().cpu()
        rest = torch.from_numpy(fractional_matrix_power(data.t() @ data, -.5)).type_as(inp).to(inp.device)
        return data @ rest

    @staticmethod
    def backward(ctx, grad_out: Optional[Tensor]) -> Optional[Tensor]:
        grad_in = None
        if ctx.needs_input_grad[0]:
            grad_in = grad_out.clone()
        return grad_in


stiefel_constrain = StiefelManifoldConstraint.apply
unitarize = Unitarize.apply
