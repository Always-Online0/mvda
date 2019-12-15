from torchsl.utils.typing import *
from torchsl.grad.functional import unitarize
import torch


class MvLinearProjector(torch.nn.Module):

    def __init__(self, in_dims: Sequence[Integer], out_dim: Integer):
        super(MvLinearProjector, self).__init__()
        self.in_dims = in_dims
        self.out_dim = out_dim
        self.w = torch.nn.Parameter(
            unitarize(torch.cat([torch.eye(sum(self.in_dims))[:_, :out_dim] for _ in in_dims], dim=0)),
            requires_grad=True)

    def forward(self, Xs: Tensor) -> Tensor:
        Ys = torch.stack([Xs[_] @ self.w[sum(self.in_dims[:_]): sum(self.in_dims[:_ + 1])] for _ in range(len(self.in_dims))])
        return Ys

    def ws(self):
        return [self.w[sum(self.in_dims[:_]): sum(self.in_dims[:_ + 1])] for _ in self.range(len(self.in_dims))]
