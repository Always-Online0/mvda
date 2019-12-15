from torchsl.utils.typing import *
from torchsl.grad.functional import unitarize
import torch


class LinearProjector(torch.nn.Module):

    def __init__(self, in_dim: Integer, out_dim: Integer):
        super(LinearProjector, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.w = torch.nn.Parameter(unitarize(torch.eye(in_dim, in_dim)[:, :out_dim]), requires_grad=True)

    def forward(self, X: Tensor) -> Tensor:
        Y = X @ self.w
        return Y
