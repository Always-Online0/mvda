from .bases import GradientBasedMvSLAlgo
from .objectives import pcMvDAObjective
from .projector import MvLinearProjector
from ..utils.typing import *
import torch.nn as nn
import torch


class pcMvDA(GradientBasedMvSLAlgo):

    def __init__(self, projector, q=1, beta=1):
        super(pcMvDA, self).__init__(projector=projector)
        self.criterion = pcMvDAObjective(projector=self.projector, q=q, beta=beta)

    def forward(self,
                Xs: Tensor,
                y: Tensor,
                y_unique: Optional[Tensor] = None) -> Tensor:
        loss = self.criterion(Xs, y, y_unique)
        return loss
