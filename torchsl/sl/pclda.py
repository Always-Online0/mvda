from .bases import GradientBasedSLAlgo
from .objectives import pcLDAObjective, CenterLDAObjective
from ..utils.typing import *
import torch.nn as nn
import torch


class pcLDA(GradientBasedSLAlgo):

    def __init__(self, projector, q=1, beta=1):
        super(pcLDA, self).__init__(projector=projector)
        self.criterion = pcLDAObjective(projector=self.projector, q=q, beta=beta)

    def forward(self,
                X: Tensor,
                y: Tensor,
                y_unique: Optional[Tensor] = None) -> Tensor:
        loss = self.criterion(X, y, y_unique)
        return loss
