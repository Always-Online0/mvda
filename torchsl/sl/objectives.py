from .bases import EOBasedSLObjective, GradientBasedSLObjective
from ..commons import affinity
import torch
import itertools


# ------------------------------------
# LDA
# ------------------------------------
class LDAIntraScatter(EOBasedSLObjective):

    def __init__(self):
        super(LDAIntraScatter, self).__init__(predicate='minimize')

    def _O_(self):
        W = torch.zeros(self.n_samples, self.n_samples)
        for ci in self._y_unique:
            W += self.ecs[ci].unsqueeze(0).t() @ self.ecs[ci].unsqueeze(0) / torch.sum(self.ecs[ci])
        I = torch.eye(self.n_samples)
        return self._X.t() @ (I - W) @ self._X


class LDAInterScatter(EOBasedSLObjective):

    def __init__(self):
        super(LDAInterScatter, self).__init__(predicate='maximize')

    def _O_(self):
        W = torch.zeros(self.n_samples, self.n_samples)
        for ci in self._y_unique:
            W += self.ecs[ci].unsqueeze(0).t() @ self.ecs[ci].unsqueeze(0) / torch.sum(self.ecs[ci])
        B = torch.ones(self.n_samples, self.n_samples) / self.n_samples
        return self._X.t() @ (W - B) @ self._X


# ------------------------------------
# pcLDA
# ------------------------------------
class pcLDAObjective(GradientBasedSLObjective):

    def __init__(self, projector, q=1, beta=1):
        super(pcLDAObjective, self).__init__(projector=projector)
        # self.sw_lda = LDAIntraScatter()
        self.q = q
        self.beta = beta

    def forward(self, X, y, y_unique=None):
        projected_X = self.projector(self._X)
        cls_W = torch.zeros(self.n_classes, self.n_samples, self.n_samples)
        cls_I = torch.zeros(self.n_classes, self.n_samples, self.n_samples)
        cls_n_samples = [self.ecs[ci].sum() for ci in self._y_unique]
        for ci in self._y_unique:
            cls_W[ci] = self.ecs[ci].unsqueeze(0).t() @ self.ecs[ci].unsqueeze(0) / cls_n_samples[ci]
            cls_I[ci] = torch.eye(self.n_samples) * self.ecs[ci]
        W = cls_W.sum(dim=0)
        I = torch.eye(self.n_samples)
        cls_Sw = [projected_X.t() @ (cls_I[ci] - cls_W[ci]) @ projected_X for ci in self._y_unique]
        Sw = projected_X.t() @ (I - W) @ projected_X

        pcs = sorted(list(itertools.combinations(self._y_unique, r=2)))
        pc_Sw = [self.beta * (cls_n_samples[a] * cls_Sw[a] + cls_n_samples[b] * cls_Sw[b]) / (cls_n_samples[a] + cls_n_samples[b]) + (1 - self.beta) * Sw
                 for a, b in pcs]
        pc_du = [projected_X.t() @ self.ecs[a].unsqueeze(0).t() / cls_n_samples[a]
                 - projected_X.t() @ self.ecs[b].unsqueeze(0).t() / cls_n_samples[b]
                 for a, b in pcs]
        # return sum([cls_n_samples[a] * cls_n_samples[b] * (pc_du[i].t() @ self._regularize(G.t() @ pc_Sw[i] @ G).inverse() @ pc_du[i]) ** -self.q for i, (a, b) in enumerate(pcs)])
        return sum([cls_n_samples[a] * cls_n_samples[b] * (torch.trace(pc_du[i] @ pc_du[i].t()) / torch.trace(pc_Sw[i])) ** -self.q for i, (a, b) in enumerate(pcs)])


class CenterLDAObjective(GradientBasedSLObjective):

    def __init__(self, projector, q=1, beta=1):
        super(CenterLDAObjective, self).__init__(projector=projector)
        self.centers = torch.nn.Parameter(
            torch.eye(3, 2) * 100, requires_grad=True
        )
        self.initialized = False
        __import__('warnings').warn('Experimental', UserWarning)

    def forward(self, X, y, y_unique=None):
        # W = torch.zeros(self.projector.out_dim, self.projector.out_dim)
        # for ci in self._y_unique:
        #     W += torch.sum(self.ecs[ci]) * (self.centers[ci].unsqueeze(0).t() @ self.centers[ci].unsqueeze(0))
        # I = torch.eye(self.n_samples)
        # B = self.n_samples * (self.centers.mean(dim=0).unsqueeze(0).t() @ self.centers.mean(dim=0).unsqueeze(0))
        # Sw = self.projector.w.t() @ self._X.t() @ I @ self._X @ self.projector.w - W
        # Sb = W - B
        projected_X = self.projector(self._X)
        global_center = projected_X.mean(dim=0)
        # if not self.initialized:
        #     with torch.no_grad():
        #         us = torch.stack([torch.mean(self.ecs[ci].unsqueeze(0) @ projected_X, dim=0) for ci in self._y_unique])
        #         self.centers.data.copy_(us)
        #     self.initialized = True

        Sw = sum([sum([(x - self.centers[ci]).unsqueeze(0).t() @ (x - self.centers[ci]).unsqueeze(0) for x in projected_X[torch.where(self.ecs[ci] == 1)]]) for ci in self._y_unique])
        Sb = sum([torch.sum(self.ecs[ci]) * (self.centers[ci] - global_center).unsqueeze(0).t() @ (self.centers[ci] - global_center).unsqueeze(0) for ci in self._y_unique])
        return Sw.trace() / Sb.trace()
        return sum([torch.dist(self.ecs[ci].unsqueeze(0) @ self._X @ G / torch.sum(self.ecs[ci]), self.centers[ci]) for ci in self._y_unique])


# ------------------------------------
# LFDA
# ------------------------------------
class LFDAIntraScatter(EOBasedSLObjective):

    def __init__(self,
                 affinity_type='kernel',
                 n_neighbors=5,
                 epsilon='auto',
                 affinity_kernel='rbf',
                 gamma=1):
        super(LFDAIntraScatter, self).__init__(predicate='minimize')
        self.affinity_params = {
            'algo': affinity_type,
            'n_neighbors': n_neighbors,
            'epsilon': epsilon,
            'kernel': affinity_kernel,
            'gamma': gamma,
            'row_norm': False
        }

    def _O_(self):
        W = torch.zeros(self.n_samples, self.n_samples)
        for ci in self._y_unique:
            W += self.ecs[ci].unsqueeze(0).t() @ self.ecs[ci].unsqueeze(0) / torch.sum(self.ecs[ci])
        A = self.__localize__(W)
        print(A)
        D = A.sum(dim=1).diag()
        return self._X.t() @ (D - A) @ self._X

    def __localize__(self, W):
        return W * affinity(self._X, **self.affinity_params)


class LFDAInterScatter(EOBasedSLObjective):

    def __init__(self,
                 affinity_type='kernel',
                 n_neighbors=5,
                 epsilon='auto',
                 affinity_kernel='rbf',
                 gamma=1):
        super(LFDAInterScatter, self).__init__(predicate='maximize')
        self.affinity_params = {
            'algo': affinity_type,
            'n_neighbors': n_neighbors,
            'epsilon': epsilon,
            'kernel': affinity_kernel,
            'gamma': gamma,
            'row_norm': False
        }

    def _O_(self):
        W = torch.zeros(self.n_samples, self.n_samples)
        for ci in self._y_unique:
            W += self.ecs[ci].unsqueeze(0).t() @ self.ecs[ci].unsqueeze(0) / torch.sum(self.ecs[ci])
        W = torch.ones(self.n_samples, self.n_samples) / self.n_samples - W
        A = self.__localize__(W)
        D = A.sum(dim=1).diag()
        return self._X.t() @ (D - A) @ self._X

    def __localize__(self, W):
        return W * affinity(self._X, **self.affinity_params)


# ------------------------------------
# LFDA with Locally Linear Embedding
# ------------------------------------
class LFDALLEIntraScatter(EOBasedSLObjective):

    def __init__(self,
                 affinity_type='lle',
                 n_neighbors=5,
                 epsilon='auto',
                 affinity_kernel='rbf',
                 gamma=1,
                 lambda_lc=1.0):
        super(LFDALLEIntraScatter, self).__init__(predicate='minimize')
        self.affinity_params = {
            'algo': affinity_type,
            'n_neighbors': n_neighbors,
            'epsilon': epsilon,
            'kernel': affinity_kernel,
            'gamma': gamma
        }
        self.lambda_lc = lambda_lc

    def _O_(self):
        W = torch.zeros(self.n_samples, self.n_samples)
        for ci in self._y_unique:
            W += self.ecs[ci].unsqueeze(0).t() @ self.ecs[ci].unsqueeze(0) / torch.sum(self.ecs[ci])
        B = torch.ones(self.n_samples, self.n_samples) / self.n_samples
        return self._X.t() @ (self.__localize__(W) - B) @ self._X

    def __localize__(self, W):
        return (1.0 - self.lambda_lc) * W + self.lambda_lc * affinity(self._X, **self.affinity_params)


class LFDALLEInterScatter(EOBasedSLObjective):

    def __init__(self,
                 affinity_type='lle',
                 n_neighbors=5,
                 epsilon='auto',
                 affinity_kernel='rbf',
                 gamma=1,
                 lambda_lc=1.0):
        super(LFDALLEInterScatter, self).__init__(predicate='maximize')
        self.affinity_params = {
            'algo': affinity_type,
            'n_neighbors': n_neighbors,
            'epsilon': epsilon,
            'kernel': affinity_kernel,
            'gamma': gamma
        }
        self.lambda_lc = lambda_lc

    def _O_(self):
        W = torch.zeros(self.n_samples, self.n_samples)
        for ci in self._y_unique:
            W += self.ecs[ci].unsqueeze(0).t() @ self.ecs[ci].unsqueeze(0) / torch.sum(self.ecs[ci])
        B = torch.ones(self.n_samples, self.n_samples) / self.n_samples
        return self._X.t() @ (self.__localize__(W) - B) @ self._X

    def __localize__(self, W):
        return (1.0 - self.lambda_lc) * W + self.lambda_lc * affinity(self._X, **self.affinity_params)
