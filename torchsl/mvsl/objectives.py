from .bases import EOBaseMvSLObjective, GradientBasedMvSLObjective
from ..commons import affinity
from ..grad.functional import stiefel_constrain
import torch
import itertools


# ------------------------------------
# MvDA
# ------------------------------------
class MvDAIntraScatter(EOBaseMvSLObjective):

    def __init__(self):
        super(MvDAIntraScatter, self).__init__(predicate='minimize')

    def _O_(self):
        W = torch.zeros(self.n_samples, self.n_samples)
        for ci in self._y_unique:
            W += self.ecs[ci].unsqueeze(0).t() @ self.ecs[ci].unsqueeze(0) / (torch.sum(self.ecs[ci]) * self.n_views)
        I = torch.eye(self.n_samples)

        def constructor(self, j, r):
            if j == r:
                s_jr = I - W
            else:
                s_jr = -W
            return self._Xs[j].t() @ s_jr @ self._Xs[r]
        return self._construct_mv_matrix(constructor)


class MvDAInterScatter(EOBaseMvSLObjective):

    def __init__(self):
        super(MvDAInterScatter, self).__init__(predicate='maximize')

    def _O_(self):
        n = self.n_views * self.n_samples
        W = torch.zeros(self.n_samples, self.n_samples)
        for ci in self._y_unique:
            W += self.ecs[ci].unsqueeze(0).t() @ self.ecs[ci].unsqueeze(0) / (torch.sum(self.ecs[ci]) * len(self._Xs))
        B = torch.ones(self.n_samples, self.n_samples) / n

        def constructor(self, j, r):
            return self._Xs[j].t() @ (W - B) @ self._Xs[r]
        return self._construct_mv_matrix(constructor)


# ------------------------------------
# MvDA-vc
# ------------------------------------
class ViewConsistency(EOBaseMvSLObjective):

    def __init__(self, reg='auto'):
        super(ViewConsistency, self).__init__(predicate='minimize', reg=reg)

    def _O_(self):
        S_cols = []
        for j in range(self.n_views):
            S_rows = []
            for r in range(self.n_views):
                if j == r:
                    k_j = self._Xs[j] @ self._Xs[j].t() if not self.kernels.statuses[j] else self._Xs[j]

                    # vc_jj = self._Xs[j] @ self._Xs[j].t() @ self._Xs[j] @ self._Xs[j].t()
                    vc_jj = self._regularize(k_j @ k_j)
                    vc_jr = 2 * self.n_views * vc_jj.inverse() - 2 * vc_jj.inverse()
                else:
                    k_j = self._Xs[j] @ self._Xs[j].t() if not self.kernels.statuses[j] else self._Xs[j]
                    k_r = self._Xs[r] @ self._Xs[r].t() if not self.kernels.statuses[r] else self._Xs[r]

                    # vc_jr = self._Xs[r] @ self._Xs[r].t() @ self._Xs[j] @ self._Xs[j].t()
                    vc_jr = self._regularize(k_r @ k_j)
                    vc_jr = -2 * vc_jr.inverse()

                s_jr = self._Xs[j].t() @ vc_jr @ self._Xs[r]
                S_rows.append(s_jr)
            S_cols.append(torch.cat(S_rows, dim=1))
        return torch.cat(S_cols, dim=0)


# ------------------------------------
# pcMvDA
# ------------------------------------
class pcMvDAObjective(GradientBasedMvSLObjective):

    def __init__(self, projector, q=1, beta=1):
        super(pcMvDAObjective, self).__init__(projector=projector)
        self.q = q
        self.beta = beta

    def forward(self, Xs, y, y_unique=None):
        cls_W = torch.zeros(self.n_classes, self.n_samples, self.n_samples)
        cls_I = torch.zeros(self.n_classes, self.n_samples, self.n_samples)
        cls_n_samples = [self.ecs[ci].sum() for ci in self._y_unique]
        W = torch.zeros(self.n_samples, self.n_samples)
        for ci in self._y_unique:
            cls_W[ci] = self.ecs[ci].unsqueeze(0).t() @ self.ecs[ci].unsqueeze(0) / (cls_n_samples[ci] * self.n_views)
            cls_I[ci] = torch.eye(self.n_samples) * self.ecs[ci]
        W = cls_W.sum(dim=0)
        I = torch.eye(self.n_samples)
        cls_Sw = [
            self._construct_mv_matrix(lambda self, j, r: self._Xs[j].t() @ (cls_I[ci] - cls_W[ci]) @ self._Xs[r] if j == r else self._Xs[j].t() @ (-cls_W[ci]) @ self._Xs[r])
            for ci in self._y_unique
        ]
        Sw = self._construct_mv_matrix(lambda self, j, r: self._Xs[j].t() @ (I - W) @ self._Xs[r] if j == r else self._Xs[j].t() @ (-W) @ self._Xs[r])

        # G = stiefel_constrain(self.projector.w)
        G = self.projector.w
        pcs = sorted(list(itertools.combinations(self._y_unique, r=2)))
        pc_Sw = [self.beta * (cls_n_samples[a] * cls_Sw[a] + cls_n_samples[b] * cls_Sw[b]) / (cls_n_samples[a] + cls_n_samples[b]) + (1 - self.beta) * Sw
                 for a, b in pcs]
        u_total = sum([G[sum(self.dims[:_]): sum(self.dims[:_ + 1])].t() @ self._Xs[_].t() for _ in range(self.n_views)])
        pc_du = [u_total @ self.ecs[a].unsqueeze(0).t() / (cls_n_samples[a] * self.n_views)
                 - u_total @ self.ecs[b].unsqueeze(0).t() / (cls_n_samples[b] * self.n_views)
                 for a, b in pcs]
        # return sum([cls_n_samples[a] * cls_n_samples[b] * (pc_du[i].t() @ self._regularize(G.t() @ pc_Sw[i] @ G).inverse() @ pc_du[i]) ** -self.q for i, (a, b) in enumerate(pcs)])
        return sum([cls_n_samples[a] * cls_n_samples[b] * (torch.trace(pc_du[i] @ pc_du[i].t()) / torch.trace(G.t() @ pc_Sw[i] @ G)) ** -self.q for i, (a, b) in enumerate(pcs)])


# ------------------------------------
# Class Separating
# ------------------------------------
class ClassSeparating(EOBaseMvSLObjective):

    def __init__(self):
        super(ClassSeparating, self).__init__(predicate='maximize')

    def _O_(self):
        W = torch.zeros(self.n_samples, self.n_samples)
        for ca in self._y_unique:
            for cb in self._y_unique:
                W += torch.sum(self.ecs[cb]) / torch.sum(self.ecs[ca]) * self.ecs[ca].unsqueeze(0).t() @ self.ecs[ca].unsqueeze(0)
        D = torch.ones(self.n_samples, self.n_samples)

        S_cols = []
        for j in range(self.n_views):
            S_rows = []
            for r in range(self.n_views):
                vc_jr = 2 * W - 2 * D

                s_jr = self._Xs[j].t() @ vc_jr @ self._Xs[r]
                S_rows.append(s_jr)
            S_cols.append(torch.cat(S_rows, dim=1))
        return torch.cat(S_cols, dim=0)


# ------------------------------------
# MvLFDA
# ------------------------------------
class MvLFDAIntraScatter(EOBaseMvSLObjective):

    def __init__(self,
                 affinity_type='lle',
                 n_neighbors=5,
                 epsilon='auto',
                 affinity_kernel='rbf',
                 gamma=1,
                 lambda_lc=1.0):
        super(MvLFDAIntraScatter, self).__init__(predicate='minimize')
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
            W += self.ecs[ci].unsqueeze(0).t() @ self.ecs[ci].unsqueeze(0) / (torch.sum(self.ecs[ci]) * self.n_views)
        D = torch.eye(self.n_samples)
        # W = torch.zeros(self.n_samples, self.n_samples)
        # for ci in self._y_unique:
        #     W += self.ecs[ci].unsqueeze(0).t() @ self.ecs[ci].unsqueeze(0)
        # D = torch.eye(self.n_samples)

        S_cols = []
        for j in range(self.n_views):
            S_rows = []
            for r in range(self.n_views):
                W_af = self.__localize__(W, j)
                if j == r:
                    s_jr = D - W_af
                else:
                    s_jr = -W
                s_jr = self._Xs[j].t() @ s_jr @ self._Xs[r]
                S_rows.append(s_jr)
            S_cols.append(torch.cat(S_rows, dim=1))
        return torch.cat(S_cols, dim=0)

    def __localize__(self, W, j):
        return (1.0 - self.lambda_lc) * W + self.lambda_lc * affinity(self._Xs[j], **self.affinity_params)


class MvLFDAInterScatter(EOBaseMvSLObjective):

    def __init__(self,
                 affinity_type='lle',
                 n_neighbors=5,
                 epsilon='auto',
                 affinity_kernel='rbf',
                 gamma=1,
                 lambda_lc=1.0):
        super(MvLFDAInterScatter, self).__init__(predicate='maximize')
        self.affinity_params = {
            'algo': affinity_type,
            'n_neighbors': n_neighbors,
            'epsilon': epsilon,
            'kernel': affinity_kernel,
            'gamma': gamma
        }
        self.lambda_lc = lambda_lc

    def _O_(self):
        n = self.n_views * self.n_samples
        W = torch.zeros(self.n_samples, self.n_samples)
        for ci in self._y_unique:
            W += self.ecs[ci].unsqueeze(0).t() @ self.ecs[ci].unsqueeze(0) / (torch.sum(self.ecs[ci]) * len(self._Xs))
        B = torch.ones(self.n_samples, self.n_samples) / n
        # n = self.n_views * self.n_samples
        # W = torch.zeros(self.n_samples, self.n_samples)
        # for ci in self._y_unique:
        #     W += self.ecs[ci].unsqueeze(0).t() @ self.ecs[ci].unsqueeze(0)
        # B = torch.ones(self.n_samples, self.n_samples) / n

        S_cols = []
        for j in range(self.n_views):
            S_rows = []
            for r in range(self.n_views):
                W_af = self.__localize__(W, j)
                s_jr = self._Xs[j].t() @ (W_af - B) @ self._Xs[r]
                S_rows.append(s_jr)
            S_cols.append(torch.cat(S_rows, dim=1))
        return torch.cat(S_cols, dim=0)

    def __localize__(self, W, j):
        return (1.0 - self.lambda_lc) * W + self.lambda_lc * affinity(self._Xs[j], **self.affinity_params)


# ------------------------------------
# MvCCDA
# ------------------------------------
class CommonComponent(EOBaseMvSLObjective):

    def __init__(self):
        super(CommonComponent, self).__init__(predicate='minimize')

    def _O_(self):
        W = torch.zeros(self.n_samples, self.n_samples)
        for ci in self._y_unique:
            W += self.ecs[ci].unsqueeze(0).t() @ self.ecs[ci].unsqueeze(0) / (torch.sum(self.ecs[ci]) * self.n_views)

        I = torch.eye(self.n_samples)
        D = torch.ones(self.n_samples, self.n_samples)

        S_cols = []
        for j in range(self.n_views):
            S_rows = []
            for r in range(self.n_views):
                if j == r:
                    s_jr = 2 / self.n_views * (I - D)
                else:
                    s_jr = 2 / self.n_views * D
                s_jr = self._Xs[j].t() @ s_jr @ self._Xs[r]
                S_rows.append(s_jr)
            S_cols.append(torch.cat(S_rows, dim=1))
        return torch.cat(S_cols, dim=0)


class DifferingClass(EOBaseMvSLObjective):

    def __init__(self):
        super(DifferingClass, self).__init__(predicate='maximize')

    def _O_(self):
        I = torch.eye(self.n_samples) * self.n_views
        E = torch.zeros(self.n_samples, self.n_samples)
        for ca in self._y_unique:
            for cb in self._y_unique:
                if ca != cb:
                    E += self.ecs[ca].unsqueeze(0).t() @ self.ecs[cb].unsqueeze(0)

        S_cols = []
        for j in range(self.n_views):
            S_rows = []
            for r in range(self.n_views):
                if j == r:
                    s_jr = 2 * (I - E)
                else:
                    s_jr = 2 * -E
                s_jr = self._Xs[j].t() @ s_jr @ self._Xs[r]
                S_rows.append(s_jr)
            S_cols.append(torch.cat(S_rows, dim=1))
        return torch.cat(S_cols, dim=0)


# ------------------------------------
# Regularizer
# ------------------------------------
class Regularization(EOBaseMvSLObjective):

    def __init__(self):
        super(Regularization, self).__init__(predicate='minimize')

    def _O_(self):
        return torch.eye(sum(self.dims))
