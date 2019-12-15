from torchsl.sl import *
from torchsl.sl.projector import *
from sklearn.datasets import make_blobs
from sklearn.manifold.t_sne import TSNE
from torchsl.utils import DataVisualizer
from torchsl.grad.optim import SPGD
import numpy as np
import torch


def main():
    dv = DataVisualizer(embed_algo=TSNE)
    X, y = make_blobs(n_features=2, centers=3, n_samples=100, random_state=113)  # 135/116/113
    # y[np.where(y == 2)] = 1

    # from lda_test import X, y
    model = LDA(n_components=1, ep_algo='eigen', kernel='none', n_neighbors=4)
    Y = model.fit_transform(X, y)
    dv.scatter(X, y, title='original')
    dv.scatter(Y, y, title='LDA')

    grad_model = pcLDA(projector=LinearProjector(2, 1))
    optim = SPGD(grad_model.parameters(), lr=0.005)
    losses = []
    for i in range(200):
        optim.zero_grad()
        loss = grad_model(X, y)
        loss.backward()
        optim.step()
        print('[{:03d}]'.format(i + 1), 'Loss:', loss.item())
        losses.append(loss.item())

    print(grad_model.projector.w.t() @ grad_model.projector.w)
    with torch.no_grad():
        Y = grad_model.transform(X)
    dv.scatter(Y, y, title='pcLDA')
    dv.plot(losses, title='pcLDA Losses')
    # dv.show(grids=[(1, 3, 0), (1, 3, 1), (1, 3, 2)], title='LDA')
    dv.show()


if __name__ == '__main__':
    main()
