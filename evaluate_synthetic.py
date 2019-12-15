from torchsl.mvsl import *
from torchsl.mvsl.projector import MvLinearProjector
from torchsl.utils import DataVisualizer
from torchsl.grad.optim import SPGD
from sklearn.manifold.t_sne import TSNE
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from dataset.utils import multiview_train_test_split, join_multiview_datasets
import numpy as np
import torch
import synthetics


def eval_multiview_model(mvmodel, clf, Xs_train, y_train, Xs_test, y_test, return_projected=False):
    Ys_train = mvmodel.fit_transform(Xs_train, y_train)
    Ys_test = mvmodel.transform(Xs_test)
    # Classify
    mv_scores = np.zeros((n_views, n_views))
    for view_train in range(n_views):
        for view_test in range(n_views):
            X_train = Ys_train[view_train]
            X_test = Ys_test[view_test]
            clf.fit(X_train, y_train)
            y_pred = clf.predict(X_test)
            score = accuracy_score(y_test, y_pred)
            mv_scores[view_train, view_test] = score
    if return_projected:
        return mv_scores, Ys_train, Ys_test
    return mv_scores


if __name__ == '__main__':
    visualize = True
    Xs, y = synthetics.single_blob_dataset(n_classes=5, n_views=3, n_features=3, seed=107)  # 107
    dv = DataVisualizer(embed_algo=TSNE(n_components=3), embed_style='global', legend=True)
    n_views = len(Xs)
    Xs_train, y_train, Xs_test, y_test = multiview_train_test_split(Xs, y)
    print(len(y_train), len(y_test))

    Xs_all, y_all = join_multiview_datasets([Xs_train, Xs_test], [y_train, y_test])
    dv.mv_scatter(Xs_all, y_all, title='Original space')

    mvmodel = MvDA(n_components=2, ep_algo='eigen', kernels='linear')
    Ys_train = mvmodel.fit_transform(Xs_train, y_train)
    Ys_test = mvmodel.transform(Xs_test)
    # mv_scores1, Ys_train, Ys_test = eval_multiview_model(mvmodel=mvmodel,
    #                                                      clf=KNeighborsClassifier(),
    #                                                      Xs_train=Xs_train, y_train=y_train,
    #                                                      Xs_test=Xs_test, y_test=y_test,
    #                                                      return_projected=True)
    # print('Projected space MvDA', mv_scores1, sep='\n', end='\n\n')
    Ys_all = join_multiview_datasets([Ys_train, Ys_test])
    dv.mv_scatter(Ys_all, y_all, title='Projected space MvDA')

    grad_mvmodel = pcMvDA(projector=MvLinearProjector([3, 3, 3], 2))
    optim = SPGD(grad_mvmodel.parameters(), lr=0.01)
    losses = []
    for i in range(300):
        optim.zero_grad()
        loss = grad_mvmodel(Xs, y)
        loss.backward()
        optim.step()
        print('[{:03d}]'.format(i + 1), 'Loss:', loss.item())
        losses.append(loss.item())

    # print(grad_mvmodel.projector.w.t() @ grad_mvmodel.projector.w)
    # grad_mvmodel.projector.w.data.copy_(unitarize(grad_mvmodel.projector.w))
    # print(grad_mvmodel.projector.w.t() @ grad_mvmodel.projector.w)
    with torch.no_grad():
        Ys_train = grad_mvmodel.transform(Xs_train)
        Ys_test = grad_mvmodel.transform(Xs_test)
    Ys_all = join_multiview_datasets([Ys_train, Ys_test])
    dv.mv_scatter(Ys_all, y_all, title='Projected space pcMvDA')
    dv.plot(losses, title='pcMvDA Losses')

    # mv_scores2, Ys_train, Ys_test = eval_multiview_model(mvmodel=MvCSDA(n_components=2, ep='eig', kernels='linear'),
    #                                                      clf=KNeighborsClassifier(),
    #                                                      Xs_train=Xs_train, y_train=y_train,
    #                                                      Xs_test=Xs_test, y_test=y_test,
    #                                                      return_projected=True)
    # print('Projected space MvCSDA', mv_scores2, sep='\n', end='\n\n')
    # Ys_all = join_multiview_datasets([Ys_train, Ys_test])
    # dv.mv_scatter(Ys_all, y_all, title='Projected space MvCSDA')
    #
    # mv_scores3, Ys_train, Ys_test = eval_multiview_model(mvmodel=MvLFDA(n_components=2, ep='eig', kernels='linear', lambda_lc=0.05),
    #                                                      clf=KNeighborsClassifier(),
    #                                                      Xs_train=Xs_train, y_train=y_train,
    #                                                      Xs_test=Xs_test, y_test=y_test,
    #                                                      return_projected=True)
    # print('Projected space MvLFDA', mv_scores3, sep='\n', end='\n\n')
    # Ys_all = join_multiview_datasets([Ys_train, Ys_test])
    # dv.mv_scatter(Ys_all, y_all, title='Projected space MvLFDA')

    # np.savetxt("gesturefair_mvda_4096.csv", mv_scores, delimiter=",")
    # print(mv_scores2 == mv_scores1)
    # print(mv_scores3 >= mv_scores1)
    dv.show()
