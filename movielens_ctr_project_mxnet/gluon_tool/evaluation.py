# coding: utf-8

from sklearn import metrics
from gluon_tool.trainer_tool import get_train_X_y
import numpy as np
import mxnet as mx


def cal_auc(label, predict_prob_y):
    test_auc = metrics.roc_auc_score(label, predict_prob_y)
    return test_auc


def get_auc(net, data_iter, ctx=mx.cpu()):
    from gluon_tool.trainer_tool import reset
    reset(data_iter)
    label = None
    pro = None
    for ele in data_iter:
        X, y = get_train_X_y(ele)
        X = X.as_in_context(context=ctx)
        y_np = y.reshape((-1,)).asnumpy()
        pre = net(X).reshape((-1,)).asnumpy()
        if pro is None:
            pro = pre
        else:
            pro = np.append(pro, pre)

        if label is None:
            label = y_np
        else:
            label = np.append(label, y_np)
    t = cal_auc(label, pro)
    return t


def get_auc_soft_max(my_net, data_iter, data_iter_length):
    pass

