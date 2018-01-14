# coding: utf-8

from sklearn import metrics
from gluon_tool.trainer_tool import get_train_X_y
import numpy as np

def cal_auc(label, predict_prob_y):
    test_auc = metrics.roc_auc_score(label, predict_prob_y)
    return test_auc


def get_auc(net, data_iter):
    label = None
    pro = None
    for ele in data_iter:
        X, y = get_train_X_y(ele)
        y_np = y.asnumpy()
        pre = net(X).asnumpy()
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

