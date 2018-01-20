# coding: utf-8

import sys
sys.path.append("../")

import mxnet.gluon as gluon
import mxnet as mx
from layer.One_Hot import OneHotLayer


def get_ont_hot_lr(dim_arr, ctx=mx.cpu()):
    """
    :param dim_arr: 各个维度的个数
    :param ctx: 运行环境
    :return:
    """
    my_net = gluon.nn.HybridSequential()
    with my_net.name_scope():
        my_net.add(OneHotLayer(dim_arr))
        my_net.add(gluon.nn.Dense(1, activation='sigmoid'))
    my_net.initialize(ctx=ctx)
    my_net.hybridize()
    return my_net


def get_embedding_hot_lr():
    my_net = gluon.nn.HybridSequential()
    with my_net.name_scope():
        pass


"""
deep wide model
"""


class DW(gluon.nn.HybridBlock):

    def __init__(self):

        pass
