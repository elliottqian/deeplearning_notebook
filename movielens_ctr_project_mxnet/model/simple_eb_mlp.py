# coding: utf-8

from mxnet import gluon
import mxnet
from layer.MyEmbedding import MyMutEmbeddingHb


def get_simple_embedding_mlp(dim_array, output_dim, activation_f="relu", context=mxnet.gpu(0)):
    my_net = gluon.nn.HybridSequential()
    with my_net.name_scope():
        my_net.add(MyMutEmbeddingHb(dim_array, output_dim, True))
        my_net.add(gluon.nn.Dense(30, activation=activation_f))
        my_net.add(gluon.nn.Dense(1, activation='sigmoid'))
    my_net.initialize(ctx=context)
    my_net.hybridize()
    return my_net
