# coding: utf-8

from mxnet import gluon
from layer.MyEmbedding import MyMutEmbeddingHb


class EmbeddingFm(gluon.nn.HybridBlock):

    def __init__(self, feature_size, **kwargs):
        super(EmbeddingFm).__init__(**kwargs)
        self.feature_size = feature_size

    def hybrid_forward(self, F, x, *args, **kwargs):

        tensor_list = []
        for i in range(self.feature_size):
            for j in range(i + 1, self.feature_size):
                a = F.slice_axis(x, axis=1, begin=i, end=i + 1)
                b = F.slice_axis(x, axis=1, begin=j, end=j + 1)
                c = F.multiply(a, b)
                tensor_list.append(c)

        from functools import reduce as functools_r
        r = functools_r(
            lambda element1, element2: F.concat(element1, element2),
            tensor_list
        )

        return r


def get_deep_fm(dim_array, embedding_size):
    """

    :type embedding_size: int
    :type dim_array: list
    :param dim_array:
    :param embedding_size:
    :return:
    """
    net = gluon.nn.HybridSequential(prefix="elliottqian_")
    with net.name_scope():
        net.add(MyMutEmbeddingHb(dim_array, embedding_size))
    pass