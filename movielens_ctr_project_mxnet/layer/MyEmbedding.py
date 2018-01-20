# coding: utf-8

import mxnet as mx
from mxnet import gluon


class MyMutEmbedding(gluon.nn.Block):
    """
    自己的embedding层
    """
    def __init__(self, dims, out_put_dim, is_reshape=False, **kwargs):
        """
        :param dims: 输入数组, 例如[2, 3, 2], 分别代表每个embedding的输入范围
        :param out_put_dim: embedding的输出范围
        :param is_reshape: embedding的输出范围
        :param kwargs:
        """
        super(MyMutEmbedding, self).__init__(**kwargs)
        self.is_reshape = is_reshape
        self.dims = list(dims)
        self.out_put_dim = out_put_dim
        with self.name_scope():
            for i in range(len(dims)):
                temp_embedding = gluon.nn.Embedding(input_dim=dims[i], output_dim=out_put_dim)
                # collect_param 方法会收集self._children里面的参数
                self._children.append(temp_embedding)

    def forward(self, x):
        """
        :type x: mxnet.nd.array
        :param x:
        :param is_reshape:
        :return:
        """
        embeding_list = []
        for i in range(len(self.dims)):
            temp = mx.nd.slice_axis(x, axis=1, begin=i, end=i + 1)  # 取得第一列
            embedding_temp = self._children[i](temp)
            embeding_list.append(embedding_temp)

        result = embeding_list[0]

        for i in range(1, len(self.dims)):
            result = mx.nd.concat(result, embeding_list[i], dim=1)

        if self.is_reshape:
            result = result.reshape((x.shape[0], -1))
        return result


class MyMutEmbeddingHb(gluon.nn.HybridBlock):

    """
    自己的embedding层
    """
    def __init__(self, dims, out_put_dim, is_flatten=False, **kwargs):
        """
        :param dims: 输入数组, 例如[2, 3, 2], 分别代表每个embedding的输入范围
        :param out_put_dim: embedding的输出范围
        :param is_reshape: embedding的输出范围
        :param kwargs:
        """
        super(MyMutEmbeddingHb, self).__init__(**kwargs)
        self.is_flatten = is_flatten
        self.dims = list(dims)
        self.out_put_dim = out_put_dim
        with self.name_scope():
            for i in range(len(dims)):
                temp_embedding = gluon.nn.Embedding(input_dim=dims[i], output_dim=out_put_dim)
                # collect_param 方法会收集self._children里面的参数
                self._children.append(temp_embedding)

    def hybrid_forward(self, F, x, **kwargs):
        """
        :type x: mxnet.nd.array
        :param x:
        :param is_reshape:
        :return:
        """
        embeding_list = []
        for i in range(len(self.dims)):
            temp = F.slice_axis(x, axis=1, begin=i, end=i + 1)  # 取得第一列
            embedding_temp = self._children[i](temp)
            embeding_list.append(embedding_temp)

        result = embeding_list[0]

        for i in range(1, len(self.dims)):
            result = F.concat(result, embeding_list[i], dim=1)

        if self.is_flatten:
            result = F.flatten(result)

        return result


'''
class OneHotLayer(gluon.nn.HybridBlock):

    def __init__(self, one_hot_array, **kwargs):
        super(OneHotLayer, self).__init__(**kwargs)
        self.one_hot_array = one_hot_array
        self.length = len(self.one_hot_array)

    def hybrid_forward(self, F, x, **kwargs):
        temp = F.slice_axis(x, axis=1, begin=0, end=1)  # 取得第一列
        result = F.one_hot(temp, self.one_hot_array[0]).flatten()  # 第一列展开

        for i in range(1, self.length):
            temp = F.slice_axis(x, axis=1, begin=i, end=i + 1)  # 取得第i列
            temp_one_hot = F.one_hot(temp, self.one_hot_array[i]).flatten()  # 第i列展开
            result = F.concat(result, temp_one_hot, dim=1)  # 按照列拼接
        return result
'''