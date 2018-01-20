# coding: utf-8
from mxnet import gluon
from layer.MyEmbedding import MyMutEmbedding
from layer.One_Hot import OneHotLayer


class DeepWideNetWork(gluon.nn.Block):

    def __init__(self, dim_array, embedding_dim, **kwargs):
        """
        两个部分, 1) wide 部分 2) deep 部分
        :param kwargs:
        """
        super(DeepWideNetWork, self).__init__(**kwargs)

        with self.name_scope():

            deep_net = gluon.nn.HybridSequential()

            with deep_net.name_scope():
                deep_net.add(MyMutEmbedding(dim_array, embedding_dim, True))
                deep_net.add(gluon.nn.Dense(128, activation="relu"))
            self.deep_net = deep_net

            wide_net = gluon.nn.HybridSequential()
            with wide_net.name_scope():
                wide_net.add(OneHotLayer(dim_array))
            self.wide_net = wide_net

            last_layer_net = gluon.nn.HybridSequential()
            with last_layer_net.name_scope():
                last_layer_net.add(gluon.nn.Dense(1, activation="sigmoid"))
            self.last_layer_net = last_layer_net





    def forward(self, x, *args):
        last_layer = self.wide_net(x).contact(self.deep_net(x))
        result = self.last_layer_net(last_layer)
        return result


"""
    my_net = gluon.nn.HybridSequential()
    with my_net.name_scope():
        my_net.add(OneHotLayer(dim_arr))
        my_net.add(gluon.nn.Dense(1, activation='sigmoid'))

net = gluon.nn.Sequential()
with net.name_scope():
    net.add(MyMutEmbedding(dim_array, 60, False))
    net.add(gluon.nn.Flatten())
    net.add(gluon.nn.Dense(128, activation="relu"))
    net.add(gluon.nn.Dense(2))
net.initialize()

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

"""