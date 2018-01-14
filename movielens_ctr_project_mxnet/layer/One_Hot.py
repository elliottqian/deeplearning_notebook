# coding: utf-8

from mxnet import gluon


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

