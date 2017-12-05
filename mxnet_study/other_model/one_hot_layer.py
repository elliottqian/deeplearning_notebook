# coding: utf-8

import mxnet as mx
from mxnet import autograd
from mxnet import gluon


class OneHotLayer(gluon.nn.HybridBlock):

    def __init__(self, one_hot_array, **kwargs):
        super(OneHotLayer, self).__init__(**kwargs)
        self.one_hot_array = one_hot_array
        self.length = len(self.one_hot_array)

    def hybrid_forward(self, F, x, **kwargs):
        temp = F.slice_axis(x, axis=1, begin=0, end=1)
        result = F.one_hot(temp, self.one_hot_array[0]).flatten()

        for i in range(1, self.length):
            temp = F.slice_axis(x, axis=1, begin=i, end=i + 1)
            temp_one_hot = F.one_hot(temp, self.one_hot_array[i]).flatten()
            result = F.concat(result, temp_one_hot, dim=1)
        return result


class InnerProductLayer(gluon.nn.Block):

    def __init__(self, **kwargs):
        super(InnerProductLayer, self).__init__(**kwargs)

    def forward(self, x):
        """
        :param x:
        :return:
        """
        y = mx.nd.swapaxes(x, 1, 2)
        result = mx.nd.batch_dot(x, y)
        result = mx.nd.flatten(result)
        return result


if __name__ == "__main__":
    a = [3, 3, 5]
    test_x = mx.nd.array([[1, 2, 3], [2, 1, 4]])
    test_x_2 = mx.nd.array([[1, 2, 1], [2, 1, 4], [2, 2, 0]])
    net = OneHotLayer(a)
    net.initialize()
    print(net(test_x))
    print(net(test_x_2))
    pass





