# coding: utf-8

import mxnet as mx
from mxnet import autograd
from mxnet import  gluon


class OneHotLayer(gluon.nn.Block):

    def __init__(self, one_hot_array, **kwargs):
        super(OneHotLayer, self).__init__(**kwargs)
        self.one_hot_array = one_hot_array

    def forward(self, x):
        ## one_hot([1, 0, 2, 0], 3)
        length = len(self.one_hot_array)
        temp = x[:, 0]
        result = mx.nd.one_hot(temp, self.one_hot_array[0])
        for i in range(1, length):
            temp = x[:, i]
            temp_one_hot = mx.nd.one_hot(temp, self.one_hot_array[i])
            result = mx.nd.concat(result, temp_one_hot, dim=1)
        return result
        pass


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
    net = OneHotLayer(a)
    net.initialize()
    print(net(test_x))
    pass





