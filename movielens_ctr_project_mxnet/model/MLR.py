# coding: utf-8

from mxnet import gluon
from mxnet import autograd
import mxnet as mx


class MLR_HybridBlock(gluon.nn.HybridBlock):

    def __init__(self, k, x_dim, **kwargs):
        super(MLR_HybridBlock, self).__init__(**kwargs)
        self.k = k
        with self.name_scope():
            self.u = self.params.get("weight_u", shape=(x_dim, self.k))
            self.u_b = self.params.get("bias_u", shape=(self.k,))
            self.w = self.params.get("weight_w", shape=(x_dim, self.k))
            self.w_b = self.params.get("bias_w", shape=(self.k,))

    def hybrid_forward(self, F, x, u, u_b, w, w_b):
        """
        mlr有三个部分, 第一部分的分子, 第一部分的分母, 第二部分
        现在分别实现, pai_son, pai_mo, lr_part
        :type x: mx.nd.array
        :param x:
        :return:
        """
        # pai的部分
        pai_son = F.exp(F.broadcast_add(F.dot(x, u), u_b))
        pai_mo = F.sum(pai_son, axis=1)
        pai = F.broadcast_div(pai_son, pai_mo.reshape((-1, 1)))
        #
        # # lr部分
        lr_part_wx = F.broadcast_add(F.dot(x, w), w_b)
        lr_part = 1 / (1 + F.exp(-lr_part_wx))

        # 结果相乘求和
        result = pai * lr_part
        return F.sum(result, axis=1)


class MLR(gluon.nn.Block):
    def __init__(self, k, x_dim, **kwargs):
        super(MLR, self).__init__(**kwargs)
        self.k = k
        with self.name_scope():
            self.u = self.params.get("weight_u", shape=(x_dim, self.k))
            self.u_b = self.params.get("bias_u", shape=(self.k,))
            self.w = self.params.get("weight_w", shape=(x_dim, self.k))
            self.w_b = self.params.get("bias_w", shape=(self.k,))

    def forward(self, x):
        """
        mlr有三个部分, 第一部分的分子, 第一部分的分母, 第二部分
        现在分别实现, pai_son, pai_mo, lr_part
        :type x: mx.nd.array
        :param x:
        :return:
        """
        # pai的部分
        pai_son = mx.nd.exp(mx.nd.dot(x, self.u.data()) + self.u_b.data())
        pai_mo = mx.nd.sum(pai_son, axis=1)
        pai = mx.nd.broadcast_div(pai_son, pai_mo.reshape((-1, 1)))

        # lr部分
        lr_part_wx = mx.nd.dot(x, self.w.data()) + self.w_b.data()
        lr_part = 1 / (1 + mx.nd.exp(-lr_part_wx))

        # 结果相乘求和
        result = pai * lr_part
        return mx.nd.sum(result, axis=1)