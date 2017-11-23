# -*- coding: utf-8 -*-

import mxnet as mx
import mxnet.gluon.nn as mgnn


class FM(mgnn.Block):
    def __init__(self, hidden_dim, user_dim, item_dim, **kwargs):
        super(FM, self).__init__(**kwargs)
        self.hidden_dim = hidden_dim
        self.user_dim = user_dim
        self.item_dim = item_dim
        with self.name_scope():
            self.embedding_user = mgnn.Embedding(input_dim=self.user_dim, output_dim=self.hidden_dim)
            self.embedding_item = mgnn.Embedding(input_dim=self.item_dim, output_dim=self.hidden_dim)


    def forward(self, x):
        """

        :param x:
        :type x: mx.ndarray
        :return:
        """
        user = x[:, 0]
        item = x[:, 1]
        emb_user = mx.ndarray.flatten(self.embedding_user(user))
        emb_item = mx.ndarray.flatten(self.embedding_item(item))
        r = mx.nd.sum(emb_user * emb_item, axis=1)
        return r

    def print_params(self):
        print(self.collect_params())
        print(self.name_scope())
        print(self.prefix)


if __name__ == "__main__":
    net = FM(hidden_dim=3, user_dim=10, item_dim=8, prefix="ewlfm_")
    net.print_params()
    net.initialize(ctx=mx.gpu())
    pass
