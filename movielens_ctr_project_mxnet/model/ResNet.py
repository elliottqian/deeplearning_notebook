# coding: utf-8

from mxnet import gluon
from layer.MyEmbedding import MyMutEmbedding


class ResBlock(gluon.nn.Block):

    def __init__(self, rest_unit_num, hidden_unit_num, **kwargs):
        """
        :type rest_unit_num: int
        :type hidden_unit_num: int
        :param rest_unit_num: int
        :param hidden_unit_num: int
        :param kwargs:
        """
        super(ResBlock, self).__init__(**kwargs)
        with self.name_scope():
            self.nn2 = gluon.nn.Dense(hidden_unit_num, activation="relu")
            self.nn3 = gluon.nn.Dense(rest_unit_num, activation="relu")

    def forward(self, x):
        y = self.nn3(self.nn2(x))
        return y + x
        pass



def get_embedding_res_net(embedding_array, embedding_size, res_net_hidden_unit_num):
    """
    :type embedding_array: list
    :param embedding_array:
    :param embedding_size:
    :param res_net_hidden_unit_num:
    :return:
    """
    res_unit_num = len(embedding_array) * embedding_size
    net = gluon.nn.Sequential()
    with net.name_scope():
        net.add(MyMutEmbedding(embedding_array, embedding_size))
        net.add(ResBlock(res_unit_num, res_net_hidden_unit_num))
        net.add(gluon.nn.Dense(1, activation="sigmoid"))


if __name__ == "__main__":

    pass
