# coding: utf-8

import sys
sys.path.append("../")
from movielens.fm_basic import MyMutEmbedding
from movielens.preprocess import get_csv_file_batch_tier

import mxnet as mx
from mxnet import gluon
from mxnet import autograd
from sklearn import metrics
import codecs
import numpy as np

try:
    my_ctx = mx.gpu()
except:
    my_ctx = mx.cpu()

print(my_ctx)

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


def simple_fm_net(dim_array, output_dim, ctx):
    """
    最简单的fm net
    :param dim_array:
    :param output_dim:
    :return: gluon.nn.Block
    """
    my_net = gluon.nn.Sequential()
    with my_net.name_scope():
        my_net.add(MyMutEmbedding(dim_array, output_dim, False))

        my_net.add(gluon.nn.Dense(18, activation='sigmoid'))
        my_net.add(gluon.nn.Dense(1, activation='sigmoid'))
    my_net.initialize()
    return my_net


def get_trainer(net, sgd_type="sgd", learning_rate=0.5):
    """
    :type net: gluon.nn.Block
    :param net:
    :return:
    """
    trainer = gluon.Trainer(net.collect_params(), sgd_type, {'learning_rate': learning_rate})
    return trainer


def train_model(train_step, trainer, net, loss, data_iter, batch_size=10):
    for i in range(train_step):
        train_loss = LossCollect()
        for temp in data_iter:
            data = temp.data[0]
            train_X = data[:, 1:]
            train_y = data[:, 0]
            with autograd.record():
                output = net(train_X)
                loss_num = loss(train_y, output)
            train_loss.collect(mx.nd.sum(loss_num).asscalar(), loss_num.shape[0])
            loss_num.backward()
            trainer.step(batch_size)
        print(train_loss.get_mean_loss())
        train_loss.plot_batch_loss()

    pass


def train_model_without_batch_size(train_step, data_X, data_y, net, trainer):
    train_loss = LossCollect()
    for i in range(train_step):
        with autograd.record():
            output = net(data_X)
            loss_num = loss(data_y, output)
        loss_num.backward()
        trainer.step(batch_size)
        print(mx.nd.sum(loss_num).asscalar() / len(data_y))
        train_loss.collect(mx.nd.sum(loss_num).asscalar(), loss_num.shape[0])
    # train_loss.plot_batch_loss()

class LossCollect(object):

    def __init__(self):
        self.data_length = 0
        self.loss_sum = 0.
        self.batch_loss = []
        pass

    def collect(self, mini_loss_sum, size):
        self.loss_sum += mini_loss_sum
        self.data_length += size
        self.batch_loss.append(mini_loss_sum/size)

    def get_mean_loss(self):
        return self.loss_sum / self.data_length

    def plot_batch_loss(self):
        import matplotlib.pyplot as plt
        x = [z for z in range(len(self.batch_loss))]
        y = self.batch_loss
        plt.figure(figsize=(10, 5))
        plt.plot(x, y)
        plt.show()


def cal_auc(label, prodict_prob_y):
    test_auc = metrics.roc_auc_score(label, prodict_prob_y)
    return test_auc


def predict(net, X):
    """
    :type X: mx.nd.array
    :type net: gluon.nn.Block
    :param net:
    :param X:
    :return: mx.nd.array
    """
    return net(X)
    pass


def get_train_data_in_memory(file_path):
    train_X = []
    train_y = []
    with codecs.open(file_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip().split(",")
            train_y.append(int(line[0]))
            train_X.append([float(line[1]), float(line[2]), float(line[3]), float(line[4]), float(line[5])])
    return train_y, train_X
    pass


def get_data_iter(memory_data_X, memory_data_y, batch_size):
    length = len(memory_data_X)
    batch_num = length // batch_size
    print(batch_num)
    for i in range(batch_num):
        start = i * batch_size
        end = (i + 1) * batch_size
        yield (memory_data_y[start:end], memory_data_X[start:end])


def get_top_n_data(file_path, top_n):
    train_X = []
    train_y = []
    i = 0
    with codecs.open(file_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip().split(",")
            train_y.append(int(line[0]))
            train_X.append([float(line[1]), float(line[2]), float(line[3]), float(line[4]), float(line[5])])
            i += 1
            if i == top_n:
                break
    return train_y, train_X


def get_auc(csv_path, net, top_n):
    train_y, train_X = get_top_n_data(csv_path, top_n)
    train_y = np.array(train_y)
    predict_y = net(mx.nd.array(train_X))
    predict_y_np = predict_y.asnumpy()
    test_auc = metrics.roc_auc_score(train_y, predict_y_np)
    print(test_auc)

if __name__ == "__main__":
    csv_path = '/home/elliottqian/Documents/PycharmProjects/deeplearning_notebook/mxnet_study/movielens/part.csv'
    dim_array = [2, 7, 21, 3706, 301]
    batch_size = 1000
    top_n = 1000000
    loss = gluon.loss.SigmoidBinaryCrossEntropyLoss(from_sigmoid=True)
    # data_iter = get_csv_file_batch_tier(csv_path, data_shape=(6,), batch=batch_size)
    train_y, train_X = get_top_n_data(csv_path, top_n)

    fm_net = simple_fm_net(dim_array, 10, my_ctx)
    trainer = get_trainer(fm_net, sgd_type="AdaGrad", learning_rate=0.8)
    # train_model(2, trainer, fm_net, csv_path, data_shape=(6,), batch_size=1024)
    data_X = mx.nd.array(train_X)
    data_y = mx.nd.array(train_y)
    train_model_without_batch_size(50, data_X, data_y, fm_net, trainer)
    get_auc(csv_path, fm_net, top_n)
    # train_y, train_X = get_train_data_in_memory(csv_path)
    # data_iter = get_data_iter(train_X, train_y, 1000000)
    # # print(next(data_iter))
    # # print(next(data_iter))
    # # print(next(data_iter))
    # # print(next(data_iter))
    # # print(next(data_iter))
    #
    # for x in data_iter:
    #     print(x)
    #
    # data_iter = get_csv_file_batch_tier(csv_path, data_shape=(6,), batch=1024)
    # for x in data_iter:
    #     print(x)
    # j = 0
    # while True:
    #     try:
    #         print(j)
    #         j += 1
    #         data_iter.next()
    #     except :
    #         break
    # print("over")


    # train_y, train_X = get_train_data_in_memory(csv_path)
    # train_y = np.array(train_y)
    # predict_y = fm_net(mx.nd.array(train_X))
    # predict_y_np = predict_y.asnumpy()
    # test_auc = metrics.roc_auc_score(train_y, predict_y_np)
    # print(test_auc)
    pass





