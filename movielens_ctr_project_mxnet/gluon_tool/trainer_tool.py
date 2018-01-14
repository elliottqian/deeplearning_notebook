# coding: utf-8

from mxnet import autograd
from mxnet import gluon
import mxnet as mx


def get_log_loss(from_sigmoid=True):
    return gluon.loss.SigmoidBinaryCrossEntropyLoss(from_sigmoid=from_sigmoid)


def get_trainer(my_net, gd_method, param_dict):
    """
    :param my_net:
    :param gd_method: "AdaGrad", 'sgd'
    :param param_dict: {'learning_rate': 0.3}
    :return:
    """
    return gluon.Trainer(my_net.collect_params(), gd_method, optimizer_params=param_dict)


def train_model(step_num, iter, trainer, my_net_work, my_loss):
    for step in range(step_num):

        temp_iter = reset(iter)
        for element in temp_iter:

            X, y = get_train_X_y(element)
            batch_size = X.shape[0]
            with autograd.record():
                output = my_net_work(X)
                loss_num = my_loss(y, output)
            print(mx.nd.sum(loss_num) / batch_size)
            loss_num.backward()
            trainer.step(batch_size)

    pass


def reset(iter_):
    """
    # reset 函数, 用于重头开始迭代
    :param iter_:
    :return:
    """
    type_str = str(type(iter_))
    if "<class 'mxnet.io.MXDataIter'>" == type_str:
        iter_.reset()
        return iter_
    return None
    pass


def get_train_X_y(element):
    element_type = str(type(element))
    if "<class 'mxnet.io.DataBatch'>" == element_type:
        X_y = element.data[0]
        length = element.data[0][0].shape[0]
        y = mx.nd.slice_axis(X_y, axis=1, begin=0, end=1)
        X = mx.nd.slice_axis(X_y, axis=1, begin=1, end=length)
        return X, y
    return 1, 0
    pass


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