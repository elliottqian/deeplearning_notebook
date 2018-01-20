# coding: utf-8

from mxnet import autograd
from mxnet import gluon
import mxnet as mx
import time


def get_log_loss(from_sigmoid=True):
    return gluon.loss.SigmoidBinaryCrossEntropyLoss(from_sigmoid=from_sigmoid)


def get_softmax_cross_entropy_loss():
    return gluon.loss.SoftmaxCrossEntropyLoss()


def get_trainer(my_net, gd_method, param_dict):
    """
    :param my_net:
    :param gd_method: "AdaGrad", 'sgd'
    :param param_dict: {'learning_rate': 0.3}
    :return:
    """
    return gluon.Trainer(my_net.collect_params(), gd_method, optimizer_params=param_dict)


def train_model(step_num, data_iter, trainer, my_net_work, my_loss):
    for step in range(step_num):
        train_loss = 0.0
        temp_iter = reset(data_iter)
        iter_num = 0
        for element in temp_iter:
            iter_num += 1
            X, y = get_train_X_y(element)
            batch_size = X.shape[0]
            with autograd.record():
                output = my_net_work(X)
                loss_num = my_loss(output, y)
            loss_num.backward()
            trainer.step(batch_size)
            train_loss += mx.nd.mean(loss_num).asscalar()
        print(train_loss / iter_num)


def train_model_in_gpu(step_num, data_iter, trainer, my_net_work, my_loss, context=mx.gpu(0)):
    """
    train my model in gpu
    :param step_num:
    :param data_iter:
    :param trainer:
    :param my_net_work:
    :param my_loss:
    :param context:
    :return:
    """
    for step in range(step_num):
        train_loss = 0.0
        temp_iter = reset(data_iter)
        iter_num = 0
        for element in temp_iter:
            iter_num += 1
            X, y = get_train_X_y(element)
            X = X.as_in_context(context=context)
            y = y.as_in_context(context=context)
            batch_size = X.shape[0]
            with autograd.record():
                output = my_net_work(X)
                loss_num = my_loss(output, y)
            loss_num.backward()
            trainer.step(batch_size)
            train_loss += mx.nd.mean(loss_num).asscalar()
        print(train_loss / iter_num)


def train_softmax_model(step_num, iter_, trainer, my_net_work, my_loss):
    for step in range(step_num):
        train_loss = 0.
        temp_iter = reset(iter_)
        iter_num = 0
        for element in temp_iter:
            iter_num += 1
            X, y = get_train_X_y(element)
            batch_size = X.shape[0]
            y = y.reshape((-1,))
            with autograd.record():
                output = my_net_work(X)
                loss_num = my_loss(output, y)
            loss_num.backward()
            trainer.step(batch_size)
            train_loss += mx.nd.mean(loss_num).asscalar()
        if step % 1 == 0:
            print(train_loss / iter_num)


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


# def train(train_data, test_data, net, loss, trainer, ctx, num_epochs, print_batches=None):
#     """Train a network"""
#     print("Start training on ", ctx)
#     if isinstance(ctx, mx.Context):
#         ctx = [ctx]
#     for epoch in range(num_epochs):
#         train_loss, train_acc, n = 0.0, 0.0, 0.0
#         if isinstance(train_data, mx.io.MXDataIter):
#             train_data.reset()
#         start = time()
#         for i, batch in enumerate(train_data):
#             data, label, batch_size = _get_batch(batch, ctx)
#             losses = []
#             with autograd.record():
#                 outputs = [net(X) for X in data]
#                 losses = [loss(yhat, y) for yhat, y in zip(outputs, label)]
#             for l in losses:
#                 l.backward()
#             train_acc += sum([(yhat.argmax(axis=1)==y).sum().asscalar()
#                               for yhat, y in zip(outputs, label)])
#             train_loss += sum([l.sum().asscalar() for l in losses])
#             trainer.step(batch_size)
#             n += batch_size
#             if print_batches and (i+1) % print_batches == 0:
#                 print("Batch %d. Loss: %f, Train acc %f" % (
#                     n, train_loss/n, train_acc/n
#                 ))
#
#         test_acc = evaluate_accuracy(test_data, net, ctx)
#         print("Epoch %d. Loss: %.3f, Train acc %.2f, Test acc %.2f, Time %.1f sec" % (
#             epoch, train_loss/n, train_acc/n, test_acc, time() - start
#         ))