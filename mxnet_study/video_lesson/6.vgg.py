# -*- coding: utf-8 -*-

import mxnet.gluon.nn as mgnn
from mxnet import nd
import mxnet


def vgg_block(num_convs, channels):
    """
    define the basic structure of vgg
    structure: n convolution with 3x3 kernel, a 2x2 pooling

    mgnn.Sequential() is a structure for building net work, see: http://zh.gluon.ai/chapter_gluon-basics/index.html
    :param num_convs:
    :param channels:
    :return:
    """
    out = mgnn.Sequential()
    for _ in range(channels):
        out.add(
            mgnn.Conv2D(channels=channels, kernel_size=3, padding=1, activation='relu')
        )
    out.add(
        mgnn.MaxPool2D(pool_size=2, strides=2)
    )
    return out


def vgg_stack(architecture):
    out = mgnn.Sequential()
    for (num_convs, channels) in architecture:
        out.add(vgg_block(num_convs, channels))
    return out


def test():
    blk = vgg_block(2, 128)
    blk.initialize()
    x = nd.random.uniform(shape=(2,3,16,16))
    y = blk(x)
    print(y.shape)


num_outputs = 10
architecture = ((1,64), (1,128))
net = mgnn.Sequential()
# add name_scope on the outermost Sequential
with net.name_scope():
    net.add(
        vgg_stack(architecture),
        mgnn.Flatten(),
        mgnn.Dense(4096, activation="relu"),
        mgnn.Dropout(.5),
        mgnn.Dense(num_outputs))


import sys
sys.path.append('..')
import utils
from mxnet import gluon
from mxnet import init

train_data, test_data = utils.load_data_fashion_mnist(
    batch_size=16, resize=28)

ctx = mxnet.gpu()
net.initialize(ctx=ctx, init=init.Xavier())

loss = gluon.loss.SoftmaxCrossEntropyLoss()
trainer = gluon.Trainer(net.collect_params(),
                        'sgd', {'learning_rate': 0.05})
utils.train(train_data, test_data, net, loss,
            trainer, ctx, num_epochs=1)