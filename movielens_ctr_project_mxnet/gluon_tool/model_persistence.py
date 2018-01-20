# coding: utf-8

import mxnet as mx


def save_model(file_name, net):
    net.save_params(file_name)


def read_model(file_name, net, ctx=mx.cpu()):
    net.load_params(file_name, ctx)
    return net
