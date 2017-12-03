# -*- coding: utf-8 -*-

import mxnet as mx
import codecs
from mxnet import autograd

class LoadData(object):

    def __init__(self, file_path):
        self.file_path = file_path
        array = []
        for line in codecs.open(self.file_path, encoding="uft-8"):
            line = line.strip().split("\t")

        pass


    def get_batch(self):

        pass


def get_csv_file_batch_tier(csc_path, data_shape=(6), batch=128):
    return mx.io.CSVIter(data_csv=csc_path, data_shape=data_shape, batch_size=batch)


def train(trainer, step_size, csc_path, net, data_shape=(6), batch=128, ctx=mx.cpu()):
    for _ in step_size:
        while True:
            try:
                data_label = dataIter.next().data[0]
                train_X = data_label[:, 1:].as_in_context(ctx)
                train_y = data_label[:, 0].as_in_context(ctx)
                with autograd.record():
                    pass
            except Exception as _:
                break



if __name__ == "__main__":
    dataIter = mx.io.CSVIter(data_csv='/home/elliottqian/Documents/PycharmProjects/deeplearning_notebook/mxnet_study/movielens/part.csv',
                             data_shape=(6,), batch_size=2)
    # print(dataIter.next().data[0])
    print(dataIter.next().data[0].as_in_context(mx.gpu()))

    # dataIter.reset()
    # while True:
    #     try:
    #         dataIter.next()
    #     except Exception as _:
    #         break
    pass

