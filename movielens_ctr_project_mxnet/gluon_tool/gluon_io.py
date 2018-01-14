# coding: utf-8

import codecs
import mxnet as mx
import mxnet.io as mio


def get_csv_file_batch_tier(csv_path, data_shape=(6), batch=128):
    """

    :param csc_path:
    :param data_shape:
    :param batch:
    :return:
    """
    return mx.io.CSVIter(data_csv=csv_path, data_shape=data_shape, batch_size=batch)



def save_model():
    pass
