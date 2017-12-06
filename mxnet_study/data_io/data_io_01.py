# coding: utf-8

import codecs
import mxnet as mx

def load_data_iter(path, batch_size, split_sym="\t"):
    """
    节省内存的方式读取
    :param path:
    :param batch_size:
    :param split_sym:
    :return:
    """
    i = 0
    label = []
    data = []
    with codecs.open(path, encoding="utf-8") as f:
        for line in f:
            if i % batch_size == 0:
                label.clear()
                data.clear()
            i += 1
            temp = line.strip().split(split_sym)
            label.append(temp[0])
            data.append(temp[1:])
            if i % batch_size == 0:
                yield label, data


def mat_2_mx_array(ll, ctx=mx.cpu()):
    l = list(map(lambda x: str2float(x), ll))
    return mx.nd.array(l, ctx)

def str2float(l):
    return list(map(lambda x: float(x), l))

def list_2_mx_array(l, ctx=mx.cpu()):
    return mx.nd.array(str2float(l), ctx)

def get():
    csv_path = "/home/elliottqian/Documents/PycharmProjects/deeplearning_notebook/mxnet_study/movielens/out"
    t = load_data_iter(csv_path, 3, ",")
    for i in t:
        print(list_2_mx_array(i[0]))
        yield i

if __name__ == "__main__":
    print(next(get()))

    # print(next(t))
    # print(next(t))
    # print(next(t))
    # ll = next(t)
    # print(list_2_mx_array(ll))

