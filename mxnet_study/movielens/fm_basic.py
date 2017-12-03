# coding: utf-8

from mxnet import gluon
import mxnet


class MyMutEmbedding(gluon.nn.Block):
    """
    自己的embedding层
    """
    def __init__(self, dims, out_put_dim, is_reshape=False, **kwargs):
        """

        :param dims: 输入数组, 例如[2, 3, 2], 分别代表每个embedding的输入范围
        :param out_put_dim: embedding的输出范围
        :param kwargs:
        """
        super(MyMutEmbedding, self).__init__(**kwargs)
        self.is_reshape = is_reshape
        self.dims = list(dims)
        self.out_put_dim = out_put_dim
        with self.name_scope():
            for i in range(len(dims)):
                temp_embedding = gluon.nn.Embedding(input_dim=dims[i], output_dim=out_put_dim)
                # collect_param 方法会收集self._children里面的参数
                self._children.append(temp_embedding)

    def forward(self, x):
        """
        :type x: mxnet.nd.array
        :param x:
        :param is_reshape:
        :return:
        """
        embeding_list = []
        for i in range(len(self.dims)):
            temp = x[:, i]
            embedding_temp = self._children[i](temp)
            embeding_list.append(embedding_temp)

        result = embeding_list[0]

        for i in range(1, len(self.dims)):
            result = mxnet.nd.concat(result, embeding_list[i], dim=1)

        if self.is_reshape:
            result = result.reshape((x.shape[0], x.shape[1], self.out_put_dim))
        return result


class MySimpleFm(object):

    @staticmethod
    def ger_simple_fm(input_dims, output_dim):
        """
        得到一个简单的三层神经网络, embedding层
        :param input_dims:
        :param output_dim:
        :return:
        """
        net = gluon.nn.Sequential()
        with net.name_scope():
            net.add(MyMutEmbedding(input_dims, output_dim))
            net.add(gluon.nn.Dense(1, activation='sigmoid'))
        return net

    @staticmethod
    def train(net, train_x, true_label, batch_size=64, method='sgd', learning_rate=0.1, step_num=1):
        sigmoid_loss = gluon.loss.SigmoidBinaryCrossEntropyLoss(from_sigmoid=True)
        trainer = gluon.Trainer(net.collect_params(), method, {'learning_rate': learning_rate})

        from mxnet import autograd
        for _ in range(step_num):
            with autograd.record():
                output = net(train_x)
                loss_num = sigmoid_loss(true_label, output)
                print(mxnet.ndarray.sum(loss_num))
            loss_num.backward()
            trainer.step(4)
        pass

    @staticmethod
    def predict():
        pass


if __name__ == "__main__":
    net = MyMutEmbedding([1, 2, 3], 3, is_reshape=True)