import mxnet as mx

X = mx.symbol.Variable("X")
y = mx.symbol.Variable("y")

number_hidden = 2
W = mx.symbol.Variable("W", shape=(-1, number_hidden))
b = mx.symbol.Variable("b", shape=(number_hidden,))




def feed():
    temp1 = mx.symbol.broadcast_add(mx.symbol.dot(X, W), b)
    layer1 = mx.symbol.relu(temp1)
    layer2 = mx.symbol.FullyConnected(layer1, num_hidden=10)
    return layer2


xxx = mx.ndarray.array([1, 2, 3])
print(xxx)
print(xxx[0])


net = feed()

args = {"X": mx.nd.array([[1, 2, 3], [4, 5, 6]]),
        "W": mx.nd.array([[1, 2], [3, 4], [5, 6]]),
        "b": mx.nd.array([1, 1, 1])}

