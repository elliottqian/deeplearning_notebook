import mxnet as mx

# a = mx.sym.Variable('a')
#
# b = mx.sym.Variable('b')
#
# c = 2 * a + b
#
# xx = mx.symbol.stack(a, b)
# #xxx = mx.sym.stack(xx, c)
# print(xx.bind(mx.gpu(), {'a': mx.nd.array([[1], [3]], ctx=mx.gpu()), 'b':mx.nd.array([[2], [5]], ctx=mx.gpu())}).forward())
# #print(xxx.bind(mx.cpu(), {'a': mx.nd.array([1,2]), 'b':mx.nd.array([2,3])}).forward())
#
# yy = mx.symbol.concat(a, b, dim=1)
# zz = mx.symbol.concat(yy, c, dim=1)
# zz = mx.symbol.zeros(shape=)
# for _ in range(10):
#     zz = mx.symbol.concat(zz, c, dim=1)
# print(zz.bind(mx.gpu(), {'a': mx.nd.array([[1], [3]], ctx=mx.gpu()), 'b':mx.nd.array([[2], [5]], ctx=mx.gpu())}).forward())


inputs = {'x': mx.nd.array([[1, 2, 3], [4, 4, 6]], ctx=mx.gpu())}
print(inputs)
b = mx.symbol.Variable(name="b", shape=(1,))

w = mx.symbol.Variable(name="w", shape=(3, 1))
v = mx.symbol.Variable(name="v", shape=(3, 2))
x = mx.symbol.Variable(name="x", shape=(2, 3))

temp1 = mx.symbol.broadcast_add(mx.symbol.dot(x, w), b)

args = {"b": mx.nd.array([1], ctx=mx.gpu()),
        "w": mx.nd.array([[1, 3, 5]], ctx=mx.gpu()).transpose(),
        "x": mx.nd.array([[1, 2, 3], [4, 5, 6]], ctx=mx.gpu())}

print(b.bind(mx.gpu(), args).forward())
print(w.bind(mx.gpu(), args).forward())
print(x.bind(mx.gpu(), args).forward())
print(temp1.bind(mx.gpu(), args).forward())


r = mx.symbol.zeros(name="tempx", shape=(2, 3))

