import mxnet as mx 

#"input_data": mx.nd.ones([43,3,224,224]),
conv_weight = mx.nd.ones([64,3,7,7])
delta = mx.nd.ones([1]) + 100
zero_point = mx.nd.ones([1]) + 4

def _round_ste(x):
    return mx.nd.stop_gradient(mx.nd.round(x) - x) + x

conv_weight.attach_grad()
zero_point.attach_grad()
delta.attach_grad()

with mx.autograd.record():
    x_int = _round_ste(conv_weight / delta) + zero_point
    x_quant = mx.nd.clip(x_int, 0, 2**8 - 1)
    x_dequant = (x_quant - zero_point) * delta

x_dequant.backward(
    mx.nd.random.uniform(shape=x_dequant.shape)
    )

print(delta.grad)
print(zero_point.grad)