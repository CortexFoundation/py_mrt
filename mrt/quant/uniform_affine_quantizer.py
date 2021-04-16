from .common import *
import mxnet as mx


def _round_ste(x):
    return mx.nd.stop_gradient(mx.nd.round(x) - x) + x


def _new_detached_nd(*args):
    res = []
    for item in args:
        res.append(item.detach())
    return res


class UniformAffineQuantizer(mx.operator.CustomOp):
    def __init__(self, n_bits):
        super(UniformAffineQuantizer, self).__init__()
        self.n_bits = n_bits
        self.n_levels = 2 ** self.n_bits

    def forward(self, is_train, req, in_data, out_data, aux):
        conv_weight, delta, zero_point = in_data[0], in_data[1], in_data[2]
        x_int = _round_ste(conv_weight / delta) + zero_point
        x_quant = mx.nd.clip(x_int, 0, self.n_levels - 1)
        x_dequant = (x_quant - zero_point) * delta
        self.assign(out_data[0], req[0], x_dequant)

    def backward(self, req, out_grad, in_data, out_data, in_grad, aux): # Seems like checkpoint techs in pytorch 
        conv_weight, delta, zero_point = _new_detached_nd(*in_data[:3])# in_data[0].copy().detach(), in_data[1].copy().detach(), in_data[2].copy().detach()
        conv_weight.attach_grad()
        delta.attach_grad()
        zero_point.attach_grad()
        with mx.autograd.record():
            x_int = _round_ste(conv_weight / delta) + zero_point
            x_quant = mx.nd.clip(x_int, 0, self.n_levels - 1)
            x_dequant = (x_quant - zero_point) * delta
        x_dequant.backward(_new_detached_nd(out_grad[0])[0])

        self.assign(in_grad[0], req[0], conv_weight.grad)
        self.assign(in_grad[1], req[1], delta.grad)
        self.assign(in_grad[2], req[2], zero_point.grad)


@mx.operator.register(QUANT_OP_PREFIX + "UniformAffineQuantizer")
class UniformAffineQuantizerProp(mx.operator.CustomOpProp):
    def __init__(self, n_bits):
        super(UniformAffineQuantizerProp, self).__init__()
        n_bits = n_bits if type(n_bits) is int else int(n_bits) 

        assert 2 <= n_bits <= 32, 'bitwidth not supported'
        self.n_bits = n_bits

    def list_arguments(self):
        return ['data', 'delta', 'zero_point']

    def list_outputs(self):
        return ['output']

    def infer_shape(self, in_shape):
        assert(len(in_shape)==3)
        return [*in_shape], [in_shape[0]], []

    def infer_type(self, in_type):
        return [*in_type], [in_type[0]], []

    def create_operator(self, ctx, shapes, dtypes):
        return UniformAffineQuantizer(n_bits=self.n_bits)

