import mxnet as mx
from .convert_utils import *
from ..fquant import *
import copy


def recon_Convolution(op: mx.sym.Symbol, graph: dict,
        quant_weight=True, quant_weight_config={},
        quant_bias=True, quant_bias_config={},
        quant_activation=True, quant_activation_config={},
        **kwargs):

    childs = sym_iter(op.get_children())

    if quant_weight:
        weight = childs[1]
        assert(weight.attr('op_name') == 'null')
        assert(weight.attr('name') in graph)
        quant_weight_config.update(kwargs)
        qw = UniformAffineQuantizerWrapper(weight, quant_weight_config)
        childs, attrs = sym_iter(op.get_children()), op.list_attr()
        childs[1] = qw.new_op()
        op = get_mxnet_op(op.attr('op_name'))(*childs, **attrs, name=op.attr('name'))

    if quant_bias:
        bias = childs[2]
        assert(bias.attr('op_name') == 'null')
        assert(bias.attr('name') in graph)
        quant_bias_config.update(kwargs)
        qb = ProxyWrapper(bias, quant_bias_config)
        childs, attrs = sym_iter(op.get_children()), op.list_attr()
        childs[2] = qb.new_op()
        op = get_mxnet_op(op.attr('op_name'))(*childs, **attrs, name=op.attr('name'))

    if quant_activation:
        quant_activation_config.update(kwargs)
        qa = UniformAffineQuantizerWrapper(op, quant_activation_config)
        op = qa.new_op()

    return op


def recon_null(op: mx.sym.Symbol, graph: dict, config: dict={}):
    assert(isinstance(op, mx.sym.Symbol))
    return op


recon_ops_dict = {
    "recon_Convolution": recon_Convolution,
    "recon_null": recon_null
}
