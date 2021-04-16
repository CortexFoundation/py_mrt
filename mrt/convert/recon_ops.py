import mxnet as mx
from .convert_utils import *
from ..quant import *
import copy

_RECON_PREFIX = "recon_"


def _has_true(key, config):
    if key in config and config[key]:
        return True
    return False


def recon_Convolution(op: mx.sym.Symbol, graph: dict, info_dict: dict
        quant_weight=True, quant_weight_config={},
        quant_bias=True, quant_bias_config={},
        quant_activation=True, quant_activation_config={},
        **kwargs):

    childs = sym_iter(op.get_children())

    if quant_weight:
        weight = childs[1]
        weight_name = weight.attr('name')
        assert(weight.attr('op_name') == 'null')
        assert(weight_name in graph)
        qweight_op_name = quant_weight_config["q_op_name"]
        del quant_weight_config["q_op_name"]
        assert(qweight_op_name not in weight_name)
        assert(_RECON_PREFIX not in weight_name)
        qweight_name = f"{_RECON_PREFIX}{qweight_op_name}_{weight_name}"
        zero_point = mx.sym.Variable(qweight_name + '_zero_point', shape=(1))
        delta = mx.sym.Variable(qweight_name+ '_delta', shape=(1))
        quant_weight_config.update(kwargs) 
        qw = mx.sym.Custom(
            data=weight,
            delta=delta,
            zero_point=zero_point,
            name=qweight_name,
            op_type=qweight_op_name,
            **quant_weight_config)
        childs, attrs = sym_iter(op.get_children()), op.list_attr()
        childs[1] = qw
        op = get_mxnet_op(op.attr('op_name'))(*childs, **attrs, name=op.attr('name'))

    if quant_bias:
        bias = childs[2]
        assert bias.attr('op_name') == 'null'
        bias_name = bias.attr('name')
        qbias_op_name = quant_bias_config["q_op_name"]
        del quant_bias_config["q_op_name"]
        assert(qbias_op_name not in bias_name)
        assert(_RECON_PREFIX not in bias_name)
        qbias_name = f"{_RECON_PREFIX}{qbias_op_name}_{bias_name}"
        qbias = mx.sym.Variable(**bias.list_attr(), name= qbias_name + "_qbias")
        quant_bias_config.update(kwargs)
        qb = mx.sym.Custom(
            data=bias,
            qbias=qbias,
            name=qbias_name,
            op_type=qbias_op_name,
            **quant_bias_config)
        childs, attrs = sym_iter(op.get_children()), op.list_attr()
        childs[2] = qb
        op = get_mxnet_op(op.attr('op_name'))(*childs, **attrs, name=op.attr('name'))

    if quant_activation:
        pass


    return op


def recon_null(op: mx.sym.Symbol, graph: dict, config: dict={}):
    assert(isinstance(op, mx.sym.Symbol))
    return op


recon_ops_dict = {
    "recon_Convolution": recon_Convolution,
    "recon_null": recon_null
}
