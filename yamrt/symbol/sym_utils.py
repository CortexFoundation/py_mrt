import logging
import json
import math

import cvm

import mxnet as mx
from mxnet import ndarray as nd
from mxnet.symbol import _internal


def is_op(sym, params):
    return (sym.attr('op_name') != 'null')

def is_var(sym, params):
    return (sym.attr('op_name') == 'null')

def is_params(sym, params):
    return is_var(sym, params) and \
        (sym.attr('name') in params)

def is_inputs(sym, params):
    return is_var(sym, params) and \
        (sym.attr('name') not in params)


def nd_array(source_array, ctx=None, dtype="float64"):
    return nd.array(source_array, ctx=ctx, dtype=dtype)

def nd_arange(*args, **kwargs):
    return nd.arange(*args, dtype="float64", **kwargs)

def nd_full(*args, **kwargs):
    return nd.full(*args, dtype="float64", **kwargs)

def nd_zeros(*args, **kwargs):
    return nd.zeros(*args, dtype="float64", **kwargs)

def nd_ones(*args, **kwargs):
    return nd.ones(*args, dtype="float64", **kwargs)


_MX_OP_CONTRIB_PREFIX = '_contrib_'
def get_mxnet_op(op_name):
    op = getattr(_internal, op_name, None)
    if op is None:
        op = getattr(mx.symbol, op_name, None)
    if op_name.startswith(_MX_OP_CONTRIB_PREFIX):
        op = getattr(mx.symbol.contrib, op_name[len(_MX_OP_CONTRIB_PREFIX):], None)

    if op is None:
        raise RuntimeError("Unable to map op_name {} to mxnet.sym".format(op_name))
    return op


_MX_OP_CONTRIB_PREFIX = '_contrib_'
def get_nd_op(op_name):
    op = getattr(nd, op_name, None)
    if op is None:
        op = getattr(nd._internal, op_name, None)
    if op_name.startswith(_MX_OP_CONTRIB_PREFIX):
        op = getattr(nd.contrib, op_name[len(_MX_OP_CONTRIB_PREFIX):], None)

    if op is None:
        raise RuntimeError("Unable to map op_name {} to mxnet.ndarray".format(op_name))
    return op


def get_nnvm_op(op_name):
    op = getattr(cvm.symbol, op_name)

    if not op:
        raise RuntimeError("Unable to map op_name {} to nnvm.sym".format(op_name))
    return op


MULTIPYE_OUTS_NODE = [
    'get_valid_counts', 'SliceChannel',
    # group's op_name is None
    'None',
]
def get_entry_id(sym):
    oindex = 0
    if sym.attr('op_name') in MULTIPYE_OUTS_NODE:
        if isinstance(sym, mx.sym.Symbol):
            oindex = json.loads(sym.tojson())['heads'][0][1]
    return oindex

def has_multi_outs(sym):
    return sym.attr('op_name') in MULTIPYE_OUTS_NODE

def get_node(sym, graph):
    """ Assume all graph node have single output.
        Multiple output node will be fused
        by `fuse_multiple_outputs` sym_pass.
    """
    name = sym.attr('name')
    if name not in graph:
        assert False, "Unrecognized layer:%s in graph keys:%s" \
            % (name, graph.keys())
    return graph[name][get_entry_id(sym)]


def sym_iter(sym):
    if sym is None:
        return None

    if isinstance(sym, mx.sym.Symbol):
        sym = [sym[i] for i in range(len(sym))]
    else:
        assert isinstance(sym, cvm.sym.Symbol)
        size = len(sym.list_output_names())
        sym = [sym[i] for i in range(size)]
    return sym


NoneAttr = object()
def get_attr(attr, name, default=NoneAttr):
    if name in attr:
        if isinstance(default, str):
            return attr[name]
        return eval(attr[name])
    if default == NoneAttr:
        assert False, "attr %s is not exists in %s" % (name, attr)
    return default


def nd_const(number, graph, params):
    name = 'const_var_' + str(number)
    prec = math.ceil(math.log2(math.fabs(number)+1)) + 1
    if name not in params and name not in graph:
        attr = { 'precision': str(prec) }
        graph[name] = mx.sym.var(name, shape=(1,), attr=attr)
        params[name] = nd_array([number])
    return graph[name]
