from mxnet import ndarray as nd
import math
import numpy as np
import time

from .tfm_utils import topo_visit_recon, topo_sort, convert_params_dtype
from .tfm_base import *

from ..symbol.sym_utils import get_entry_id, get_mxnet_op, get_node, is_inputs, is_var, is_params


def name_duplicate_check(symbol, params):
    names = set()
    for sym in topo_sort(symbol):
        name = sym.attr('name')
        assert name not in names, "duplicated name in graph: %s" % name
        names.add(name)


@N.register_nm("ais")
def attach_input_shape(symbol, params, input_shapes):
    assert isinstance(input_shapes, dict)

    def _impl(op, params, graph):
        name, attr = op.attr('name'), op.list_attr()
        if is_inputs(op, params) and name in input_shapes:
            op = mx.sym.var(name, shape=input_shapes[name], attr=attr)
        return op

    return topo_visit_recon(symbol, params, _impl)


@N.register_nm("fmi")
def fuse_multiple_inputs(sym, params):
    infer_shapes = infer_shape(sym, params)
    dim_sum, dim_per, dims = 0, {}, {}
    def _sum_input(node, params, **kwargs):
        name = node.attr('name')
        nonlocal dim_sum, dim_per, dims
        if is_inputs(node, params):
            dims[name] = infer_shapes[name][0]
            dot = np.product(dims[name])
            dim_per[name] = dot
            dim_sum += dot
    topo_visit_recon(sym, params, _sum_input)

    assert len(dim_per) > 0, "no input in graph"
    if len(dim_per) == 1:
        return sym, params

    data_sum = mx.sym.var('data', shape=(dim_sum,))
    first, last = 0, 0
    def _change_node(op, params, graph, **kwargs):
        name = op.attr('name')
        if is_inputs(op, params):
            nonlocal first, last
            last = first + dim_per[name]
            op = mx.sym.slice(data_sum, name=N.n('slice'),
                    begin=(first,), end=(last,))
            op = mx.sym.reshape(op, name=N.n('reshape'),
                    shape=dims[name])
            first = last
        return op
    sym, params = topo_visit_recon(sym, params, _change_node)
    return sym, params


def model_inputs(symbol, params):
    input_count = 0
    def _count(op, params, graph):
        nonlocal input_count
        input_count += is_inputs(op, params)
    topo_visit_recon(symbol, params, _count)
    return input_count


def input_name_replace(symbol, params):
    def _name_replace(op, params, graph):
        name, attr = op.attr('name'), op.list_attr()
        if is_inputs(op, params):
            op = mx.sym.var("data", attr=attr)
        return op
    return topo_visit_recon(symbol, params, _name_replace)


def infer_shape(symbol, params, input_shape=None):
    infer_shapes = {}
    def _impl(op, params, graph):
        name, op_name = op.attr('name'), op.attr('op_name')
        if is_params(op, params):
            oshp = [params[name].shape]
            op = mx.sym.var(name, shape=oshp[0])
        else:
            _, oshp, _ = op.infer_shape()

        if is_inputs(op, params):
            if input_shape is None:
                assert oshp is not None, "It seems that graph doesn't set \
                        input_shape, please invoke attach_input_shape first."
            else:
                oshp = [input_shape]
                op = mx.sym.var(name, shape=oshp[0])
        infer_shapes[name] = oshp
        return op
    topo_visit_recon(symbol, params, _impl)
    return infer_shapes


@N.register_nm("fmo")
def fuse_multiple_outputs(symbol, params):
    infer_shapes = infer_shape(symbol, params)
    channel, graph = {}, {}
    for sym in topo_sort(symbol):
        name, op_name = sym.attr('name'), sym.attr('op_name')
        childs, attr = sym_iter(sym.get_children()), sym.list_attr()
        if childs is not None:
            childs = [get_node(c, graph) for c in childs]
            sym = get_mxnet_op(op_name)(*childs, **attr, name=name)
        if op_name == 'SliceChannel':
            # Only designed for special usei, thus
            # check of "SliceChannel" has not been added to _sym_check
            assert childs is not None and len(childs) == 1, \
                "Invalid Layer: %s, the 'SliceChannel' \
                operator must have exactly one input" % name
            axis = get_attr(attr, 'axis', 1)
            num_outputs = get_attr(attr, 'num_outputs')
            chchild_shape = infer_shapes[childs[0].attr('name')]
            eid = get_entry_id(childs[0])
            dim = chchild_shape[eid][axis]
            assert num_outputs > 0 and dim % num_outputs == 0, \
                "Invalid Layer: %s, the 'SliceChannel' operator \
                has a wrong attribute, 'num_outputs': %d" \
                % (name, num_outputs)
            stride = int(dim / num_outputs)
            interval = [(i * stride, (i + 1) * stride) \
                       for i in range(num_outputs)]
            channel[name] = [childs, axis, interval]
        elif childs is not None:
            is_split = False
            for i in range(len(childs)):
                cname = childs[i].attr('name')
                if cname in channel:
                    is_split = True
                    eid = get_entry_id(childs[i])
                    chchilds, axis, interval = channel[cname]
                    begin, end = interval[eid]
                    chattr = {'axis': axis, 'begin': begin, 'end': end}
                    slp_name = N.n('slice_axis')
                    if slp_name not in graph:
                        graph[slp_name] = mx.sym.slice_axis(*chchilds,
                                **chattr, name=slp_name)
                    childs[i] = graph[slp_name]
            if is_split:
                sym = get_mxnet_op(op_name)(*childs, **attr, name=name)
        graph[name] = sym

    nodes = [get_node(sym, graph) for sym in symbol]
    ret = mx.sym.Group(nodes) if len(nodes) > 1 else nodes[0]
    return ret, params


@N.register_nm("fc")
def fuse_constant(symbol, params):
    nparams = convert_params_dtype(params, dest_dtype="float32")

    def _impl(op, params, graph):
        name, op_name = op.attr('name'), op.attr('op_name')
        childs, attr = sym_iter(op.get_children()), op.list_attr()
        if is_var(op, params):
            pass
        elif childs is None:
            params[name] = get_nd_op(op_name)(**attr)
            attr = { 'precision': str(get_bit(params[name])) }
            op = mx.sym.var(name, shape=params[name].shape, attr=attr)
        elif all([is_params(c, params) for c in childs]):
            in_params = [params[c.attr('name')] for c in childs]
            params[name] = get_nd_op(op_name)(*in_params, **attr)
            attr = { 'precision': str(get_bit(params[name])) }
            op = mx.sym.var(name, shape=params[name].shape, attr=attr)
        return op

    sym, params = topo_visit_recon(symbol, nparams, _impl)
    params = convert_params_dtype(params, dest_dtype="float64")
    return sym, params


@N.register_nm("fuse_transpose")
def fuse_transpose(symbol, params):
    infer_shapes = infer_shape(symbol, params)
    return topo_visit_recon(symbol, params,
            apply_pass("fuse_transpose", infer_shapes=infer_shapes))


@N.register_nm("rewrite")
def rewrite(symbol, params):
    infer_shapes = infer_shape(symbol, params)
    return topo_visit_recon(symbol, params,
            apply_pass("rewrite", infer_shapes=infer_shapes))


@N.register_nm("ptq_pre")
def ptq_pre(symbol, params, **kwargs):
    func = apply_pass("ptq_pre")
    rule_list = kwargs['rule_list']
    def dispatcher(op, **inkwargs):
        config = inkwargs.copy()
        for rule in rule_list:
            if eval(f"{rule['pattern']}"):
                config.update(rule['kwargs'])
        return func(op, **config)

    return topo_visit_recon(symbol, params, dispatcher, **kwargs)


def params_unique(symbol, params):
    new_params = {s.attr('name'):params[s.attr('name')] \
            for s in topo_sort(symbol) if is_params(s, params)}
    return symbol, new_params

