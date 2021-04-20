import mxnet as mx
from .convert_utils import *
from .convert_utils import _RECON_PREFIX
from .recon_ops import recon_ops_dict




def _op_func(op: mx.symbol.Symbol):
    return recon_ops_dict[f"{_RECON_PREFIX}{op.attr('op_name')}"]


def sym_recon(model: mx.sym.Symbol, params: mx.ndarray.ndarray.NDArray, **kwargs):

    def dispatcher(op, params:dict={}, graph, rule_list:list=[], **kwargs):
        config = kwargs.copy()
        for rule in rule_list:
            if eval(f"{rule['pattern']}"):
                config.update(rule['kwargs'])
        return _op_func(op) (op, params, graph, **config)

    model, params = topo_visit_recon(model, params, dispatcher, **kwargs)
    return model, params


def sym_collect(model: mx.sym.Symbol, **kwargs):
    pass