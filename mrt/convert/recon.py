import mxnet as mx
from .convert_utils import *
from .recon_ops import recon_ops_dict, _RECON_PREFIX




def _op_func(op: mx.symbol.Symbol):
    return recon_ops_dict[f"{_RECON_PREFIX}{op.attr('op_name')}"]


def sym_recon(model: mx.sym.Symbol, **kwargs):

    def dispatcher(op, graph, rule_list:list=[], **kwargs):
        config = kwargs.copy()
        for rule in rule_list:
            if eval(f"{rule['pattern']}"):
                config.update(rule['kwargs'])
        return _op_func(op) (op, graph, **config)

    model = topo_visit_recon(model, dispatcher, **kwargs)
    return model

def sym_collect(model: mx.sym.Symbol, **kwargs):
    pass