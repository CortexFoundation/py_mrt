# General
# None
# Mxnet Backend
import mxnet as mx
from mx import autograd
from mx import sym
from mx import nd
from ..model import *

from ...symbol.sym_utils import *

class SymbolModel(Model):
    def __init__(self, op:sym.Symble, param:dict):
        super(SymbolModel, self).__init__()
        self._flush_name()
        self._op_names = [op.attr('name')]
        self._ops = { op.attr('name'): op }

    def _flush_name(self):
        self._name = f"{list(self._inputs.keys())}->{list(self._outputs.keys())}"

    def attach_sym(self, op:sym.Symbol, param_dict:dict):
        op_name = op.attr('name')
        assert op_name not in self._op_names
        assert op_name not in self._inputs

        for child in op.get_children():
            child_name = child.attr("name")
            child_op = child.attr("op_name")
            if child_op == "null":
                assert (child_name not in self._param)
                if child_name not in param_dict:
                    if child_name not in self._inputs:
                        self._inputs[child_name] = get_entry_id(child)
                else:
                    self._param[child_name] = param_dict[child_name]
            else:
                if child_name not in self._inputs:
                    self._inputs[child_name] = get_entry_id(child)
        self._op_names.append(op_name)
        self._ops[op_name] = op
        self._flush_name()

    def _set_train(self):
        for key, op in self._ops.items():
            op.attach_grad()
    
    def _set_eval(self):
        pass

    def add_output(self, name):
        assert name in self._op_names
        if name not in self._outputs:
            self._outputs.append(name)
        self._flush_name()

    def forward(self, data:dict):
        for key in self.inputs:
            assert ( key in data )
        contex = data
        contex.update(self._param)
        if self._training:
            with autograd.record():
                topo_sort()