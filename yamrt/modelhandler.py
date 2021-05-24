# General
# None
# Mxnet Backend
import mxnet as mx
from .transformer.tfm_utils import convert_params_dtype, topo_sort


class ModelHandler(object):
    """ Wrapper of Model, design with user-friendly model API.
    """
    def __init__(self):
        pass

    @classmethod
    def load(*args, **kwargs):
        raise NotImplementedError

    def __next__(self):
        raise NotImplementedError

    def __call__(self, data):
        raise NotImplementedError

    def train(self):
        raise NotImplementedError
    
    def eval(self):
        raise NotImplementedError

    def visit_sym(self, func, *args, **kwargs):
        """Visit the architecture description of model in topo order.
        """
        raise NotImplementedError

    def visit_model(self, func, *args, **kwargs):
        """Visit the architecture description, parameters and other data of model in topo order.
        """
        raise NotImplementedError
    
    def update_model(self, func, *args, **kwargs):
        """Update the architecture description, parameters or other data of model in topo order.
        """
        raise NotImplementedError


class MxnetModelHandler(ModelHandler):
    """ Wrapper of Mxnet Model, design with user-friendly model API.
    """
    def __init__(self, model_sym:mx.sym.Symbol, model_params:mx.ndarray.NDArray, dtype:str="float64"):
        super(MxnetModelHandler, self).__init__()
        self._sym = model_sym
        self._param = convert_params_dtype(model_params, dest_dtype=dtype)
        self._check()
        self._extended_sym = None
        self._extended_param = None
        self._extended_dict = None
        self._train = False
    
    def symbol(self):
        return self._sym
    
    def params(self):
        return self._param

    def _check(self):
        for op in self._sym:
            if op.attr('op_name') == 'null':
                assert op.attr('name') in self._param

    @classmethod
    def load(cls, symbol_filepath, params_filepath, dtype:str='float64'):
        """ Model load from disk. """
        symbol = mx.sym.load(symbol_filepath)
        param = mx.nd.load(params_filepath)
        return cls(symbol, param, dtype)

    def __iter__(self):
        return topo_sort(self._sym)

    def __next__(self):
        for item in topo_sort(self._sym):
            yield item
        raise StopIteration

    def train(self):
        self._train = True

    def eval(self):
        self._train = False

    def visit_sym(self, func, *args,**kwargs):
        return func(self._sym, *args, **kwargs)

    def visit_model(self, func, *args, **kwargs):
        return func(self._sym, self._param, *args, **kwargs)

    def update_model(self, func, *args, **kwargs):
        self._sym, self._param = func(self._sym, self._param, *args, **kwargs)

    def extend_model(self, func, *args, **kwargs):
        self._extended_sym, self._extended_param, self._extended_dict = func(self._sym, self._param, *args, **kwargs)
