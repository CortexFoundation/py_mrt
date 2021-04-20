# General
from .modelhandler import ModelHandler

# Mxnet
from .transformer import *
from .transformer import tfm_pass as tpass 


class AutoQuanter(object):
    def __init__(self, model:ModelHandler):
        self._model = model

    def prepare(self, *args, **kwargs):
        raise NotImplementedError

    def ptq_pre(self, *args, **kwargs):
        raise NotImplementedError

    def ptq_pre_param(self, *args, **kwargs):
        raise NotImplementedError

    def ptq(self, *args, **kwargs):
        raise NotImplementedError 

    def ptq_collect(self, *args, **kwargs):
        raise NotImplementedError

    #TODO: Add full APIs.

class MxnetAutoQuanter(AutoQuanter):
    def __init__(self, model:ModelHandler):
        super(MxnetAutoQuanter, self).__init__(model)

    def prepare(self, input_shape:dict=None): #TODO: Turn configurable like ptq_pre.
        assert(input_shape is not None)
        self._model.visit_model(tpass.name_duplicate_check)
        if isinstance(input_shape, dict):
            self._model.update_model(tpass.attach_input_shape, input_shape=input_shape)
            self._model.update_model(tpass.fuse_multiple_inputs)
        elif input_shape is not None:
            model_inputs = self._model.visit_model(tpass.model_inputs)
            assert model_inputs == 1, "Multiple inputs non-known shape"
            self._model.update_model(tpass.input_name_replace)
            self._model.update_model(tpass.attach_input_shape, {"data": input_shape})
        self._model.visit_model(tpass.infer_shape)

        self._model.update_model(tpass.fuse_multiple_outputs)
        self._model.update_model(tpass.fuse_constant)
        self._model.update_model(tpass.fuse_transpose)
        self._model.update_model(tpass.rewrite)
        self._model.update_model(tpass.fuse_constant)
        self._model.update_model(tpass.params_unique)

    def ptq_pre(self, config):
        self._model.update_model(tpass.ptq_pre, rule_list=config)

    def ptq_pre_param(self):
        raise NotImplementedError

    def ptq(self, ):
        raise NotImplementedError 

    def ptq_collect(self):
        raise NotImplementedError
