import site
site.addsitedir('../../')
from yamrt.modelhandler import MxnetModelHandler
from yamrt.autoquanter import MxnetAutoQuanter
from yamrt.transformer.tfm_utils import topo_sort
model = MxnetModelHandler.load('/home/ljz/mrt_model/resnet50_v2.json', '/home/ljz/mrt_model/resnet50_v2.params')
quanter = MxnetAutoQuanter(model)
quanter.prepare(input_shape=[64, 3, 224, 224])


recon_config_list = [
    {
        "pattern": "op.attr('op_name') == 'Convolution'", # Injectable
        "kwargs": {
            "quant_weight": True,
            "quant_weight_config": {
                "q_op_name": 'MRT_UniformAffineQuantizer',
                "n_bits": 32,
                "channel_wise": True,
            },
            "quant_bias": True,
            "quant_bias_config": {
                "q_op_name": "MRT_Proxy",
            },
            "quant_activation": True,
            "quant_activation_config": {
                "q_op_name": 'MRT_UniformAffineQuantizer',
                "n_bits": 32,
                "channel_wise": True,
            }
        },
    }
]

def print_sym_name(model, *args, **kwargs):
    for op in topo_sort(model):
        print(op.attr('name'))

model.visit_model(print_sym_name)

quanter.ptq_pre(recon_config_list)

model.visit_model(print_sym_name)
