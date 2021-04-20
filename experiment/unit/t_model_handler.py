import site
site.addsitedir('../../')
from yamrt import MxnetModelHandler
from yamrt.transformer import topo_sort
model = MxnetModelHandler.load('/home/ljz/mrt_model/resnet50_v2.json', '/home/ljz/mrt_model/resnet50_v2.params')

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

name_seq_0 = []
for op in model:
    name_seq_0.append(op.attr('name'))

name_seq_1 = []
def print_sym_name(model, *args, **kwargs):
    for op in topo_sort(model):
        name_seq_1.append(op.attr('name'))
model.visit_sym(print_sym_name)

assert(len(name_seq_0) == len(name_seq_1))
for pair in zip(name_seq_0, name_seq_1):
    assert(pair[0] == pair[1])
