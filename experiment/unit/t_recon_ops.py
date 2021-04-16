import mxnet as mx
from mrt.convert import sym_recon
from mrt.convert.convert_utils import topo_sort, sym_iter


recon_config_list = [
    {
        "pattern": "op.attr('op_name') == 'Convolution'", # Injectable
        "kwargs": {
            "quant_weight": True,
            "quant_weight_config": {
                "q_op_name": 'MRT_UniformAffineQuantizer',
                "n_bits": 32,
            },
            "quant_bias": True,
            "quant_bias_config": {
                "q_op_name": "MRT_Proxy",
            },
            "quant_activation": True,
            "quant_activation_config": {
                "q_op_name"
            }
            

        },
    }
]
weight = mx.sym.Variable('weight', shape=(64,3,7,7))
bias = mx.sym.Variable('bias', shape=(64,))
qbias = mx.sym.Variable('bias', shape=(64,))
print(bias == qbias)
input_data = mx.sym.Variable('input_data', shape=(64, 3, 224, 224))
zero_point = mx.sym.Variable('zero_point', shape=(1))
delta = mx.sym.Variable('delta', shape=(1))
model = mx.sym.Convolution(input_data, weight, bias, kernel=(7,7), num_filter=64)
model = sym_recon(model, rule_list=recon_config_list)

for node in topo_sort(model):
    if 'Convolution' == node.attr('op_name'):
        childs = sym_iter(node.get_children())
        for child in childs:
            print(child.attr("name"))
            print(child.list_attr())

print(model.list_inputs())

