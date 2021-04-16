import mxnet as mx
import gluoncv as cv
from mxnet import ndarray as nd
from mxnet.symbol import _internal
from mxnet import symbol as _sym

from mrt.transformer import Model
from mrt import dataset as ds

if mx.__version__ > '1.5.1':
    print(f"[Warning] Untested Version: {mx.__version__}")

_MX_OP_CONTRIB_PREFIX = '_contrib_'
def get_mxnet_op(op_name):
    op = getattr(_internal, op_name, None)
    if op is None:
        op = getattr(_sym, op_name, None)
    if op_name.startswith(_MX_OP_CONTRIB_PREFIX):
        op = getattr(_sym.contrib, op_name[len(_MX_OP_CONTRIB_PREFIX):], None)

    if op is None:
        raise RuntimeError("Unable to map op_name {} to mxnet.sym".format(op_name))
    return op

MULTIPYE_OUTS_NODE = [
    'get_valid_counts', 'SliceChannel',
    # group's op_name is None
    'None',
]
def get_entry_id(sym):
    oindex = 0
    if sym.attr('op_name') in MULTIPYE_OUTS_NODE:
        if isinstance(sym, _sym.Symbol):
            oindex = json.loads(sym.tojson())['heads'][0][1]
        elif isinstance(sym, cvm.sym.Symbol):
            graph = cvm.graph.create(sym)
            oindex = json.loads(graph.json())['heads'][0][1]
    return oindex

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


def convert_params_dtype(params, src_dtypes=["float32", "float64"],
        dest_dtype="float64"):
    if not params:
        return {}
    if isinstance(src_dtypes, str):
        src_dtypes = [src_dtypes]
    nparams = {}
    for k, v in params.items():
        dtype = v.dtype.__name__
        if dtype != dest_dtype and dtype in src_dtypes:
            nparams[k] = v.astype(dest_dtype)
        else:
            nparams[k] = v
    return nparams

def mx_sym_iter(sym):
    if sym is None:
        return None
    if isinstance(sym, mx.sym.Symbol):
        sym = [sym[i] for i in range(len(sym))]
    return sym

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

def topo_sort(symbol, with_deps=False):
    """Sort all symbols in the mxnet graph in topological order.
    """
    queue = []
    symbol_map = {}
    deps = {}
    dep_cnts = {}
    for s in symbol:
        symbol_map[s.attr('name')] = s
        queue.append(s)

    while queue:
        sym = queue.pop(0)
        name = sym.attr('name')
        childs = sym.get_children()
        if childs is None:
            dep_cnts[name] = 0
        else:
            childs = sym_iter(childs)
            # remove duplication dependency
            dep_cnts[name] = len({c.attr('name') for c in childs})
            for child in childs:
                child_name = child.attr('name')
                if child_name not in deps:
                    deps[child_name] = set()
                deps[child_name].add(name)
                if child_name not in symbol_map:
                    symbol_map[child_name] = child
                    queue.append(child)

    order = []
    reduce_flag = True
    while dep_cnts:
        if not reduce_flag:
            # logger.critical("deps cannot reduce -> %s", dep_cnts)
            assert False
        remove = []
        reduce_flag = False
        for name in dep_cnts:
            if dep_cnts[name] == 0:
                # order.append(symbol_map[name])
                yield symbol_map[name]
                remove.append(name)
                if name in deps:
                    for other in deps[name]:
                        dep_cnts[other] -= 1
                reduce_flag = True
        for name in remove:
            del dep_cnts[name]



resnet50_v2_sym = mx.sym.load('/home/ljz/mrt_model/resnet50_v2.json')
resnet50_v2_par = convert_params_dtype(nd.load('/home/ljz/mrt_model/resnet50_v2.params'), src_dtypes='float64', dest_dtype='float32')

resnet50_v2 = Model(resnet50_v2_sym, resnet50_v2_par)
resnet50_v2.prepare([64, 3, 224, 224]) # Fuse serveral models.
#resnet50_v2 = resnet50_v2_sym


# Steal the first conv from the model.
for item in topo_sort(resnet50_v2.symbol):
    if 'Convolution' == item.attr('op_name'):
        conv = item
        break

def round_ste(x):
    return mx.nd.stop_gradient(mx.nd.round(x) - x) + x


def new_detached_nd(*args):
    res = []
    for item in args:
        res.append(item.detach())
    return args


class QWeight(mx.operator.CustomOp):
    def __init__(self, n_bits: int = 8, symmetric: bool = False, channel_wise: bool = False, scale_method: str = 'max'):
        super(QWeight, self).__init__()
        self.sym = symmetric if type(symmetric) is bool else eval(symmetric)
        assert 2 <= n_bits <= 32, 'bitwidth not supported'
        self.n_bits = n_bits
        self.n_levels = 2 ** self.n_bits
        self.channel_wise = channel_wise if type(channel_wise) is bool else eval(channel_wise)
        self.scale_method = scale_method

    def forward(self, is_train, req, in_data, out_data, aux):
        conv_weight, delta, zero_point = in_data[0], in_data[1], in_data[2]
        x_int = round_ste(conv_weight / delta) + zero_point
        x_quant = mx.nd.clip(x_int, 0, self.n_levels - 1)
        x_dequant = (x_quant - zero_point) * delta
        self.assign(out_data[0], req[0], x_dequant)

    def backward(self, req, out_grad, in_data, out_data, in_grad, aux):
        conv_weight, delta, zero_point = new_detached_nd(*in_data[:3])# in_data[0].copy().detach(), in_data[1].copy().detach(), in_data[2].copy().detach()
        conv_weight.attach_grad()
        delta.attach_grad()
        zero_point.attach_grad()
        with mx.autograd.record():
            x_int = round_ste(conv_weight / delta) + zero_point
            x_quant = mx.nd.clip(x_int, 0, self.n_levels - 1)
            x_dequant = (x_quant - zero_point) * delta
        x_dequant.backward(new_detached_nd(out_grad[0])[0])
        self.assign(in_grad[0], req[0], conv_weight.grad)
        self.assign(in_grad[1], req[1], delta.grad)
        self.assign(in_grad[2], req[2], zero_point.grad)

#TODO
## 1 Forward 
### 1.1 Parameter Init
### 1.2 Forward Precs Eval
### 1.3 
## 2 Backward
### 2.1 
    #def backward(self, req, out_grad, in_data, out_data, in_grad, aux):

@mx.operator.register("qweight")
class QWeightProp(mx.operator.CustomOpProp):
    def __init__(self, **kwargs):
        super(QWeightProp, self).__init__()
        self._weight_quant_params=kwargs

    def list_arguments(self):
        return ['data', 'delta', 'zero_point']

    def list_outputs(self):
        return ['output']

    def infer_shape(self, in_shape):
        return in_shape, [in_shape[0]], []

    def infer_type(self, in_type):
        return [*in_type], [in_type[0]], []

    def create_operator(self, ctx, shapes, dtypes):
        return QWeight(**self._weight_quant_params)

# print(conv.infer_shape((1,0,1)))
data = mx.symbol.Variable('data')
print(f"conv.list_arguments(): {conv.list_arguments()}")

conv_childs = conv.get_children()
conv_weights = list()
for child in conv_childs:
    print(f"{child.attr('name')}:{child.attr('op_name')}:{child.infer_shape()}")
print(conv.attr('num_filter') )
exit()
childs, attrs = sym_iter(conv.get_children()), conv.list_attr()
conv_wq = mx.symbol.Custom(data=childs[1], name='conv1_wq', op_type='qweight', channel_wise=True)
childs[1] = conv_wq
conv = get_mxnet_op(conv.attr('op_name'))(*childs,**attrs, name=conv.attr('name'))
    #            print(item.attr('name'))
            #mlp = mx.symbol.Custom(data=item, name='conv1_wq', op_type='qweight', channel_wise=True)

graph = {conv.attr('name'): conv}
    #params = {k:v[:] for k, v in params.items()}
for op in topo_sort(resnet50_v2.symbol):
    name, op_name = op.attr('name'), op.attr('op_name')
    if name == conv.attr('name'):
        continue
    childs, attr = sym_iter(op.get_children()), op.list_attr()
    if childs is not None:
        childs = [get_node(c, graph) for c in childs]
        op = get_mxnet_op(op_name)(*childs, **attr, name=name)
    graph[name] = op
    if graph[name] is None:
        graph[name] = op
nodes = [get_node(op, graph) for op in resnet50_v2.symbol]
resnet50_v2.symbol = get_mxnet_op("Group")(nodes) if len(
    nodes) > 1 else nodes[0]
resnet50_v2.params = convert_params_dtype(resnet50_v2.params, src_dtypes='float64', dest_dtype='float32')
input_names = resnet50_v2.symbol.list_inputs()
output_names = resnet50_v2.symbol.list_outputs()


dataset = ds.DS_REG['imagenet']([64, 3, 224, 224], root="/home/ljz/.mxnet/datasets")
data_iter_func = dataset.iter_func()


for data in data_iter_func():
    #pd = mx.gluon.ParameterDict.load_dict(resnet50_v2.params)
    #model = mx.gluon.SymbolBlock(outputs, inputs, params=pd)
    
    inputs_dict = {**resnet50_v2.params}
    grad_dict = {}
    for key in resnet50_v2.params:
        grad_dict[key] = mx.nd.zeros(resnet50_v2.params[key].shape)
    inputs_dict.update({'data': data.as_in_context(mx.cpu()), 'conv1_wq_delta': mx.nd.array([0.000001]), 'conv1_wq_zero_point': mx.nd.array([0.0001])})
    args_grad = {'conv1_wq_delta': mx.nd.ones((1)), 'conv1_wq_zero_point': mx.nd.ones((1))}
    grad_dict.update(args_grad)
    c = resnet50_v2.symbol.bind(mx.cpu(), args=inputs_dict, args_grad=grad_dict)
    y = c.forward(is_train=True).copy()
    res = c.backward(y)
    print(res)
    exit()

#c = mlp.bind(mx.cpu(), {'data': mx.nd.array([1,2,3]), 'qconv1_weight_delta': mx.nd.array([1,2,3]), 'qconv1_weight_zero_point': mx.nd.array([1,2,3])})
#y = c.forward()
#print(mlp.get_children())
#print(y)