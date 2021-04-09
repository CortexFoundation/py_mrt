import mxnet as mx
import gluoncv as cv
from mxnet import ndarray as nd
from mrt.transformer import Model

if mx.__version__ > '1.5.1':
    print(f"[Warning] Untested Version: {mx.__version__}")

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
            childs = mx_sym_iter(childs)
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
resnet50_v2_par = convert_params_dtype(nd.load('/home/ljz/mrt_model/resnet50_v2.params'))

resnet50_v2 = Model(resnet50_v2_sym, resnet50_v2_par)
resnet50_v2.prepare([64, 3, 224, 224])
# resnet50_v2 = resnet50_v2.to_graph()


for item in topo_sort(resnet50_v2.symbol):
    if 'Convolution' == item.attr('op_name'):
        conv = item
        break
print(conv)
class QConvolution(mx.operator.CustomOp):
    def forward(self, is_train, req, in_data, out_data, aux):
        self.assign(out_data[0], req[0], in_data[0])

    def backward(self, req, out_grad, in_data, out_data, in_grad, aux):
        pass

@mx.operator.register("qconvolution")
class QConvolutionProp(mx.operator.CustomOpProp):
    def __init__(self):
        super(QConvolutionProp, self).__init__()

    def list_arguments(self):
        return ['data', 'weight_qp', 'activation_qp']

    def list_outputs(self):
        return ['output']

    def infer_shape(self, in_shape):# ['data', 'weight_qp', 'activation_qp' ]
        print(in_shape)
        return [in_shape[0], in_shape[1], in_shape[2]], [in_shape[0]], []

    def infer_type(self, in_type):
        print(in_type)
        return [*in_type], [in_type[0]], []

    def create_operator(self, ctx, shapes, dtypes):
        return QConvolution()

print(conv.infer_shape())
data = mx.symbol.Variable('data')
mlp = mx.symbol.Custom(data=data, name='qconv1', op_type='qconvolution')

#c = mlp.bind(mx.cpu(), {'data': mx.nd.array([1,2,3]), 'qconv1_weight_qp': mx.nd.array([1,2,3]), 'qconv1_activation_qp': mx.nd.array([1,2,3])})
#y = c.forward()

#print(y)