import mxnet as mx
from mxnet.symbol import _internal


_MX_OP_CONTRIB_PREFIX = '_contrib_'


def get_mxnet_op(op_name):
    op = getattr(_internal, op_name, None)
    if op is None:
        op = getattr(mx.symbol, op_name, None)
    if op_name.startswith(_MX_OP_CONTRIB_PREFIX):
        op = getattr(mx.symbol.contrib, op_name[len(_MX_OP_CONTRIB_PREFIX):], None)

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
        if isinstance(sym, mx.sym.Symbol):
            oindex = json.loads(sym.tojson())['heads'][0][1]
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
    raise StopIteration


def topo_visit_recon(symbol, callback, get_op=get_mxnet_op, **kwargs):
    graph = {}

    for op in topo_sort(symbol):
        name, op_name = op.attr('name'), op.attr('op_name')
        childs, attr = sym_iter(op.get_children()), op.list_attr()
        if childs is not None:
            childs = [get_node(c, graph) for c in childs]
            op = get_op(op_name)(*childs, **attr, name=name)

        graph[name] = callback(op, graph=graph, **kwargs)
        if graph[name] is None:
            graph[name] = op
    nodes = [get_node(op, graph) for op in symbol]
    ret = get_op("Group")(nodes) if len(nodes) > 1 else nodes[0]

    return ret


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