import torch
from torch.utils._pytree import tree_map, tree_flatten
from typing import List, Any
from numbers import Number
from collections import defaultdict
 
aten = torch.ops.aten
iters = 0

def prod(x):
    res = 1
    for i in x:
        res *= i
    return res

def matmul_flop(inputs: List[Any], outputs: List[Any]) -> Number:
    """
    Count flops for matmul.
    """
    # Inputs contains the shapes of two matrices.
    input_shapes = [v.shape for v in inputs]
    assert len(input_shapes) == 2, input_shapes
    assert input_shapes[0][-1] == input_shapes[1][-2], input_shapes
    flop = prod(input_shapes[0]) * input_shapes[-1][-1]
    return flop
 
def addmm_flop(inputs: List[Any], outputs: List[Any]) -> Number:
    """
    Count flops for fully connected layers.
    """
    input_shapes = [v.shape for v in inputs[1:3]]
    # input_shapes[0]: [batch size, input feature dimension]
    # input_shapes[1]: [batch size, output feature dimension]
    assert len(input_shapes[0]) == 2, input_shapes[0]
    assert len(input_shapes[1]) == 2, input_shapes[1]
    batch_size, input_dim = input_shapes[0]
    output_dim = input_shapes[1][1]
    flops = batch_size * input_dim * output_dim
    return flops
 
def bmm_flop(inputs: List[Any], outputs: List[Any]) -> Number:
    """
    Count flops for the bmm operation.
    """
    # Inputs should be a list of length 2.
    # Inputs contains the shapes of two tensor.
    assert len(inputs) == 2, len(inputs)
    input_shapes = [v.shape for v in inputs]
    n, c, t = input_shapes[0]
    d = input_shapes[-1][-1]
    flop = n * c * t * d
    return flop

def log_softmax_flop(inputs: List[Any], outputs: List[Any]) -> Number:
    return 2 * inputs[0].numel()

def nln_flop(inputs: List[Any], outputs: List[Any]) -> Number:
    return 4 * inputs[0].numel()
 
def softmax_flop(inputs: List[Any], outputs: List[Any]) -> Number:
    return 2 * inputs[0].numel()

flop_mapping = {
    aten.mm: matmul_flop,
    aten.mm.default: matmul_flop,
    aten.matmul: matmul_flop,
    aten.addmm: addmm_flop,
    aten.addmm.default: addmm_flop,
    aten.bmm: bmm_flop,
    aten._log_softmax: log_softmax_flop,
    aten._log_softmax.default: log_softmax_flop,
    aten.native_layer_norm: nln_flop,
    aten.native_layer_norm.default: nln_flop,
    aten.softmax: softmax_flop,
    aten._softmax.default: softmax_flop

}

######### FLOP Counting Util Functions #########
flop_counts = defaultdict(lambda: defaultdict(int))
funcs = set()
parents = ['Global']

def start_counting():
    global parents, flop_counts
    parents = ['Global']
    flop_counts.clear()

def enter_module(name):
    def f(module, inputs):
        global parents
        parents.append(name)
        return inputs
 
    return f
 
def exit_module(name):
    def f(module, inputs, outputs):
        global parents
        assert(parents[-1] == name)
        parents.pop()
        return outputs
    return f
     
def instrument_module(mod):
    for name, module in dict(mod.named_children()).items():
        module.register_forward_pre_hook(enter_module(name))
        module.register_forward_hook(exit_module(name))
 

def display_flops():
    for mod in flop_counts.keys():
        print(f"Module: ", mod)
        for k,v in flop_counts[mod].items():
            print(k, v/1e9)

######### FLOPTensor Wrapper Class #########

class FlopTensor(torch.Tensor):
    elem: torch.Tensor
 
    __slots__ = ['elem']
 
    @staticmethod
    def __new__(cls, elem):
        # The wrapping tensor (FlopTensor) shouldn't hold any
        # memory for the class in question, but it should still
        # advertise the same device as before
        r = torch.Tensor._make_wrapper_subclass(  # type: ignore[attr-defined]
            cls, elem.size(),
            strides=elem.stride(), storage_offset=elem.storage_offset(),
            # TODO: clone storage aliasing
            dtype=elem.dtype, layout=elem.layout,
            device=elem.device, requires_grad=elem.requires_grad
        )
        # ...the real tensor is held as an element on the tensor.
        r.elem = elem
        return r
 
    def __repr__(self):
        if self.grad_fn:
            return f"FlopTensor({self.elem}, grad_fn={self.grad_fn})"
        return f"FlopTensor({self.elem})"
    
    def tolist(self):
        return self.elem.tolist()
    
    @classmethod
    def __torch_dispatch__(cls, func, types, args=(), kwargs=None):
        def unwrap(e):
            return e.elem if isinstance(e, FlopTensor) else e
 
        rs = func(*tree_map(unwrap, args), **tree_map(unwrap, kwargs))
        outs = rs if isinstance(rs, tuple) else (rs, )
        if func in flop_mapping:
            global flop_counts
            flop_count = flop_mapping[func](args, outs)
            for par in parents:
                flop_counts[par][func.__name__] += flop_count
        else:
            funcs.add(func.__name__)
        def wrap(e):
            return FlopTensor(e) if isinstance(e, torch.Tensor) else e
 
        rs = tree_map(wrap, rs)
        return rs


__all__ = ["instrument_module", "display_flops"]
