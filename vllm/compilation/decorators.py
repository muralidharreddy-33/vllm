import inspect
from typing import Dict, List, Union

import torch

import vllm.envs as envs
from vllm.compilation.levels import CompilationLevel
from vllm.compilation.wrapper import TorchCompileWrapperWithCustomDispatcher
from vllm.sequence import IntermediateTensors
from vllm.utils import supports_dynamo


def support_torch_compile(dynamic_arg_dims: Dict[str, Union[int, List[int]]]):

    def cls_decorator_helper(cls: type):
        # helper to pass `dynamic_arg_dims`` to `_support_torch_compile``
        # to avoid too much indentation for `_support_torch_compile``
        sig = inspect.signature(cls.forward)
        for k in dynamic_arg_dims:
            if k not in sig.parameters:
                raise ValueError(
                    f"Argument {k} not found in the forward method of {cls}")
        return _support_torch_compile(cls, dynamic_arg_dims)

    return cls_decorator_helper


def _support_torch_compile(cls: type,
                           dynamic_arg_dims: Dict[str, Union[int, List[int]]]):
    """
    A decorator to add support for compiling the forward method of a class.
    """

    # for CompilationLevel.DYNAMO_AS_IS , the upper level model runner
    # will handle the compilation, so we don't need to do anything here.
    if envs.VLLM_TORCH_COMPILE_LEVEL in [
            CompilationLevel.NO_COMPILATION, CompilationLevel.DYNAMO_AS_IS
    ] or not supports_dynamo():
        return cls

    # take care of method resolution order
    # make sure super().__init__ is called on the base class
    #  other than TorchCompileWrapperWithCustomDispatcher
    cls.__bases__ = cls.__bases__ + (TorchCompileWrapperWithCustomDispatcher, )

    old_init = cls.__init__

    def __init__(self, *args, **kwargs):
        old_init(self, *args, **kwargs)
        TorchCompileWrapperWithCustomDispatcher.__init__(self)

    cls.__init__ = __init__

    def __call__(self, *args, **kwargs):
        # torch.compiler.is_compiling() means we are inside the compilation
        # e.g. TPU has the compilation logic in model runner, so we don't
        # need to compile the model inside.
        if torch.compiler.is_compiling():
            return self.forward(*args, **kwargs)

        # the first compilation needs to have dynamic shapes marked
        if len(self.compiled_codes) < 1:
            sig = inspect.signature(self.__class__.forward)
            bound_args = sig.bind(self, *args, **kwargs)
            bound_args.apply_defaults()
            for k, dims in dynamic_arg_dims.items():
                if isinstance(dims, int):
                    dims = [dims]
                arg = bound_args.arguments.get(k)
                if arg is not None:
                    if isinstance(arg, torch.Tensor):
                        for dim in dims:
                            torch._dynamo.mark_dynamic(arg, dim)
                    elif isinstance(arg, IntermediateTensors):
                        for tensor in arg.tensors.values():
                            for dim in dims:
                                torch._dynamo.mark_dynamic(tensor, dim)

        # if we don't use custom dispatcher, we can directly call the
        # compiled function and let torch.compile handle the dispatching,
        # with the overhead of guard evaluation and recompilation.
        if len(self.compiled_codes) < 1 or not self.use_custom_dispatcher:
            return self.compiled_callable(*args, **kwargs)

        # usually, capturing the model once is enough, and then we can
        # dispatch to the compiled code directly, without going through
        # the Dynamo guard mechanism.
        with self.dispatch_to_code(0):
            model_output = self.forward(*args, **kwargs)
            return model_output

    cls.__call__ = __call__
    return cls
