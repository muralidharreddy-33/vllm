import operator
import torch
import torch.fx as fx
from torch._higher_order_ops.auto_functionalize import auto_functionalized
from typing import Iterable, Optional

def find_fn(nodes: Iterable[fx.Node], op) -> Optional[fx.Node]:
    for node in nodes:
        if node.op == "call_function" and node.target == op:
            return node
    return None


def find_auto_fn(nodes: Iterable[fx.Node], op) -> Optional[fx.Node]:
    for node in nodes:
        if node.op == "call_function" and node.target == auto_functionalized and node.args[0] == op:
            return node
    return None


def find_getitem(node: Iterable[fx.Node], idx: int) -> Optional[fx.Node]:
    for user in node.users:
        if user.op == "call_function" and user.target == operator.getitem and user.args[1] == idx:
            return user
    return None
