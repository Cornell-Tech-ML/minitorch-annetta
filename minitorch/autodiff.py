from dataclasses import dataclass
from typing import Any, Iterable, List, Tuple

from typing_extensions import Protocol

# ## Task 1.1
# Central Difference calculation


def central_difference(f: Any, *vals: Any, arg: int = 0, epsilon: float = 1e-6) -> Any:
    r"""
    Computes an approximation to the derivative of `f` with respect to one arg.

    See :doc:`derivative` or https://en.wikipedia.org/wiki/Finite_difference for more details.

    Args:
        f : arbitrary function from n-scalar args to one value
        *vals : n-float values $x_0 \ldots x_{n-1}$
        arg : the number $i$ of the arg to compute the derivative
        epsilon : a small constant

    Returns:
        An approximation of $f'_i(x_0, \ldots, x_{n-1})$
    """
    # TODO: Implement for Task 1.1.
    vals1 = [v for v in vals]
    vals2 = [v for v in vals]
    vals1[arg] = vals1[arg] + epsilon
    vals2[arg] = vals2[arg] - epsilon
    delta = f(*vals1) - f(*vals2)
    return delta / (2 * epsilon)
    # raise NotImplementedError("Need to implement for Task 1.1")


variable_count = 1


class Variable(Protocol):
    def accumulate_derivative(self, x: Any) -> None:
        pass

    @property
    def unique_id(self) -> int:
        pass

    def is_leaf(self) -> bool:
        pass

    def is_constant(self) -> bool:
        pass

    @property
    def parents(self) -> Iterable["Variable"]:
        pass

    def chain_rule(self, d_output: Any) -> Iterable[Tuple["Variable", Any]]:
        pass


def topological_sort(variable: Variable) -> Iterable[Variable]:
    """
    Computes the topological order of the computation graph.

    Args:
        variable: The right-most variable

    Returns:
        Non-constant Variables in topological order starting from the right.
    """
    # TODO: Implement for Task 1.4.
    order: List[Variable] = []
    seen = set()

    def dfs(node: Variable) -> None:
        if node.unique_id in seen or node.is_constant():
            return
        if not node.is_leaf():
            for input_var in node.parents:
                if not input_var.is_constant():
                    dfs(input_var)
        seen.add(node.unique_id)
        order.insert(0, node)

    dfs(variable)
    return order


def backpropagate(variable: Variable, deriv: Any) -> None:
    """
    Runs backpropagation on the computation graph in order to
    compute derivatives for the leave nodes.

    Args:
        variable: The right-most variable
        deriv  : Its derivative that we want to propagate backward to the leaves.

    No return. Should write to its results to the derivative values of each leaf through `accumulate_derivative`.
    """
    # TODO: Implement for Task 1.4.
    topo_order = topological_sort(variable)
    derivs = {variable.unique_id: deriv}

    for node in topo_order:
        d_out = derivs[node.unique_id]
        # print("Processing node: ", node,"id",node.unique_id)
        if node.is_leaf():
            # print(f"Leaf node: {node}",'id',node.unique_id)
            node.accumulate_derivative(d_out)
        else:
            # self.history.last_fn._backward(h.ctx, d_output)
            # res = self.last_fn.chain_rule(self.ctx, self.inputs, d_output)
            # for v_, deriv_ in var.history.backprop_step(dict[var.unique_id]):
            # print("----------------")
            for _input, _deriv in node.chain_rule(d_out):
                if _input.is_constant():
                    continue
                derivs.setdefault(_input.unique_id, 0.0)
                derivs[_input.unique_id] = derivs[_input.unique_id] + _deriv
                # print(_input.unique_id)


@dataclass
class Context:
    """
    Context class is used by `Function` to store information during the forward pass.
    """

    no_grad: bool = False
    saved_values: Tuple[Any, ...] = ()

    def save_for_backward(self, *values: Any) -> None:
        "Store the given `values` if they need to be used during backpropagation."
        if self.no_grad:
            return
        self.saved_values = values

    @property
    def saved_tensors(self) -> Tuple[Any, ...]:
        return self.saved_values
