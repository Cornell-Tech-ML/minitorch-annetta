import random
from typing import Callable, Dict, Iterable, List, Tuple
import minitorch
from minitorch import Tensor
import numba
from numba import cuda
import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis.strategies import DataObject, data, integers, lists, permutations

import minitorch
from minitorch import MathTestVariable, Tensor, TensorBackend, grad_check

from .strategies import assert_close, small_floats
from .tensor_strategies import assert_close_tensor, shaped_tensors, tensors


@pytest.mark.task4_4
@given(tensors(shape=(2, 3, 4)))
def test_cuda(t: Tensor) -> None:

    t = minitorch.tensor([0, 1, 2, 3]).view(1, 1, 4)
    t.requires_grad_(True)
    t2 = minitorch.tensor([[1, 2, 3]]).view(1, 1, 3)
    out = minitorch.Conv1dFun.apply(t, t2)
    assert out[0, 0, 0] == 0 * 1 + 1 * 2 + 2 * 3
    assert out[0, 0, 1] == 1 * 1 + 2 * 2 + 3 * 3
    assert out[0, 0, 2] == 2 * 1 + 3 * 2
    assert out[0, 0, 3] == 3 * 1
