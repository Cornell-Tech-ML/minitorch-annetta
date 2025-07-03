from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from numba import njit, prange

from .tensor_data import (
    MAX_DIMS,
    broadcast_index,
    index_to_position,
    shape_broadcast,
    to_index,
)
from .tensor_ops import MapProto, TensorOps

if TYPE_CHECKING:
    from typing import Callable, Optional

    from .tensor import Tensor
    from .tensor_data import Shape, Storage, Strides

# TIP: Use `NUMBA_DISABLE_JIT=1 pytest tests/ -m task3_1` to run these tests without JIT.

# This code will JIT compile fast versions your tensor_data functions.
# If you get an error, read the docs for NUMBA as to what is allowed
# in these functions.
to_index = njit(inline="always")(to_index)
index_to_position = njit(inline="always")(index_to_position)
broadcast_index = njit(inline="always")(broadcast_index)


class FastOps(TensorOps):
    @staticmethod
    def map(fn: Callable[[float], float]) -> MapProto:
        "See `tensor_ops.py`"

        # This line JIT compiles your tensor_map
        f = tensor_map(njit()(fn))

        def ret(a: Tensor, out: Optional[Tensor] = None) -> Tensor:
            if out is None:
                out = a.zeros(a.shape)
            f(*out.tuple(), *a.tuple())
            return out

        return ret

    @staticmethod
    def zip(fn: Callable[[float, float], float]) -> Callable[[Tensor, Tensor], Tensor]:
        "See `tensor_ops.py`"

        f = tensor_zip(njit()(fn))

        def ret(a: Tensor, b: Tensor) -> Tensor:
            c_shape = shape_broadcast(a.shape, b.shape)
            out = a.zeros(c_shape)
            f(*out.tuple(), *a.tuple(), *b.tuple())
            return out

        return ret

    @staticmethod
    def reduce(
        fn: Callable[[float, float], float], start: float = 0.0
    ) -> Callable[[Tensor, int], Tensor]:
        "See `tensor_ops.py`"
        f = tensor_reduce(njit()(fn))

        def ret(a: Tensor, dim: int) -> Tensor:
            out_shape = list(a.shape)
            out_shape[dim] = 1

            # Other values when not sum.
            out = a.zeros(tuple(out_shape))
            out._tensor._storage[:] = start

            f(*out.tuple(), *a.tuple(), dim)
            return out

        return ret

    @staticmethod
    def matrix_multiply(a: Tensor, b: Tensor) -> Tensor:
        """
        Batched tensor matrix multiply ::

            for n:
              for i:
                for j:
                  for k:
                    out[n, i, j] += a[n, i, k] * b[n, k, j]

        Where n indicates an optional broadcasted batched dimension.

        Should work for tensor shapes of 3 dims ::

            assert a.shape[-1] == b.shape[-2]

        Args:
            a : tensor data a
            b : tensor data b

        Returns:
            New tensor data
        """

        # Make these always be a 3 dimensional multiply
        both_2d = 0
        if len(a.shape) == 2:
            a = a.contiguous().view(1, a.shape[0], a.shape[1])
            both_2d += 1
        if len(b.shape) == 2:
            b = b.contiguous().view(1, b.shape[0], b.shape[1])
            both_2d += 1
        both_2d = both_2d == 2

        ls = list(shape_broadcast(a.shape[:-2], b.shape[:-2]))
        ls.append(a.shape[-2])
        ls.append(b.shape[-1])
        assert a.shape[-1] == b.shape[-2]
        out = a.zeros(tuple(ls))

        tensor_matrix_multiply(*out.tuple(), *a.tuple(), *b.tuple())

        # Undo 3d if we added it.
        if both_2d:
            out = out.view(out.shape[1], out.shape[2])
        return out


# Implementations


def tensor_map(
    fn: Callable[[float], float]
) -> Callable[[Storage, Shape, Strides, Storage, Shape, Strides], None]:
    """
    NUMBA low_level tensor_map function. See `tensor_ops.py` for description.

    Optimizations:

    * Main loop in parallel
    * All indices use numpy buffers
    * When `out` and `in` are stride-aligned, avoid indexing

    Args:
        fn: function mappings floats-to-floats to apply.

    Returns:
        Tensor map function.
    """

    def _map(
        out: Storage,
        out_shape: Shape,
        out_strides: Strides,
        in_storage: Storage,
        in_shape: Shape,
        in_strides: Strides,
    ) -> None:
        # TODO: Implement for Task 3.1.
        if (
            len(out_strides) == len(in_strides)
            and (out_strides == in_strides).all()
            and (out_shape == in_shape).all()
        ):
            for i in prange(len(out)):
                out[i] = fn(in_storage[i])
        else:
            for i in prange(len(out)):
                out_index = np.empty(MAX_DIMS, dtype=np.int32)
                in_index = np.empty(MAX_DIMS, dtype=np.int32)
                to_index(i, out_shape, out_index)
                o_i = index_to_position(out_index, out_strides)
                broadcast_index(out_index, out_shape, in_shape, in_index)
                out[o_i] = fn(in_storage[index_to_position(in_index, in_strides)])
        # raise NotImplementedError("Need to implement for Task 3.1")

    return njit(parallel=True)(_map)  # type: ignore


def tensor_zip(
    fn: Callable[[float, float], float]
) -> Callable[
    [Storage, Shape, Strides, Storage, Shape, Strides, Storage, Shape, Strides], None
]:
    """
    NUMBA higher-order tensor zip function. See `tensor_ops.py` for description.


    Optimizations:

    * Main loop in parallel
    * All indices use numpy buffers
    * When `out`, `a`, `b` are stride-aligned, avoid indexing

    Args:
        fn: function maps two floats to float to apply.

    Returns:
        Tensor zip function.
    """

    def _zip(
        out: Storage,
        out_shape: Shape,
        out_strides: Strides,
        a_storage: Storage,
        a_shape: Shape,
        a_strides: Strides,
        b_storage: Storage,
        b_shape: Shape,
        b_strides: Strides,
    ) -> None:
        # TODO: Implement for Task 3.1.
        if (
            len(out_strides) == len(a_strides)
            and len(out_strides) == len(b_strides)
            and (out_strides == a_strides).all()
            and (out_strides == b_strides).all()
            and (out_shape == a_shape).all()
            and (out_shape == b_shape).all()
        ):
            for i in prange(len(out)):
                out[i] = fn(a_storage[i], b_storage[i])
        else:
            for i in prange(len(out)):
                out_index = np.empty(MAX_DIMS, dtype=np.int32)
                a_index = np.empty(MAX_DIMS, dtype=np.int32)
                b_index = np.empty(MAX_DIMS, dtype=np.int32)
                to_index(i, out_shape, out_index)
                o_i = index_to_position(out_index, out_strides)
                broadcast_index(out_index, out_shape, a_shape, a_index)
                a_i = index_to_position(a_index, a_strides)
                broadcast_index(out_index, out_shape, b_shape, b_index)
                b_i = index_to_position(b_index, b_strides)
                out[o_i] = fn(a_storage[a_i], b_storage[b_i])
        # raise NotImplementedError("Need to implement for Task 3.1")

    return njit(parallel=True)(_zip)  # type: ignore


def tensor_reduce(
    fn: Callable[[float, float], float]
) -> Callable[[Storage, Shape, Strides, Storage, Shape, Strides, int], None]:
    """
    NUMBA higher-order tensor reduce function. See `tensor_ops.py` for description.

    Optimizations:

    * Main loop in parallel
    * All indices use numpy buffers
    * Inner-loop should not call any functions or write non-local variables

    Args:
        fn: reduction function mapping two floats to float.

    Returns:
        Tensor reduce function
    """

    def _reduce(
        out: Storage,
        out_shape: Shape,
        out_strides: Strides,
        a_storage: Storage,
        a_shape: Shape,
        a_strides: Strides,
        reduce_dim: int,
    ) -> None:
        # TODO: Implement for Task 3.1.
        for i in prange(len(out)):
            out_index = np.empty(MAX_DIMS, dtype=np.int32)
            to_index(i, out_shape, out_index)
            out_i = index_to_position(out_index, out_strides)
            a_i = index_to_position(out_index, a_strides)
            total = out[out_i]
            for _ in range(a_shape[reduce_dim]):
                total = fn(total, a_storage[a_i])
                a_i += a_strides[reduce_dim]
            out[out_i] = total
        # raise NotImplementedError("Need to implement for Task 3.1")

    return njit(parallel=True)(_reduce)  # type: ignore


def _tensor_matrix_multiply(
    out: Storage,
    out_shape: Shape,
    out_strides: Strides,
    a_storage: Storage,
    a_shape: Shape,
    a_strides: Strides,
    b_storage: Storage,
    b_shape: Shape,
    b_strides: Strides,
) -> None:
    """
    NUMBA tensor matrix multiply function.

    Should work for any tensor shapes that broadcast as long as

    ```
    assert a_shape[-1] == b_shape[-2]
    ```

    Optimizations:

    * Outer loop in parallel
    * No index buffers or function calls
    * Inner loop should have no global writes, 1 multiply.


    Args:
        out (Storage): storage for `out` tensor
        out_shape (Shape): shape for `out` tensor
        out_strides (Strides): strides for `out` tensor
        a_storage (Storage): storage for `a` tensor
        a_shape (Shape): shape for `a` tensor
        a_strides (Strides): strides for `a` tensor
        b_storage (Storage): storage for `b` tensor
        b_shape (Shape): shape for `b` tensor
        b_strides (Strides): strides for `b` tensor

    Returns:
        None : Fills in `out`
    """
    a_batch_stride = a_strides[0] if a_shape[0] > 1 else 0
    b_batch_stride = b_strides[0] if b_shape[0] > 1 else 0
    # TODO: Implement for Task 3.2.
    if a_shape[-1] != b_shape[-2]:
        return
        # print(out_shape)
        # print('a',a_storage, a_shape)
        # print("a0",a_storage[a_strides])
        # print('b',b_storage, b_shape)
        # print('b0', b_storage[b_strides])
    for i in prange(len(out)):
        # calculate from the out_size
        # batch dimension is out_i_0
        # out_i_1 is the 1st dimension of a
        # and out_i_2 is the 2nd dimension of b
        out_i_0 = i // (out_shape[-1] * out_shape[-2])
        out_i_1 = (i % (out_shape[-1] * out_shape[-2])) // out_shape[-1]
        out_i_2 = i % out_shape[-1]
        # get the absolute position of out
        out_i = (
            out_i_0 * out_strides[0]
            + out_i_1 * out_strides[1]
            + out_i_2 * out_strides[2]
        )
        # get the absolute position of a and b
        a_i = out_i_0 * a_batch_stride + out_i_1 * a_strides[1]
        b_i = out_i_0 * b_batch_stride + out_i_2 * b_strides[2]
        # use total to store the product of a and b
        total = 0.0
        for j in range(a_shape[2]):
            total += a_storage[a_i] * b_storage[b_i]
            # update the position of a and b
            a_i += a_strides[2]
            b_i += b_strides[1]
        # write the total to out
        out[out_i] = total

        # for i in prange(out_shape[0]):
        #     for j in prange(out_shape[1]):
        #         for k in prange(out_shape[2]):
        #             a_i = i * a_batch_stride + j * a_strides[1]
        #             b_i = i * b_batch_stride + k * b_strides[2]
        #             total = 0.0
        #             for _ in range(a_shape[2]):
        #                 total += a_storage[a_i] * b_storage[b_i]
        #                 a_i += a_strides[2]
        #                 b_i += b_strides[1]
        #             out_i = i * out_strides[0] + j * out_strides[1] + k * out_strides[2]
        #             out[out_i] = total

        # out_index = np.zeros(len(out_shape), dtype=np.int32)
        # a_index = np.zeros(MAX_DIMS, dtype=np.int32)
        # b_index = np.zeros(MAX_DIMS, dtype=np.int32)
        # to_index(i, out_shape, out_index)
        # broadcast_index(out_index, out_shape, a_shape, a_index)
        # broadcast_index(out_index, out_shape, b_shape, b_index)
        # a_data = a_storage[index_to_position(a_index, a_strides)]
        # b_data = b_storage[index_to_position(b_index, b_strides)]
        # out[index_to_position(out_index, out_strides)] += a_data * b_data
    # raise NotImplementedError("Need to implement for Task 3.2")


tensor_matrix_multiply = njit(parallel=True, fastmath=True)(_tensor_matrix_multiply)
