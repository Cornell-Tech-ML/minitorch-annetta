from typing import Tuple
from .autodiff import Context
from .tensor import Tensor
from .tensor_functions import Function

from typing import Callable, Optional

import numba
from numba import cuda

from .tensor import Tensor
from .tensor_data import (
    MAX_DIMS,
    Shape,
    Storage,
    Strides,
    TensorData,
    broadcast_index,
    index_to_position,
    shape_broadcast,
    to_index,
)
from .tensor_ops import MapProto, TensorOps

# This code will CUDA compile fast versions your tensor_data functions.
# If you get an error, read the docs for NUMBA as to what is allowed
# in these functions.

to_index = cuda.jit(device=True)(to_index)
index_to_position = cuda.jit(device=True)(index_to_position)
broadcast_index = cuda.jit(device=True)(broadcast_index)

THREADS_PER_BLOCK = 32


def _cuda_tensor_conv1d(
    out: Tensor,
    out_shape: Shape,
    out_strides: Strides,
    out_size: int,
    input: Tensor,
    input_shape: Shape,
    input_strides: Strides,
    weight: Tensor,
    weight_shape: Shape,
    weight_strides: Strides,
    reverse: bool,
) -> None:
    """
    1D Convolution implementation.

    Given input tensor of

       `batch, in_channels, width`

    and weight tensor

       `out_channels, in_channels, k_width`

    Computes padded output of

       `batch, out_channels, width`

    `reverse` decides if weight is anchored left (False) or right.
    (See diagrams)

    Args:
        out (Storage): storage for `out` tensor.
        out_shape (Shape): shape for `out` tensor.
        out_strides (Strides): strides for `out` tensor.
        out_size (int): size of the `out` tensor.
        input (Storage): storage for `input` tensor.
        input_shape (Shape): shape for `input` tensor.
        input_strides (Strides): strides for `input` tensor.
        weight (Storage): storage for `input` tensor.
        weight_shape (Shape): shape for `input` tensor.
        weight_strides (Strides): strides for `input` tensor.
        reverse (bool): anchor weight at left or right
    """
    batch_, out_channels, out_width = out_shape
    batch, in_channels, width = input_shape
    out_channels_, in_channels_, kw = weight_shape

    assert (
        batch == batch_
        and in_channels == in_channels_
        and out_channels == out_channels_
    )
    s1 = input_strides
    s2 = weight_strides

    # TODO: Implement for Task 4.1.
    i = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
    if i < out_size:
        out_index = cuda.local.array(MAX_DIMS, numba.int32)
        to_index(i, out_shape, out_index)
        total = 0.0
        for j in range(input_shape[1]):  # in_channels
            for k in range(weight_shape[2]):  # kw
                w_i = cuda.local.array([out_index[1], j, k], numba.int32)
                w_p = index_to_position(w_i, weight_strides)
                if reverse:
                    if out_index[2] - k >= 0:
                        in_index = cuda.local.array([out_index[0], j, out_index[2] - k], numba.int32)
                        in_pos = index_to_position(in_index, input_strides)
                        total += input[in_pos] * weight[w_p]
                else:
                    if input_shape[2] > out_index[2] + k:
                        in_index = cuda.local.array([out_index[0], j, out_index[2] + k], numba.int32)
                        in_pos = index_to_position(in_index, input_strides)
                        total += input[in_pos] * weight[w_p]
        out[i] = total
    # raise NotImplementedError("Need to implement for Task 4.1")
    

cuda_tensor_conv1d = cuda.jit(device=True)(_cuda_tensor_conv1d)


class Conv1dFun(Function):
    @staticmethod
    def forward(ctx: Context, input: Tensor, weight: Tensor) -> Tensor:
        """
        Compute a 1D Convolution

        Args:
            ctx : Context
            input : batch x in_channel x h x w
            weight : out_channel x in_channel x kh x kw

        Returns:
            batch x out_channel x h x w
        """
        ctx.save_for_backward(input, weight)
        batch, in_channels, w = input.shape
        out_channels, in_channels2, kw = weight.shape
        assert in_channels == in_channels2

        # Run convolution
        output = input.zeros((batch, out_channels, w))
    
        threads_per_block = 256
        blocks_per_grid = (output.size + (threads_per_block - 1)) // threads_per_block
        cuda_tensor_conv1d[blocks_per_grid, threads_per_block](
            *output.tuple(), output.size, *input.tuple(), *weight.tuple(), False
        )
        return output

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tuple[Tensor, Tensor]:
        input, weight = ctx.saved_values
        batch, in_channels, w = input.shape
        out_channels, in_channels, kw = weight.shape
        grad_weight = grad_output.zeros((in_channels, out_channels, kw))
        new_input = input.permute(1, 0, 2)
        new_grad_output = grad_output.permute(1, 0, 2)
        threads_per_block = 256
        blocks_per_grid = (grad_weight.size + (threads_per_block - 1)) // threads_per_block
        cuda_tensor_conv1d[blocks_per_grid, threads_per_block](
            *grad_weight.tuple(),
            grad_weight.size,
            *new_input.tuple(),
            *new_grad_output.tuple(),
            False,
        )
        grad_weight = grad_weight.permute(1, 0, 2)

        grad_input = input.zeros((batch, in_channels, w))
        new_weight = weight.permute(1, 0, 2)
        threads_per_block = 256
        blocks_per_grid = (grad_input.size + (threads_per_block - 1)) // threads_per_block
        cuda_tensor_conv1d[blocks_per_grid, threads_per_block](
            *grad_input.tuple(),
            grad_input.size,
            *grad_output.tuple(),
            *new_weight.tuple(),
            True,
        )
        return grad_input, grad_weight


cuda_conv1d = Conv1dFun.apply


def _cuda_tensor_conv2d(
    out: Tensor,
    out_shape: Shape,
    out_strides: Strides,
    out_size: int,
    input: Tensor,
    input_shape: Shape,
    input_strides: Strides,
    weight: Tensor,
    weight_shape: Shape,
    weight_strides: Strides,
    reverse: bool,
) -> None:
    """
    2D Convolution implementation.

    Given input tensor of

       `batch, in_channels, height, width`

    and weight tensor

       `out_channels, in_channels, k_height, k_width`

    Computes padded output of

       `batch, out_channels, height, width`

    `Reverse` decides if weight is anchored top-left (False) or bottom-right.
    (See diagrams)


    Args:
        out (Storage): storage for `out` tensor.
        out_shape (Shape): shape for `out` tensor.
        out_strides (Strides): strides for `out` tensor.
        out_size (int): size of the `out` tensor.
        input (Storage): storage for `input` tensor.
        input_shape (Shape): shape for `input` tensor.
        input_strides (Strides): strides for `input` tensor.
        weight (Storage): storage for `input` tensor.
        weight_shape (Shape): shape for `input` tensor.
        weight_strides (Strides): strides for `input` tensor.
        reverse (bool): anchor weight at top-left or bottom-right
    """
    batch_, out_channels, _, _ = out_shape
    batch, in_channels, height, width = input_shape
    out_channels_, in_channels_, kh, kw = weight_shape

    assert (
        batch == batch_
        and in_channels == in_channels_
        and out_channels == out_channels_
    )

    s1 = input_strides
    s2 = weight_strides
    # inners
    s10, s11, s12, s13 = s1[0], s1[1], s1[2], s1[3]
    s20, s21, s22, s23 = s2[0], s2[1], s2[2], s2[3]

    # TODO: Implement for Task 4.2.
    i = cuda.grid(1)

    if i < out_size:
        out_index = cuda.local.array(MAX_DIMS, numba.int32)
        to_index(i, out_shape, out_index)
        total = 0.0
        for j in range(input_shape[1]):  # in_channels
            for k in range(weight_shape[2]):  # kw
                w_i = cuda.local.array([out_index[1], j, k], numba.int32)
                w_p = index_to_position(w_i, weight_strides)
                if reverse:
                    if out_index[2] - k >= 0:
                        in_index = cuda.local.array([out_index[0], j, out_index[2] - k], numba.int32)
                        in_pos = index_to_position(in_index, input_strides)
                        total += input[in_pos] * weight[w_p]
                else:
                    if input_shape[2] > out_index[2] + k:
                        in_index = cuda.local.array([out_index[0], j, out_index[2] + k], numba.int32)
                        in_pos = index_to_position(in_index, input_strides)
                        total += input[in_pos] * weight[w_p]
        out[i] = total
    # raise NotImplementedError("Need to implement for Task 4.2")


cuda_tensor_conv2d = cuda.jit(device=True)(_cuda_tensor_conv2d)


class Conv2dFun(Function):
    @staticmethod
    def forward(ctx: Context, input: Tensor, weight: Tensor) -> Tensor:
        """
        Compute a 2D Convolution

        Args:
            ctx : Context
            input : batch x in_channel x h x w
            weight  : out_channel x in_channel x kh x kw

        Returns:
            (:class:`Tensor`) : batch x out_channel x h x w
        """
        ctx.save_for_backward(input, weight)
        batch, in_channels, h, w = input.shape
        out_channels, in_channels2, kh, kw = weight.shape
        assert in_channels == in_channels2
        output = input.zeros((batch, out_channels, h, w))
        cuda_tensor_conv2d(
            *output.tuple(), output.size, *input.tuple(), *weight.tuple(), False
        )
        return output

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tuple[Tensor, Tensor]:
        input, weight = ctx.saved_values
        batch, in_channels, h, w = input.shape
        out_channels, in_channels, kh, kw = weight.shape

        grad_weight = grad_output.zeros((in_channels, out_channels, kh, kw))
        new_input = input.permute(1, 0, 2, 3)
        new_grad_output = grad_output.permute(1, 0, 2, 3)
        cuda_tensor_conv2d(
            *grad_weight.tuple(),
            grad_weight.size,
            *new_input.tuple(),
            *new_grad_output.tuple(),
            False,
        )
        grad_weight = grad_weight.permute(1, 0, 2, 3)

        grad_input = input.zeros((batch, in_channels, h, w))
        new_weight = weight.permute(1, 0, 2, 3)
        cuda_tensor_conv2d(
            *grad_input.tuple(),
            grad_input.size,
            *grad_output.tuple(),
            *new_weight.tuple(),
            True,
        )
        return grad_input, grad_weight


cuda_conv2d = Conv2dFun.apply
