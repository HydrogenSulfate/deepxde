"""Utilities of paddle."""
import paddle


def gather_nprocs(tensor, axis=0):
    """gather tensor from all GPUs and concatenate them along given axis

    Args:
        tensor (paddle.Tensor): tensor to be gathered from all GPUs
        axis (int, optional): concat axis. Defaults to 0.

    Returns:
        paddle.Tensor: gathered Tensor
    """
    tensor_list = []
    paddle.distributed.all_gather(tensor_list, tensor)
    return paddle.concat(tensor_list, axis)


def get_nprocs_and_rank():
    nprocs = paddle.distributed.get_world_size()
    rank = paddle.distributed.get_rank()
    return nprocs, rank
