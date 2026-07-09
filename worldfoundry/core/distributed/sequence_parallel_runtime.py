import datetime
import os
from typing import Any, Optional, Tuple

import torch
import torch.distributed as dist

from torch.nn import functional as F


class SequenceParallelInfo:
    def __init__(self):
        self.group = None
        self.sp_size = 1
        self.global_rank = 0
        self.rank_within_group = 0
        self.group_id = 0


nccl_info = SequenceParallelInfo()
_SEQUENCE_PARALLEL_STATE = False
_SEQUENCE_PARALLEL_GROUPS: dict[str, dist.ProcessGroup] = {}


def _resolve_group(group: Optional[dist.ProcessGroup] = None):
    return group if group is not None else get_sequence_parallel_group()


def _rank_in_group(group: Optional[dist.ProcessGroup] = None):
    if group is None:
        return dist.get_rank()
    return dist.get_group_rank(group, dist.get_rank())


def initialize_sequence_parallel_group(sp_size: int) -> None:
    world_size = dist.get_world_size()
    rank = dist.get_rank()
    assert world_size % sp_size == 0, "world_size must be divisible by sequence_parallel_size"

    nccl_info.sp_size = sp_size
    nccl_info.global_rank = rank
    num_sequence_parallel_groups = world_size // sp_size
    for i in range(num_sequence_parallel_groups):
        ranks = range(i * sp_size, (i + 1) * sp_size)
        group = dist.new_group(ranks)
        if rank in ranks:
            set_sequence_parallel_group(group)
            nccl_info.rank_within_group = rank - i * sp_size
            nccl_info.group_id = i


def set_sequence_parallel_group(group: dist.ProcessGroup) -> None:
    _SEQUENCE_PARALLEL_GROUPS["sequence"] = group
    nccl_info.group = group
    nccl_info.sp_size = dist.get_world_size(group)
    nccl_info.rank_within_group = dist.get_rank(group)


def get_sequence_parallel_group() -> Optional[dist.ProcessGroup]:
    return _SEQUENCE_PARALLEL_GROUPS.get("sequence", nccl_info.group)


def initialize_sequence_parallel_state(sequence_parallel_size: int):
    global _SEQUENCE_PARALLEL_STATE
    if sequence_parallel_size > 1:
        _SEQUENCE_PARALLEL_STATE = True
        initialize_sequence_parallel_group(sequence_parallel_size)
    else:
        _SEQUENCE_PARALLEL_STATE = False
        nccl_info.sp_size = 1
        nccl_info.global_rank = int(os.getenv("RANK", "0"))
        nccl_info.rank_within_group = 0
        nccl_info.group_id = int(os.getenv("RANK", "0"))


def get_sequence_parallel_state():
    return _SEQUENCE_PARALLEL_STATE


def initialize_distributed(seed):
    local_rank = int(os.getenv("RANK", 0))
    world_size = int(os.getenv("WORLD_SIZE", 1))
    # Set defaults for distributed env vars required by env:// init method
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29500")
    if world_size == 1 and not dist.is_initialized():
        # Single-GPU mode: skip full init, use default device
        torch.cuda.set_device(local_rank if torch.cuda.is_available() else 0)
    else:
        torch.cuda.set_device(local_rank)
        dist.init_process_group(
            backend="nccl",
            init_method="env://",
            timeout=datetime.timedelta(seconds=2**31 - 1),
            world_size=world_size,
            rank=local_rank,
        )
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    initialize_sequence_parallel_state(world_size)


def broadcast(input_: torch.Tensor, group: Optional[dist.ProcessGroup] = None):
    group = _resolve_group(group)
    src = dist.get_global_rank(group, 0) if group is not None else 0
    dist.broadcast(input_, src=src, group=group)
    return input_.contiguous()


def _all_to_all_4D(
    input: torch.tensor, scatter_idx: int = 2, gather_idx: int = 1, group=None
) -> torch.tensor:
    """
    all-to-all for QKV

    Args:
        input (torch.tensor): a tensor sharded along dim scatter dim
        scatter_idx (int): default 1
        gather_idx (int): default 2
        group : torch process group

    Returns:
        torch.tensor: resharded tensor (bs, seqlen/P, hc, hs)
    """
    assert (
        input.dim() == 4
    ), f"input must be 4D tensor, got {input.dim()} and shape {input.shape}"

    group = _resolve_group(group)
    seq_world_size = dist.get_world_size(group)

    if scatter_idx == 2 and gather_idx == 1:

        seq_lens = [None] * seq_world_size
        dist.all_gather_object(seq_lens, input.shape[1], group)
        # uneven
        if seq_lens[-1] != seq_lens[0]:
            assert seq_lens[0] > seq_lens[-1]
            gap = seq_lens[0] - seq_lens[-1]
            if _rank_in_group(group) == seq_world_size - 1:
                assert input.shape[1] == seq_lens[-1]
                input = F.pad(input, (0, 0, 0, 0, 0, gap))
        else:
            gap = 0

        # input (torch.tensor): a tensor sharded along dim 1 (bs, seqlen/P, hc, hs) output: (bs, seqlen, hc/P, hs)
        bs, shard_seqlen, hc, hs = input.shape
        seqlen = shard_seqlen * seq_world_size
        assert (
            hc % seq_world_size == 0
        ), f"Invalid size: {hc}, which should be divisible by {seq_world_size}"
        shard_hc = hc // seq_world_size

        # transpose groups of heads with the seq-len parallel dimension, so that we can scatter them!
        # (bs, seqlen/P, hc, hs) -reshape-> (bs, seq_len/P, P, hc/P, hs) -transpose(0,2)-> (P, seq_len/P, bs, hc/P, hs)
        input_t = (
            input.reshape(bs, shard_seqlen, seq_world_size, shard_hc, hs)
            .transpose(0, 2)
            .contiguous()
        )

        output = torch.empty_like(input_t)
        # https://pytorch.org/docs/stable/distributed.html#torch.distributed.all_to_all_single
        # (P, seq_len/P, bs, hc/P, hs) scatter seqlen -all2all-> (P, seq_len/P, bs, hc/P, hs) scatter head
        if seq_world_size > 1:
            dist.all_to_all_single(output, input_t, group=group)
        else:
            output = input_t
        # if scattering the seq-dim, transpose the heads back to the original dimension
        output = output.reshape(seqlen, bs, shard_hc, hs)

        # (seq_len, bs, hc/P, hs) -reshape-> (bs, seq_len, hc/P, hs)
        output = output.transpose(0, 1).contiguous().reshape(bs, seqlen, shard_hc, hs)
        if gap > 0:
            output = output[:, :-gap]

        return output

    elif scatter_idx == 1 and gather_idx == 2:
        # input (torch.tensor): a tensor sharded along dim 1 (bs, seqlen, hc/P, hs) output: (bs, seqlen/P, hc, hs)
        bs, seqlen, shard_hc, hs = input.shape

        hc = shard_hc * seq_world_size
        if seqlen % seq_world_size != 0:
            new_seqlen = (seqlen // seq_world_size + 1) * seq_world_size
            gap = new_seqlen - seqlen
            input = F.pad(input, (0, 0, 0, 0, 0, gap))
            bs, seqlen, shard_hc, hs = input.shape
        else:
            gap = 0

        assert seqlen % seq_world_size == 0

        shard_seqlen = seqlen // seq_world_size
        seq_world_size = dist.get_world_size(group)

        # transpose groups of heads with the seq-len parallel dimension, so that we can scatter them!
        # (bs, seqlen, hc/P, hs) -reshape-> (bs, P, seq_len/P, hc/P, hs) -transpose(0, 3)->
        # (hc/P, P, seqlen/P, bs, hs) -transpose(0, 1) -> (P, hc/P, seqlen/P, bs, hs)
        input_t = (
            input.reshape(bs, seq_world_size, shard_seqlen, shard_hc, hs)
            .transpose(0, 3)
            .transpose(0, 1)
            .contiguous()
            .reshape(seq_world_size, shard_hc, shard_seqlen, bs, hs)
        )

        output = torch.empty_like(input_t)
        # https://pytorch.org/docs/stable/distributed.html#torch.distributed.all_to_all_single
        # (P, bs x hc/P, seqlen/P, hs) scatter seqlen -all2all-> (P, bs x seq_len/P, hc/P, hs) scatter head
        if seq_world_size > 1:
            dist.all_to_all_single(output, input_t, group=group)
        else:
            output = input_t

        # if scattering the seq-dim, transpose the heads back to the original dimension
        output = output.reshape(hc, shard_seqlen, bs, hs)

        # (hc, seqlen/N, bs, hs) -tranpose(0,2)-> (bs, seqlen/N, hc, hs)
        output = output.transpose(0, 2).contiguous().reshape(bs, shard_seqlen, hc, hs)

        if (
            gap > 0
            and _rank_in_group(group) == seq_world_size - 1
        ):
            output = output[:, :-gap]

        return output
    else:
        raise RuntimeError("scatter_idx must be 1 or 2 and gather_idx must be 1 or 2")


class SeqAllToAll4D(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx: Any,
        group: dist.ProcessGroup,
        input: torch.Tensor,
        scatter_idx: int,
        gather_idx: int,
    ) -> torch.Tensor:
        ctx.group = group
        ctx.scatter_idx = scatter_idx
        ctx.gather_idx = gather_idx

        return _all_to_all_4D(input, scatter_idx, gather_idx, group=group)

    @staticmethod
    def backward(
        ctx: Any, *grad_output: torch.Tensor
    ) -> Tuple[None, torch.Tensor, None, None]:
        return (
            None,
            SeqAllToAll4D.apply(
                ctx.group, *grad_output, ctx.gather_idx, ctx.scatter_idx
            ),
            None,
            None,
        )


def all_to_all_4D(
    input_: torch.Tensor,
    group: Optional[dist.ProcessGroup] = None,
    scatter_dim: int = 2,
    gather_dim: int = 1,
):
    group = _resolve_group(group)
    return SeqAllToAll4D.apply(group, input_, scatter_dim, gather_dim)


def _all_to_all(
    input_: torch.Tensor,
    world_size: int,
    group: dist.ProcessGroup,
    scatter_dim: int,
    gather_dim: int,
):
    input_list = [
        t.contiguous() for t in torch.tensor_split(input_, world_size, scatter_dim)
    ]
    output_list = [torch.empty_like(input_list[0]) for _ in range(world_size)]
    dist.all_to_all(output_list, input_list, group=group)
    return torch.cat(output_list, dim=gather_dim).contiguous()


class _AllToAll(torch.autograd.Function):
    """All-to-all communication.

    Args:
        input_: input matrix
        process_group: communication group
        scatter_dim: scatter dimension
        gather_dim: gather dimension
    """

    @staticmethod
    def forward(ctx, input_, process_group, scatter_dim, gather_dim):
        ctx.process_group = process_group
        ctx.scatter_dim = scatter_dim
        ctx.gather_dim = gather_dim
        ctx.world_size = dist.get_world_size(process_group)
        output = _all_to_all(
            input_, ctx.world_size, process_group, scatter_dim, gather_dim
        )
        return output

    @staticmethod
    def backward(ctx, grad_output):
        grad_output = _all_to_all(
            grad_output,
            ctx.world_size,
            ctx.process_group,
            ctx.gather_dim,
            ctx.scatter_dim,
        )
        return (
            grad_output,
            None,
            None,
            None,
        )


def all_to_all(
    input_: torch.Tensor,
    group: Optional[dist.ProcessGroup] = None,
    scatter_dim: int = 2,
    gather_dim: int = 1,
):
    group = _resolve_group(group)
    return _AllToAll.apply(input_, group, scatter_dim, gather_dim)


class _AllGather(torch.autograd.Function):
    """All-gather communication with autograd support.

    Args:
        input_: input tensor
        dim: dimension along which to concatenate
    """

    @staticmethod
    def forward(ctx, input_, dim, group):
        ctx.dim = dim
        ctx.group = group
        world_size = dist.get_world_size(group)
        input_size = list(input_.size())

        sizes = [None] * world_size
        dist.all_gather_object(sizes, input_.shape, group)

        ctx.input_size = input_size[dim]

        tensor_list = [
            torch.empty(sizes[i], dtype=input_.dtype, device=input_.device)
            for i in range(world_size)
        ]
        input_ = input_.contiguous()
        dist.all_gather(tensor_list, input_, group=group)

        output = torch.cat(tensor_list, dim=dim)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        group = ctx.group
        world_size = dist.get_world_size(group)
        rank = _rank_in_group(group)
        dim = ctx.dim
        input_size = ctx.input_size

        sizes = [None] * world_size
        dist.all_gather_object(sizes, input_size, group=group)

        grad_input_list = torch.split(grad_output, sizes, dim=dim)
        grad_input = grad_input_list[rank]

        return grad_input, None, None


def all_gather(input_: torch.Tensor, dim: int = 1, group=None):
    """Performs an all-gather operation on the input tensor along the specified dimension.

    Args:
        input_ (torch.Tensor): Input tensor of shape [B, H, S, D].
        dim (int, optional): Dimension along which to concatenate. Defaults to 1.

    Returns:
        torch.Tensor: Output tensor after all-gather operation, concatenated along 'dim'.
    """
    return _AllGather.apply(input_, dim, _resolve_group(group))


def _split(input_: torch.Tensor, dim: int, group: Optional[dist.ProcessGroup]) -> torch.Tensor:
    group = _resolve_group(group)
    world_size = dist.get_world_size(group)
    rank = dist.get_rank(group)
    dim_size = input_.size(dim)
    assert (
        dim_size % world_size == 0
    ), f"The dimension to split ({dim_size}) is not a multiple of world size ({world_size})"
    output_list = torch.split(input_, dim_size // world_size, dim=dim)
    return output_list[rank].contiguous()


def _gather(input_: torch.Tensor, dim: int, group: Optional[dist.ProcessGroup]) -> torch.Tensor:
    group = _resolve_group(group)
    world_size = dist.get_world_size(group)
    input_ = input_.contiguous()
    output_list = [torch.empty_like(input_) for _ in range(world_size)]
    torch.distributed.all_gather(output_list, input_, group=group)
    return torch.cat(output_list, dim=dim).contiguous()


class _SplitForwardGatherBackward(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input_: torch.Tensor, dim: int, group: Optional[dist.ProcessGroup]):
        ctx.dim = dim
        ctx.group = group
        return _split(input_, dim, group)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        return _gather(grad_output, ctx.dim, ctx.group), None, None


class _GatherForwardSplitBackward(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input_: torch.Tensor, dim: int, group: Optional[dist.ProcessGroup]):
        ctx.dim = dim
        ctx.group = group
        return _gather(input_, dim, group)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        return _split(grad_output, ctx.dim, ctx.group), None, None


def split_forward_gather_backward(
    input_: torch.Tensor,
    dim: int,
    group: Optional[dist.ProcessGroup] = None,
) -> torch.Tensor:
    return _SplitForwardGatherBackward.apply(input_, dim, _resolve_group(group))


def gather_forward_split_backward(
    input_: torch.Tensor,
    dim: int,
    group: Optional[dist.ProcessGroup] = None,
) -> torch.Tensor:
    return _GatherForwardSplitBackward.apply(input_, dim, _resolve_group(group))
