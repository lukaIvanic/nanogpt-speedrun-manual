import torch
import numpy as np
import torch.distributed as dist

from src.s00_dist_setup import device, world_size, grad_accum_steps

# -----------------------------------------------------------------------------
# Sparse Comms for bigram embedding gradient reduce-scatter
def _sparse_comms_active():
    # we count on this in order for sparse communication to be worthwhile
    return world_size == 8 and grad_accum_steps == 1

@torch.no_grad
def sparse_comms_start(idxes_np, N, rank, world, send_idxes_buffer):
    rows_per_rank = N // world

    # queue upload of indexes to gpu
    send_idxes = send_idxes_buffer[:idxes_np.shape[0]]
    send_idxes.copy_(torch.from_numpy(idxes_np))
    send_idxes = send_idxes.to(device, non_blocking=True)

    # calculate how many gradient rows we will send to every rank
    insertion_points = np.searchsorted(
        idxes_np,
        np.arange(0, rows_per_rank * (world + 1), rows_per_rank, dtype=np.int32),
    )
    send_counts = torch.from_numpy(insertion_points[1:] - insertion_points[:-1])
    # zero-out own send-count - we won't send our own gradient rows to ourselves as it's a waste:
    # in sparse_comms_merge_gradients, we'll use the slice of the gradient that already includes them as the base tensor
    send_counts[rank] = 0

    # remove indexes owned by our rank from the send list
    send_idxes = torch.cat([send_idxes[: insertion_points[rank]], send_idxes[insertion_points[rank + 1] :]])

    # share the send counts so that each rank will know how many rows
    # to expect from every other rank
    recv_counts = torch.empty_like(send_counts)
    recv_counts_fut = dist.all_to_all_single(recv_counts, send_counts, async_op=True).get_future()
    return send_idxes, send_counts, recv_counts, recv_counts_fut

@torch.no_grad
def sparse_comms_share_indexes(send_idxes, send_counts, recv_counts):
    # cpu tensors, so these ops are cheap and don't force a host<->device sync
    total_recv_count = recv_counts.sum().item()
    recv_counts = recv_counts.tolist()
    send_counts = send_counts.tolist()

    # queue sharing of row indexes
    recv_idxes = torch.empty(total_recv_count, dtype=torch.int32, device=device)
    idxes_fut = dist.all_to_all_single(
        recv_idxes,
        send_idxes,
        output_split_sizes=recv_counts,
        input_split_sizes=send_counts,
        async_op=True,
    ).get_future()

    sparse_state = {
        "send_idxes": send_idxes,
        "send_counts": send_counts,
        "recv_counts": recv_counts, # list for sharing
    }
    return recv_idxes, sparse_state, idxes_fut

@torch.compile
@torch.no_grad
def sparse_comms_share_gradients(grad, idxes, send_counts, recv_counts):
    # gather the rows that we want to send
    send_vals = grad[idxes]

    d = grad.shape[1]

    send_sizes = [i*d for i in send_counts]
    recv_sizes = [i*d for i in recv_counts]

    recv_vals = torch.empty(sum(recv_sizes), device=send_vals.device, dtype=grad.dtype)

    val_fut = dist.all_to_all_single(
        recv_vals,
        send_vals.view(-1),
        input_split_sizes=send_sizes,
        output_split_sizes=recv_sizes,
        async_op=True,
    ).get_future()

    return recv_vals, val_fut

@torch.no_grad
def sparse_comms_merge_gradients(grad, recv_idx, recv_vals, rank, world):
    d = grad.shape[1]
    rows_per_rank = grad.shape[0] // world

    grad.index_add_(0, recv_idx, recv_vals.view(-1, d))

    # return the slice of the gradient for parameters our rank updates
    return grad[rows_per_rank * rank : rows_per_rank * (rank + 1)].mul_((1 / world))
