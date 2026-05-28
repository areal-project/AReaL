# SPDX-License-Identifier: Apache-2.0

# Adapted from Megatron-LM

import torch

from areal.infra.platforms import current_platform


def local_multi_tensor_applier(op, noop_flag_buffer, tensor_lists, *args):
    return op(2048 * 32, noop_flag_buffer, tensor_lists, *args)


# computes l2 norm for a list of contiguous tensors
# works as a drop-in replacement for amp_C.multi_tensor_l2norm
def local_multi_tensor_l2_norm(chunk_size, noop_flag, tensor_lists, per_tensor, *args):
    l2 = [
        [(torch.norm(tensor)) for tensor in tensor_list] for tensor_list in tensor_lists
    ]
    l2_reduced = torch.norm(torch.tensor(l2))
    l2_cuda = torch.tensor(
        [float(l2_reduced)], dtype=torch.float, device=current_platform.device_type
    )
    return l2_cuda, None


# works as a drop-in replacement for amp_C.multi_tensor_scale
def local_multi_tensor_scale(chunk_size, noop_flag, tensor_lists, scale):
    for src, dst in zip(tensor_lists[0], tensor_lists[1]):
        dst.copy_(src * scale)
