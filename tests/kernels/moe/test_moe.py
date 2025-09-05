# SPDX-License-Identifier: Apache-2.0
import torch

import vllm_xpu_kernels._moe_C  # noqa: F401
from tests.utils import opcheck


def test_moe_align_block_size_opcheck():
    num_experts = 4
    block_size = 4
    topk_ids = torch.randint(0,
                             num_experts, (3, 4),
                             dtype=torch.int32,
                             device='xpu')

    max_num_tokens_padded = topk_ids.numel() + num_experts * (block_size - 1)
    sorted_ids = torch.empty((max_num_tokens_padded, ),
                             dtype=torch.int32,
                             device=topk_ids.device)
    sorted_ids.fill_(topk_ids.numel())
    max_num_m_blocks = max_num_tokens_padded // block_size
    expert_ids = torch.empty((max_num_m_blocks, ),
                             dtype=torch.int32,
                             device=topk_ids.device)
    num_tokens_post_pad = torch.empty((1),
                                      dtype=torch.int32,
                                      device=topk_ids.device)

    opcheck(torch.ops._moe_C.moe_align_block_size,
            (topk_ids, num_experts, block_size, sorted_ids, expert_ids,
             num_tokens_post_pad))
