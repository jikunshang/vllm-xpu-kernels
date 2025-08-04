# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch

DTYPES = [torch.half, torch.bfloat16]


@pytest.mark.parametrize("dtype", DTYPES)
@torch.inference_mode()
def test_cutlass_op(dtype: torch.dtype, ):
    torch.set_default_device("xpu")
    a = torch.zeros((2, 3), dtype=dtype, device="xpu")
    torch.ops._C.cutlass_sycl_demo(a)
