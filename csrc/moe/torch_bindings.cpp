#include "core/registration.h"
#include "moe_ops.h"

TORCH_LIBRARY_EXPAND(TORCH_EXTENSION_NAME, m) {
  // Calculate the result of moe by summing up the partial results
  // from all selected experts.
  m.def("moe_sum(Tensor input, Tensor! output) -> ()");
  m.impl("moe_sum", torch::kXPU, &moe_sum);

  m.def(
      "moe_align_block_size(Tensor topk_ids, int num_experts,"
      "                     int block_size, Tensor! sorted_token_ids,"
      "                     Tensor! experts_ids,"
      "                     Tensor! num_tokens_post_pad) -> ()");
  m.impl("moe_align_block_size", torch::kXPU, &moe_align_block_size);
}

REGISTER_EXTENSION(TORCH_EXTENSION_NAME)
