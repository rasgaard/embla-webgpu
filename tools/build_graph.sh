#!/bin/sh
# Build the browser graph models/graph/embla_step.onnx (+ .data): export, then four ONNX rewrites (see optimizations.md).
#   export_onnx.py              torch.onnx export of one step (~1.5 min)
#   patch_syncs.py              remove the 40 mid-graph GPU->CPU syncs (int64 / IsInf / Sign ops the WebGPU EP runs on the CPU)
#   patch_moe_dense.py          routed experts as two 4-bit MatMulNBits over all experts (no fp32 expert temporaries)
#   patch_rmsnorm.py            fuse the 185 decomposed RMSNorms into SimplifiedLayerNormalization
#   patch_fuse_gating_rope.py   RoPE -> RotaryEmbedding, AttnRes logit stacks built once, common-subexpression elimination
# The rewrites only touch the .onnx; the 0.48 GB weights file is hard-linked from step to step.
# usage (from the repo root): sh tools/build_graph.sh
set -e
cd "$(dirname "$0")/.."
[ -f tools/embla_ref/modeling_embla.py ] || uv run python tools/download.py
[ -f build/export/embla_step.onnx ] || uv run python tools/export_onnx.py build/export/embla_step.onnx
rm -rf build/a build/b build/c models/graph
uv run python tools/patch_syncs.py build/export/embla_step.onnx build/a/embla_step.onnx
uv run python tools/patch_moe_dense.py build/a/embla_step.onnx build/b/embla_step.onnx
uv run python tools/patch_rmsnorm.py build/b/embla_step.onnx build/c/embla_step.onnx
uv run python tools/patch_fuse_gating_rope.py build/c/embla_step.onnx models/graph/embla_step.onnx
rm -rf build/a build/b build/c
cp tools/graph_card.md models/graph/README.md                 # the Hugging Face model card (hf upload <repo> models/graph .)
echo "done: models/graph/embla_step.onnx + .data"
