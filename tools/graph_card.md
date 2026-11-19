---
license: mit
base_model: sevren-ai/Embla1-4bit
library_name: onnx
pipeline_tag: text-generation
tags: [embla, webgpu, onnxruntime-web, onnx]
---

# Embla1-4bit, WebGPU step graph

The ONNX graph that runs prefill for Embla 1 on WebGPU, a browser engine for
[`sevren-ai/Embla1-4bit`](https://huggingface.co/sevren-ai/Embla1-4bit). One step of the model (a prefill chunk of up to 64
tokens, or one decode token) with the KV / engram-conv cache as explicit inputs and outputs, exported from the PyTorch reference
and rewritten for onnxruntime-web's WebGPU execution provider.

- `embla_step.onnx` + `embla_step.onnx.data` (0.48 GB): the non-expert weights in MLX's 4-bit format, as `MatMulNBits` with an
  exact offset term for the affine bias.
- The routed experts (3.9 GB) are **graph inputs**, not part of this repo. The engine streams them from
  `sevren-ai/Embla1-4bit/model.safetensors`. The engram rows are an input too (looked up in that repo's `engram_table.safetensors`).

Build: `sh tools/build_graph.sh` in the GitHub repo (export + four graph rewrites, documented there).

Model by [sevren.ai](https://huggingface.co/sevren-ai), MIT licensed. This is a derived format of their weights, under the same license.
