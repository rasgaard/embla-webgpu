# Embla 1 on WebGPU

[Embla 1](https://huggingface.co/sevren-ai/Embla1-4bit) (by sevren.ai) running entirely in the browser. It is a 10B-parameter
mixture-of-experts language model (1B active per token, 3B of them in a hashed n-gram "engram" table), in its 4-bit MLX
checkpoint. Nothing leaves the machine: the weights stream from the Hugging Face Hub into GPU memory once and are cached by the
browser.

On an Apple M4 Pro in Chromium: **~180 tokens/s** decode in the chat (5.4 ms a token; 6.5 ms at 2,000+ tokens of context) and
**~100 tokens/s** prefill, with up to 8,192 tokens of context. [optimizations.md](optimizations.md) is the log of how it got there from 630 ms a token.

## How it works

- **Prefill** runs on [onnxruntime-web](https://onnxruntime.ai/) (WebGPU), on a one-step ONNX graph exported from the PyTorch
  reference and rewritten for the WebGPU execution provider (`tools/`), in chunks of 64 tokens.
- **Decode** runs on a hand-written WGSL decoder (`src/wgsl_decode.mjs`, ~440 dispatches a token):
  - 4-bit GEMVs read MLX's packed format directly, and only the 8 routed experts a token uses;
  - Block AttnRes, sliding / full / sparse attention, router and engram each have their own kernels;
  - sampling runs on the GPU;
  - the 1.7 GB engram table lives on the GPU and is hashed there (for prefill too), so the next token's step is queued before the
    current one is read back.
- The KV cache moves between the two engines on the GPU. The routed experts (3.9 GB) stream from `model.safetensors` straight into
  GPU buffers that both engines share.
- The tokenizer is [Transformers.js](https://github.com/huggingface/transformers.js).

## Run it

You need a Chromium-based browser with WebGPU and the `subgroups` feature (recent Chrome / Edge on desktop) and about 7 GB of GPU
memory (unified memory on Apple silicon). The first load downloads about 6.1 GB; later loads read from the browser cache.
Without `subgroups`, decode falls back to the ONNX graph, which is much slower.

Locally:

```sh
node serve.mjs          # http://localhost:8000/
```

`index.html` is the chat, and `bench.html` runs the correctness and speed checks. The model files come from the Hugging Face Hub:
- the weights, engram table, tokenizer and config from [`sevren-ai/Embla1-4bit`](https://huggingface.co/sevren-ai/Embla1-4bit),
  pinned to a revision;
- the ONNX graph from [`rasgaard/Embla1-4bit-webgpu`](https://huggingface.co/rasgaard/Embla1-4bit-webgpu).

Both are set in `src/config.mjs`. The server needs HTTP Range support, which is why it isn't `python -m http.server`.

## Build the graph yourself

The ONNX graph is the only derived artifact. To rebuild it you need [uv](https://docs.astral.sh/uv/), ~6 GB of disk for the
weights and a few minutes:

```sh
uv run python tools/download.py   # weights -> models/Embla1-4bit, reference code -> tools/embla_ref (pinned revisions)
sh tools/build_graph.sh           # export + 4 rewrites -> models/graph/embla_step.onnx (+ .data, 0.48 GB)
uv run python tools/make_reference.py   # optional: regenerate test/reference.json (PyTorch, fp32)
```

With `models/` populated, `serve.mjs` serves the local copies and the pages use them automatically (`?source=hub` forces the Hub).

| Step | What it does |
|---|---|
| `export_onnx.py` | `torch.onnx` export of one step, with the cache as inputs / outputs and the routed experts as inputs; dense layers as `MatMulNBits` (`fused.py`), export blockers patched (`export_patches.py`, `q4_patches.py`) |
| `patch_syncs.py` | removes 40 mid-graph GPU-to-CPU syncs (int64 / `IsInf` / `Sign` ops the WebGPU EP runs on the CPU) |
| `patch_moe_dense.py` | routed experts as two 4-bit `MatMulNBits` over all experts (good for 64-token prefill chunks) |
| `patch_rmsnorm.py` | 185 decomposed RMSNorms to `SimplifiedLayerNormalization` |
| `patch_fuse_gating_rope.py` | `RotaryEmbedding`, AttnRes stacks built once, common-subexpression elimination |

To host the graph yourself: `uv run hf upload <you>/<repo> models/graph .`, then point `GRAPH` in `src/config.mjs` at it.

## Checks

`bench.html` loads the model and reports:
- the 16 greedy tokens against the PyTorch reference (`test/reference.json`), expected 16/16;
- the GPU engram lookup against the JS one, expected maxdiff 0;
- WGSL decoder vs ONNX graph logits from the same state, about 7e-5;
- prefill and decode speed.

Options:
- `?ctx=2600` prefills filler first (long context, which exercises sparse-attention key selection);
- `?cmp=8` sets the steps compared;
- `?n=` sets the decode tokens timed;
- `?prof=1` gives per-kernel GPU timestamps;
- `?ablate=1` gives the cost of each kernel class;
- `?decode=ort` decodes on the ONNX graph.

## Layout

```
index.html            chat
bench.html            correctness and speed checks
src/
  engine.mjs          loading, cache handoff, prefill (ORT) / decode (WGSL), sampling
  wgsl_decode.mjs     the WGSL decode step
  experts.mjs         routed experts: safetensors -> GPU buffers
  engram_table.mjs    engram table over HTTP Range requests
  engram_hash.mjs     engram n-gram hashing
  weight_cache.mjs    Cache API store for downloaded weights
  config.mjs          where the model files come from
  ort.mjs             onnxruntime-web 1.26.0 from jsDelivr
tools/                Python: download, export, graph rewrites, reference vectors
test/reference.json   PyTorch reference tokens
serve.mjs             local server with Range support
```

## Credits

- Embla 1 and its reference implementation: [sevren.ai](https://huggingface.co/sevren-ai) (MIT).
- The WGSL decoder follows the structure of their Metal engine, [embla-metal](https://github.com/sevren-ai/embla-metal).
- Built on [onnxruntime-web](https://github.com/microsoft/onnxruntime) and [Transformers.js](https://github.com/huggingface/transformers.js).
