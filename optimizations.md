# Optimization log

How Embla 1 (4-bit) in the browser went from **630 ms to 5.4 ms per token** of decode (1.6 to ~185 tokens/s), and prefill from
~4 to ~100 tokens/s. In order, with what was measured, what helped, and what didn't.

All numbers are from one machine: an **Apple M4 Pro** (273 GB/s memory bandwidth), Chromium, files served from localhost. They show
relative progress, not a benchmark. "Decode" is the time per generated token with the context given; "chat" is end-to-end tokens/s in
`index.html`, including sampling and rendering.

Every step was checked for correctness before its timing counted:
- **PyTorch reference:** a Danish chat prompt and its 16 greedy tokens from sevren-ai's `modeling_embla.py` (fp32, real engram
  table), from `tools/make_reference.py`. The browser must produce the same 16 tokens.
- **Same-state comparisons:** the WGSL decoder against the ORT graph, logits after the same token from the same cache, at short
  and long context (sparse attention only selects keys once the context passes 1,152 tokens).
- **Long greedy runs:** 121- to 300-token GPU-sampled greedy sequences, identical before and after each change.

| Milestone | Decode | Prefill | Chat |
|---|---|---|---|
| First working graph in the browser | 630 ms | ~4 tok/s | |
| Fused 4-bit ops in the graph | 155 ms | ~13 tok/s | ~6 tok/s |
| ORT engine and graph rewrites | 66 ms | ~100 tok/s | ~13 tok/s |
| Hand-written WGSL decoder | 17.4 ms | ~100 tok/s | 47-49 tok/s |
| Coalesced GEMVs, GPU sampling | 12.1 ms | | 58 tok/s |
| No CPU in the decode loop | 7.2 ms | | 136 tok/s |
| Long-context kernels, tuning | 6.6 ms | | 147 tok/s |
| Subgroup mixes, packed scales | **5.4 ms** (6.5 ms at 2.2K context) | **~100 tok/s** | **180 tok/s** |

## 1. Making it run at all

The model (`sevren-ai/Embla1-4bit`: 10B parameters, 1B active, 96 routed experts a layer, Block AttnRes, sliding / full / sparse
attention, a hashed n-gram "engram" table of 23.5M rows) was not written for export:

- **`torch.export`** fails on the MoE's per-expert Python loop, the chunked key selection and sparse attention loops, and the engram
  lookup (`torch.unique` and a host-side table). `tools/export_patches.py` replaces the four constructs with export-friendly
  equivalents:
  - a gather-based MoE;
  - fixed-k key selection with -inf padding;
  - loop-free sparse attention;
  - engram rows as a graph input.

  Patched eager PyTorch matches stock eager to 2e-6.
- **Opset 18**, not 23: ORT-web's WebGPU `RMSNormalization` (opset 23) failed with a scale-size error.
- **fp32 activations.** fp16 overflowed: the shared expert's ReLU² reaches ~15,000 by layer 2. bf16 isn't available on WebGPU.
- **Routed experts as graph inputs.** 3.9 GB of weights cannot live in the WASM heap, so `src/experts.mjs` streams them from
  `model.safetensors` straight into GPU buffers.
- **No uint8 gathers.** ORT's WebGPU `Gather` on uint8 falls back to the CPU and copies the whole tensor (+3.7 GB heap, then
  `bad_alloc`). The fix was to keep MLX's uint32 words, unpack nibbles with exact float math, and use flat `Gather` instead of `GatherND`.
- **Engram hashing in JS** (`src/engram_hash.mjs`, a BigInt port) matches PyTorch's row indices exactly. The 1.7 GB table is read
  by 4 KB page with HTTP Range requests.
- `graphOptimizationLevel: "disabled"`: ORT's constant folding expanded the dequantization subgraphs into multi-GB fp32 tensors.

Result: the real model in the browser, 16/16 tokens equal to PyTorch, **~630 ms a token**.

## 2. Fused 4-bit ops in the graph: 630 to 155 ms

The in-graph float nibble unpack cost ~7 ms per 2048x2048 layer. MLX's affine 4-bit `w = s*q + b` can be rewritten exactly as
`s*(q-8) + c` with `c = b + 8s`, which maps onto ORT's contrib ops (`tools/fused.py`):

- **Dense layers:** `MatMulNBits(x; q, s)` plus `groupsum(x) @ C^T` for the offset. ~1 ms a layer, error ~8e-6.
- **Routed experts:** `GatherBlockQuantized` plus a gathered offset.

`MatMulNBits` can't take MLX's offsets directly: its zero points must be integers.

Tried and slower: a lookup-table unpack (18 ms a layer).

## 3. The ORT engine and four graph rewrites: 155 to 66 ms, prefill to ~100 tok/s

Engine changes, for 128 ms:
- The KV cache stays on the GPU between steps (`preferredOutputLocation: "gpu-buffer"`), and logits are read only for the last chunk.
- A 60-byte bootstrap session hands over ORT's `GPUDevice` early, so the experts stream to the GPU while the graph compiles.
- A warm-up moves shader compilation into the load.

Graph rewrites, which are ONNX surgery with no re-export (`tools/build_graph.sh`):

1. **`patch_syncs.py`: 128 to 107 ms.** ORT's WebGPU EP lacks int64 `Reshape` / `Where` / `Concat` / `Cast` and `IsInf` / `Sign`
   kernels, so those nodes ran on the CPU behind a `MemcpyToHost`: 40 GPU drains a step, found with ORT's verbose node-placement log.
   The fixes:
   - int64 to int32 as `Gather(arange_int32, idx)`, since `Gather` is the one WebGPU op that takes int64 indices;
   - `IsInf(x)` to `x < -1e30`;
   - `GatherND` to a flat `Gather`;
   - `Sign` from two comparisons.
2. **`patch_moe_dense.py`: 107 to 73 ms, prefill ~23 to ~96 tok/s.** The gathered MoE dequantized the 8 chosen experts to fp32
   and ran a generic Einsum, ~8 GB of traffic a token. Now each MoE layer is two `MatMulNBits` over **all 96 experts** (4-bit,
   ~100 MB a layer), with routing weights that are zero for unchosen experts. The math is exact; only the summation order differs.
   Prefill cost no longer grows with chunk length, so chunks went from 8 to 64 tokens. The loader transposes the projection experts
   to `[out, expert*in]` while uploading.
3. **`patch_rmsnorm.py`: 73 to 66 ms.** 185 decomposed RMSNorms (6-7 nodes each) became `SimplifiedLayerNormalization`.
4. **`patch_fuse_gating_rope.py`: ~3%.**
   - RoPE (30 sites, ~11 nodes each) became one `RotaryEmbedding` with a negated sin table.
   - Each AttnRes source stack is built once instead of per mix.
   - Common-subexpression elimination.

   Graph size: 5.5k to 4.5k nodes.

At this point ORT spent most of a step dispatching ~3,800 nodes from the CPU.

Not kept:
- **fp16 expert matmuls:** ~6% faster decode and ~12% faster prefill, but logits drifted from fp32 by up to 0.25 after long
  prompts. ORT's fp16 kernels accumulate less precisely.
- **A graph split at every MoE layer** with a WGSL kernel reading only the 8 chosen experts between ORT segments: 47 ms, but
  CPU-bound on 32 `session.run()` calls a token. The WGSL decoder made it obsolete.
- `graphOptimizationLevel: "extended"`, EP validation off, `accuracy_level=4`: no change.
- Giving ORT our own `GPUDevice`: broke its readbacks.

## 4. A hand-written WGSL decoder: 66 to 17.4 ms

Prompted by [sevren-ai/embla-metal](https://github.com/sevren-ai/embla-metal) (the authors' Metal engine, ~220 dispatches a token).
It showed the remaining gap was per-node overhead, not arithmetic.

The decode step is now ~20 kinds of WGSL kernel written from `modeling_embla.py` (`src/wgsl_decode.mjs`), fp32 throughout:
- **4-bit GEMV** directly on MLX's layout, reading `model.safetensors` (the routed experts reuse the ORT graph's GPU buffers).
- **Block AttnRes mixes**, with each source's key logit computed on the fly.
- **Attention:** sliding (q/k norm, RoPE from the graph's own tables, ring cache), full (ReLU² over all positions), top-1024 key
  selection, sparse.
- **Router, shared expert, engram** (gate, RMS norm, dilated causal conv).

About 440 dispatches in one compute pass; every bind group is built once.

Prefill stays on ORT (64-token chunks are compute-friendly there). The cache moves between the two on the GPU (`importOrt` /
`exportOrt`).

Checks:
- per layer vs ORT on the same state: every layer's MLP input within 4e-5, the same experts chosen in all 31 MoE layers;
- 16/16 vs PyTorch;
- at 1,400 tokens of context plus a handoff to ORT and back: 5.3e-5.

Lessons: WebGPU validation errors are silent (every dispatch becomes a no-op and the output is zeros). All three bugs came from
there: 11 storage bindings (the limit is 8), a binding offset that wasn't 256-byte aligned, and one buffer bound writable twice.
`WgslDecoder.diagnostics()` reports compile and validation errors.

## 5. Coalesced GEMVs, GPU sampling: 17.4 to 12.1 ms

Measured first: encoding a step took only 0.2 ms, so "encode the next step ahead" was not worth doing. The costs were GPU time
(10.5 ms), a 256 KB logits readback (2.7 ms) and the engram fetch (up to 5-7 ms for unseen n-grams).

- **GEMV rewrite:** one thread per u32 word (8 weights, inputs in registers); the workgroup spans a row and does several rows;
  `subgroupAdd` reductions. All 4-bit matmuls went from ~9.9 to ~4.3 ms.
- **Router:** 96 workgroups for the logits, then a 32-lane subgroup top-8. A first one-workgroup version cost 7-8 ms.
- **GPU sampling:** greedy, or top-20 / temperature / top-p as in `sample()`. The token stays on the GPU as the next step's input,
  and 4 bytes come back instead of 256 KB.
- **Engram:** page-contiguous packing, 4 Range requests a token instead of 12 (later replaced, see 6).

## 6. No CPU in the decode loop: 12.1 to 7.2 ms (chat 58 to 136 tok/s)

A GPU-sampled step took 11.1 ms of wall time but only ~6.7 ms of GPU time back to back, after ~2.2 ms waiting on the engram fetch.
A bare submit + `mapAsync` round trip is only 0.2 ms. The real cost was the idle GPU: after a 2 ms pause a step takes 12-15 ms
instead of ~9 ms, because the GPU clocks down while it waits.

- **The engram table on the GPU** (1.7 GB: two buffers, one per n-gram order). A kernel hashes the last three tokens (64-bit
  multiply / xor / shift on u32 pairs) from a 4-slot token ring, then gathers and dequantizes the 16 rows. Bit-exact against the JS
  path. The pages are interleaved from `engram_table.safetensors` while loading, so no extra file is needed.
- **The next step is queued before the current token is read back.** Its input is the sampled token, still on the GPU. If the
  caller feeds a different token, the queued step is simply overwritten: it only wrote position `pos+1`, and the ring buffers (128
  for sliding attention, 12 for the engram conv) are larger than what is ever read back.
- **Prefill reads its engram rows from the GPU table too:** the same gather kernel, one small submit a token, then a readback. Served
  from the Hugging Face Hub, the page-by-page HTTP lookups had made a 24-token prefill take 37 s (every Range request pays a redirect
  and a CDN round trip); locally it went from 362 to 278 ms.
- **Two-stage sampler:** 64 workgroups find the top-20 of 1,024 logits each, then one workgroup picks from 1,280 candidates.
  Temperature sampling had cost 0.5 ms more than greedy; now it costs the same.
- **Flash-decoding for the full and sparse attention output:** split over 64-position chunks and merged with log-sum-exp. Before,
  16 workgroups walked every position serially: ~16 ms of a step at 2,131 tokens of context. Decode at 2.2K context went from
  21.9 to 9.4 ms.
- **q / k / v as one GEMV:** correct, but no measurable gain. Dispatch count was not the limit.

## 7. Long-context kernels and tuning: 7.2 to 6.6 ms (9.4 to 7.4 ms at 2.2K)

- **Scores, one thread per key for all 8 heads of its group.** Each key row is read once (as vec4) instead of 8 times, and the
  full layers' score and key-selection kernels became one.
- **Top-1024 key selection by 8-bit radix select:** 4 passes with a workgroup-atomic histogram and keys in registers, ties resolved
  with a prefix sum. It replaced 32 one-bit passes and a serial tie scan; the result is identical.
- **Sliding attention:** vec4 key dots, and the value sum split four ways.
- **Rows per workgroup:** a sweep within one page load (`TUNE`, `rerecord()`) found 4 rows per workgroup best for every GEMV
  variant (6.45 vs 6.7 ms).

## 8. Ablation-driven: 6.6 to 5.4 ms (chat 147 to 180 tok/s)

Per-dispatch timestamps were misleading: each dispatch ran in its own pass, at varying clocks. Instead, a step was timed with
whole classes of kernels left out (`bench.html?ablate=1`; the output is wrong but the timing is honest). At 6.4 ms a step:

| Kernel class | Cost |
|---|---|
| Routed experts | 1.8 ms |
| **AttnRes mixes** | **1.4 ms** |
| Attention q/k/v/o matmuls | 1.0 ms |
| Other attention kernels | 0.8 ms |
| Shared expert | 0.5 ms |
| Router | 0.4 ms |
| LM head | 0.3 ms |
| Everything except embedding and sampling | 0.2 ms floor |

- **Mixes:** each did up to 20 sums, each an 8-step barrier tree in one workgroup. `subgroupAdd` plus one barrier replaced them:
  6.7 to 5.8 ms.
- **Packed scales:** every 4-bit GEMV read an fp32 scale and an fp32 bias (or offset) per 64 weights, 20% of all bytes. They are
  now the original bf16 pair in one u32, which is exact. The routed experts' packed copies are made on the GPU from ORT's fp32
  buffers (`b = c - 8s` is exact). 5.8 to 5.5 ms.

Tried, no gain (not kept):
- shared and routed experts fused into 2 dispatches (62 fewer dispatches, same time);
- `vec4<u32>` weight loads;
- issuing all weight loads before the math;
- more rows per workgroup;
- a deliberately wrong, cheaper nibble unpack, at the same speed: the GEMVs are not ALU-bound.

## Where it stands

- **Bandwidth:** the large GEMVs run near peak (LM head ~250 GB/s). The per-layer ones (2-5 MB) reach about 60%: short,
  latency-bound dispatches in a fully serial chain.
- **Next lever:** reading the weights once for several tokens. A batched WGSL step would serve both prefill (still ORT, ~100
  tok/s) and speculative decoding, for example n-gram drafts from the context verified in one step.
- **Context:** the cache is capped at 4,096 tokens.
- **Tested only on Apple silicon in Chromium.** The decoder needs WebGPU `subgroups` and ~4 GB storage buffers; without subgroups
  the engine decodes on the ORT graph (~66 ms a token).
