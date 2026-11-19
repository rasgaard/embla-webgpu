// Streams the routed experts (3.9 GB of model.safetensors) straight into GPU buffers, in the layout the ONNX graph takes as inputs
// (tools/patch_moe_dense.py), so they never touch the WASM heap. The WGSL decoder reads the same buffers.
//   e<i>_fcN_{q4,sc,c}  : fc with the experts stacked into rows [96*512, 2048]          (a plain reshape of MLX's bytes)
//   e<i>_projT_{q4,sc,c}: proj reordered to [2048 out, 96*512 in]                        (transposed from MLX's [expert, out, in])
// q4: packed nibbles, sc: fp32 scales, c: fp32 offsets c = b + 8s (MLX's w = s*q + b rewritten as s*(q-8) + c for MatMulNBits).
import * as ort from "./ort.mjs";

/** Header of a remote safetensors file: {url, hdr, base} (base = byte offset of the data). */
export async function openSafetensors(url) {
  const get = async (a, b) => new Uint8Array(await (await fetch(url, { headers: { Range: `bytes=${a}-${b}` } })).arrayBuffer());
  const n = Number(new DataView((await get(0, 7)).buffer).getBigUint64(0, true));
  return { url, hdr: JSON.parse(new TextDecoder().decode(await get(8, 8 + n - 1))), base: 8 + n };
}
export const tensorBytes = (st, key) => st.hdr[key].data_offsets[1] - st.hdr[key].data_offsets[0];

export const EXPERT_INPUT = /^e(\d+)_(fcN|projT)_(q4|sc|c)$/;
/** The graph's expert input names, from the safetensors header (so experts stream while the graph compiles). */
export function expertInputNames(st) {
  const out = [];
  for (const k of Object.keys(st.hdr)) { const m = /^transformer\.h\.(\d+)\.mlp\.experts_(fc|proj)\.weight$/.exec(k);
    if (m) for (const p of ["q4", "sc", "c"]) out.push(`e${m[1]}_${m[2] === "fc" ? "fcN" : "projT"}_${p}`); }
  return out;
}

/** [E, O, R] row blocks -> [O, E, R]  (MLX expert-major -> output-major, for the proj MatMulNBits). */
function swapExpertOut(src, E, O, R) {
  const dst = new src.constructor(src.length);
  if (R >= 64) { for (let e = 0; e < E; e++) for (let o = 0; o < O; o++) dst.set(src.subarray((e * O + o) * R, (e * O + o + 1) * R), (o * E + e) * R); }
  else for (let e = 0; e < E; e++) for (let o = 0; o < O; o++) { const a = (e * O + o) * R, b = (o * E + e) * R; for (let r = 0; r < R; r++) dst[b + r] = src[a + r]; }
  return dst;
}

/** Upload every expert input in `inputNames` to its own GPU buffer -> {name: ort.Tensor (gpu-buffer)}.
 *  fetchTensor(key) -> {t, u8}: a safetensors tensor (the caller serves it from its cache or the network). */
export async function loadExperts(st, device, inputNames, onProgress, fetchTensor) {
  const feeds = {}, want = inputNames.filter(n => EXPERT_INPUT.test(n));
  let done = 0, bytes = 0;
  const bf = (u8) => { const u16 = new Uint16Array(u8.buffer, u8.byteOffset, u8.byteLength / 2), f = new Uint32Array(u16.length);
    for (let k = 0; k < u16.length; k++) f[k] = u16[k] << 16; return new Float32Array(f.buffer); };                    // bf16 -> fp32
  const gpu = (data, dtype, dims) => { const u8 = new Uint8Array(data.buffer, data.byteOffset, data.byteLength);
    const buf = device.createBuffer({ size: Math.ceil(u8.byteLength / 4) * 4, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST });
    device.queue.writeBuffer(buf, 0, u8); bytes += u8.byteLength; return ort.Tensor.fromGpuBuffer(buf, { dataType: dtype, dims }); };
  const one = async (nm) => {
    const [, i, w, part] = EXPERT_INPUT.exec(nm), base = `transformer.h.${i}.mlp.experts_${w === "fcN" ? "fc" : "proj"}`;
    const swap = w === "projT", G2 = 32;                                                // MatMulNBits B: [N, K/64, 32 bytes]
    const T = (x, t, rowElems) => swap ? swapExpertOut(x, t.shape[0], t.shape[1], rowElems) : x;
    const dims = (t, last) => { const [E, O] = t.shape;
      if (!swap) return part === "q4" ? [E * O, last / G2, G2] : [E * O, last];
      return part === "q4" ? [O, E * last / G2, G2] : [O, E * last]; };
    if (part === "q4") { const { t, u8 } = await fetchTensor(base + ".weight"); const row = t.shape.at(-1) * 4;   // MLX u32 words = packed bytes, low nibble first
      feeds[nm] = gpu(T(u8, t, row), "uint8", dims(t, row)); }
    else if (part === "sc") { const { t, u8 } = await fetchTensor(base + ".scales"); const row = t.shape.at(-1);
      feeds[nm] = gpu(T(bf(u8), t, row), "float32", dims(t, row)); }
    else { const [s, b] = await Promise.all([fetchTensor(base + ".scales"), fetchTensor(base + ".biases")]); const S = bf(s.u8), B = bf(b.u8);
      for (let k = 0; k < B.length; k++) B[k] += 8 * S[k];                                                               // c = b + 8s
      const row = s.t.shape.at(-1); feeds[nm] = gpu(T(B, s.t, row), "float32", dims(s.t, row)); }
    onProgress?.(++done, want.length, bytes);
  };
  const queue = [...want]; await Promise.all(Array.from({ length: 4 }, async () => { while (queue.length) await one(queue.shift()); }));
  await device.queue.onSubmittedWorkDone();
  return feeds;
}
