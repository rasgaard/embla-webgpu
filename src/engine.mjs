// Embla 1 (4-bit) in the browser on WebGPU.
//   prefill: onnxruntime-web runs the exported one-step graph (tools/build_graph.sh) in chunks of up to 64 tokens;
//   decode:  the hand-written WGSL decoder (wgsl_decode.mjs), one token a step, sampled on the GPU, the next step queued before the
//            current token is read back. Without the WebGPU "subgroups" feature decode runs on the ORT graph too (slower).
// Stateful: the KV / engram-conv cache stays on the GPU between calls, so a chat only prefills its new tokens.
import * as ort from "./ort.mjs";
import { pageRows } from "./engram_hash.mjs";
import { EngramTable, PAGE_BYTES } from "./engram_table.mjs";
import { openSafetensors, loadExperts, tensorBytes, expertInputNames, EXPERT_INPUT } from "./experts.mjs";
import { WeightCache, remoteSignature, fetchBytes, contentTag } from "./weight_cache.mjs";
import { WgslDecoder } from "./wgsl_decode.mjs";

const ENGRAM_CFG = { orders: [2, 3], heads: 8, pageHeads: 4, pages: 209716, S: 56, pad: 65536 };
export const MAX_CONTEXT = 8192;                      // the graph's RoPE / key-index tables cover 8,192 + 64 positions

// A 60-byte ONNX model (y = -x). Creating a session for it starts ORT's WebGPU backend and hands us ORT's GPUDevice before the
// real graph is compiled, so the experts can stream to the GPU meanwhile. (Giving ORT a device of our own instead breaks its
// GPU readbacks: "Failed to wait for the operation".) The session is kept alive so ORT never releases that device.
const BOOT_ONNX = "CAg6MgoLCgF4EgF5IgNOZWcSAWJaDwoBeBIKCggIARIECgIIAWIPCgF5EgoKCAgBEgQKAggBQgQKABAS";
async function ortDevice() {
  const boot = await ort.InferenceSession.create(Uint8Array.from(atob(BOOT_ONNX), c => c.charCodeAt(0)), { executionProviders: ["webgpu"] });
  return { boot, device: await ort.env.webgpu.device };
}
const onGpu = t => t?.location === "gpu-buffer";

/** fp32 initializers of an ONNX model by name (inline raw_data, or external data in `data`). A minimal protobuf walk:
 *  ModelProto.graph (7) -> GraphProto.initializer (5) -> TensorProto name (8) / raw_data (9) / external_data (13). */
function onnxFloats(model, data, names) {
  const want = new Set(names), out = {}, td = new TextDecoder();
  const varint = (u, p) => { let x = 0, m = 1, b; do { b = u[p++]; x += (b & 127) * m; m *= 128; } while (b & 128); return [x, p]; };
  function* fields(u, p, end) {
    while (p < end) {
      let tag, v; [tag, p] = varint(u, p); const f = Math.floor(tag / 8), w = tag & 7;
      if (w === 0) { [v, p] = varint(u, p); yield [f, v]; }
      else if (w === 2) { [v, p] = varint(u, p); yield [f, p, p + v]; p += v; }
      else if (w === 1) p += 8; else if (w === 5) p += 4; else throw new Error(`onnx: wire type ${w}`);
    }
  }
  const str = (a, b) => td.decode(model.subarray(a, b));
  for (const [f, a, b] of fields(model, 0, model.length)) if (f === 7)
    for (const [g, c, d] of fields(model, a, b)) if (g === 5) {
      let name, raw; const ext = {};
      for (const [h, x, y] of fields(model, c, d)) {
        if (h === 8) name = str(x, y); else if (h === 9) raw = [x, y];
        else if (h === 13) { let k, v; for (const [i, s0, s1] of fields(model, x, y)) { if (i === 1) k = str(s0, s1); else if (i === 2) v = str(s0, s1); } ext[k] = v; }
      }
      if (!want.has(name)) continue;
      const o = +(ext.offset ?? 0), bytes = raw ? model.subarray(raw[0], raw[1]) : data.subarray(o, o + +ext.length);
      out[name] = new Float32Array(bytes.slice().buffer);
    }
  for (const n of names) if (!out[n]) throw new Error(`onnx: initializer ${n} not found`);
  return out;
}

export class Embla {
  /**
   * Load the model: the ONNX graph (0.48 GB) and the routed experts (3.9 GB) concurrently, then the WGSL decoder's weights and the
   * engram table (1.7 GB) onto the GPU. Everything downloaded is kept in a Cache API store, so later loads read from disk.
   * weightsUrl: folder with model.safetensors, engram_table.safetensors, config.json. graphUrl: folder with embla_step.onnx(.data).
   * onProgress({cachedBytes, netBytes, total, seconds}).
   */
  static async load({ weightsUrl, graphUrl, useCache = true, warmup = true, decode = "wgsl", onStatus = () => {}, onProgress = () => {} } = {}) {
    if (!navigator.gpu) throw new Error("WebGPU is not available in this browser.");
    const t0 = performance.now(), secs = () => (performance.now() - t0) / 1000;
    const modelUrl = graphUrl + "embla_step.onnx", dataUrl = modelUrl + ".data";
    const stUrl = weightsUrl + "model.safetensors", etUrl = weightsUrl + "engram_table.safetensors";

    onStatus("Checking cache…");
    const [{ boot, device }, sig, st, tab, cfg] = await Promise.all([ortDevice(),
      Promise.all([modelUrl, dataUrl, stUrl, etUrl].map(remoteSignature)), openSafetensors(stUrl), EngramTable.open(etUrl),
      fetch(weightsUrl + "config.json").then(r => r.json())]);
    const cache = useCache ? await WeightCache.open() : new WeightCache(null, "off");
    const [tagOnnx, tagData, tagWeights, tagEngram] = sig.map(contentTag), used = new Set();
    const wgsl = decode === "wgsl" && device.features.has("subgroups");
    const keys = Object.keys(st.hdr).filter(k => k !== "__metadata__"), isExpertKey = k => k.includes(".mlp.experts_");
    const acc = { cachedBytes: 0, netBytes: 0, total: Number(sig[0].split("|")[1]) + Number(sig[1].split("|")[1])
      + keys.filter(isExpertKey).reduce((n, k) => n + tensorBytes(st, k), 0)
      + (wgsl ? keys.filter(k => !isExpertKey(k)).reduce((n, k) => n + tensorBytes(st, k), 0) + 2 * ENGRAM_CFG.pages * PAGE_BYTES : 0) };
    const report = () => onProgress({ ...acc, seconds: secs() });
    let failed = false;                                                  // one half failed: stop the other from uploading more

    // every download goes through here: cache hit -> disk, miss -> network + store
    const getOrFetch = async (key, fetcher) => {
      if (failed) throw new Error("load aborted");
      used.add(key);
      const hit = await cache.get(key);
      if (hit) { acc.cachedBytes += hit.byteLength; report(); return hit; }
      const u8 = await fetcher(); await cache.put(key, u8); return u8;
    };
    let graphMsg = "loading graph", expertMsg = "";
    const status = () => onStatus([graphMsg, expertMsg].filter(Boolean).join(" · ").replace(/^./, c => c.toUpperCase()) + "…");
    status();

    let graphSeconds = 0;
    const graphJob = (async () => {
      const modelBytes = await getOrFetch("graph/onnx@" + tagOnnx, () => fetchBytes(modelUrl, n => { acc.netBytes += n; report(); }));
      const dataBytes = await getOrFetch("graph/data@" + tagData, () => fetchBytes(dataUrl, n => { acc.netBytes += n; report(); }));
      const r = onnxFloats(modelBytes, dataBytes, ["rope_cos_cache", "rope_neg_sin_cache"]);   // the WGSL decoder uses the graph's RoPE tables
      const rope = new Float32Array(r.rope_cos_cache.length * 2); rope.set(r.rope_cos_cache); rope.set(r.rope_neg_sin_cache, r.rope_cos_cache.length);
      graphMsg = "compiling graph"; status();
      const s = await ort.InferenceSession.create(modelBytes, {
        executionProviders: ["webgpu"], graphOptimizationLevel: "disabled", enableMemPattern: false,   // folding would expand the dequant subgraphs to GBs
        preferredOutputLocation: "gpu-buffer",                           // KV cache stays on the GPU between steps; logits are read explicitly
        externalData: [{ path: "embla_step.onnx.data", data: dataBytes }],
      });
      graphSeconds = secs(); graphMsg = "graph ready"; status(); return { s, rope };
    })();

    const memo = new Map();                                              // expert scales are read by two inputs (sc and c): fetch once
    const fetchTensor = (key) => {
      const go = async () => {
        const t = st.hdr[key], [a, b] = t.data_offsets, o = st.base + a;
        const u8 = await getOrFetch(`weights/${key}@${tagWeights}`, async () => {
          const r = await fetch(st.url, { headers: { Range: `bytes=${o}-${o + (b - a) - 1}` } });
          if (!r.ok) throw new Error(`model.safetensors: HTTP ${r.status}`);
          const u = new Uint8Array(await r.arrayBuffer()); acc.netBytes += u.byteLength; report(); return u;
        });
        return { t, u8 };
      };
      if (!key.endsWith(".weight") && isExpertKey(key)) { if (!memo.has(key)) memo.set(key, go()); return memo.get(key); }
      return go();
    };
    const expertJob = loadExperts(st, device, expertInputNames(st), (done, total) => { expertMsg = `uploading experts to GPU ${done}/${total}`; status(); }, fetchTensor);
    graphJob.catch(() => { failed = true; }); expertJob.catch(() => { failed = true; });
    const [{ s, rope }, experts] = await Promise.all([graphJob, expertJob]);
    memo.clear();
    if (await ort.env.webgpu.device !== device) throw new Error("ORT switched GPU devices between sessions; expert buffers would be on the wrong device.");
    const missing = s.inputNames.filter(n => EXPERT_INPUT.test(n) && !experts[n]);
    if (missing.length) throw new Error(`Expert inputs missing from model.safetensors: ${missing.slice(0, 4).join(", ")}…`);

    const stats = { graphSeconds, decode: wgsl ? "wgsl" : "ort" };
    const e = new Embla(s, experts, tab, stats); e.boot = boot;
    if (wgsl) {
      const t2 = performance.now(), P = ENGRAM_CFG.pages;
      e.wgsl = await WgslDecoder.create({ device, fetchTensor, experts, rope, cfg, onStatus,
        engram: { pages: P, chunk: (k, p0, n) => getOrFetch(`engram/${k}/${p0}/${n}@${tagEngram}`, async () => {
          const u = await tab.pages(k * P + p0, n); acc.netBytes += u.byteLength; report(); return u; }) } });
      stats.wgslSeconds = (performance.now() - t2) / 1000;
    }
    if (warmup) {                                                        // compile the shaders a chat uses (first chunk, a chunk with history, decode)
      onStatus("Warming up…"); const w0 = performance.now();
      await e.feed(new Array(11).fill(0)); await e.feed([0, 0, 0]); await e.feed([0]); e.reset();
      stats.warmupSeconds = (performance.now() - w0) / 1000;
    }
    Object.assign(stats, { totalSeconds: secs(), cachedBytes: acc.cachedBytes, netBytes: acc.netBytes, fromCache: acc.netBytes === 0 && acc.cachedBytes > 0,
      cacheWriteError: cache.writeError ? String(cache.writeError.name || cache.writeError) : null, cacheAvailable: cache.available });
    if (!cache.writeError) cache.prune(used).catch(() => {});          // entries of older graphs / weights; not awaited
    return e;
  }

  constructor(session, experts, table, loadStats) {
    this.s = session; this.ex = experts; this.tab = table; this.loadStats = loadStats;
    this.cacheNames = session.inputNames.filter(n => n.endsWith("_in"));
    this.epoch = 0; this.running = null; this.kv = {}; this.chunk = 64;   // a chunk costs about the same for 1..64 tokens (export max T = 64)
    this.reset();
  }

  /** Forget the conversation. Safe while a step is in flight: that step's result is discarded and its inputs are freed after it ends. */
  reset() {
    const old = Object.values(this.kv).filter(onGpu);
    const free = () => old.forEach(t => t.dispose());
    if (this.running) this.running.finally(free); else free();
    this.epoch++; this.kv = {}; this.owner = "ort";
    for (const n of this.cacheNames)
      this.kv[n] = n === "conv_in" ? new ort.Tensor("float32", new Float32Array(9 * 2048), [1, 9, 2048])
        : new ort.Tensor("float32", new Float32Array(0), [1, 0, n.startsWith("sw") ? 4 : 2, 128]);
    this.all = []; this.pos = 0; this.spec = null;
  }
  get length() { return this.pos; }
  get usesWgsl() { return !!this.wgsl && this.useWgsl !== false; }

  /** Run one chunk of tokens on ORT. The new KV cache stays on the GPU; the logits (Float32Array [vocab]) are only read back when asked. */
  async #step(toks, engram, wantLogits) {
    if (toks.length === 1 && this.usesWgsl) return this.#stepWgsl(toks[0], wantLogits);
    this.#toOrt();
    const T = toks.length, epoch = this.epoch;
    const feeds = { ids: new ort.Tensor("int64", BigInt64Array.from(toks.map(BigInt)), [1, T]),
      engram_rows: new ort.Tensor("float32", await engram, [1, T, 16, 128]), ...this.kv, ...this.ex };
    const job = this.s.run(feeds); this.running = job.catch(() => {});
    let r; try { r = await job; } finally { this.running = null; }
    if (epoch !== this.epoch) { for (const t of Object.values(r)) if (onGpu(t)) t.dispose(); throw new DOMException("aborted", "AbortError"); }
    this.pos += T;
    for (const n of this.cacheNames) { const old = this.kv[n]; this.kv[n] = r[n.replace(/_in$/, "_out")]; if (onGpu(old)) old.dispose(); }
    const out = wantLogits ? await r.logits.getData() : null;
    if (onGpu(r.logits)) r.logits.dispose();
    return out;
  }

  /** One token on the WGSL decoder (with logits read back). It owns the cache from its first step until ORT needs it back. */
  async #stepWgsl(tok, wantLogits) {
    const epoch = this.epoch;
    this.spec = null;
    if (this.owner !== "wgsl") { this.wgsl.importOrt(this.kv, this.pos, this.all); this.owner = "wgsl"; }
    const job = this.wgsl.step(tok, this.pos, { readLogits: wantLogits });
    this.running = job.catch(() => {});
    let out; try { out = await job; } finally { this.running = null; }
    if (epoch !== this.epoch) throw new DOMException("aborted", "AbortError");
    this.pos += 1;
    return out.logits ?? null;
  }

  /** Feed one token and return the next one, sampled as sample() does. On the WGSL decoder the sampling runs on the GPU, only the
   *  token (4 bytes) is read back, and the step for the returned token is already queued (its input is still in the GPU's tok[0]).
   *  If the caller then feeds that token, the queued step is used; otherwise it is simply overwritten (it only wrote position
   *  pos + 1, which the next real step at that position rewrites). */
  async decodeSample(token, opts = {}, { signal } = {}) {
    if (!this.usesWgsl) return sample(await this.feed([token], { signal }), opts);
    if (this.pos + 1 > MAX_CONTEXT) throw new Error(`Context limit (${MAX_CONTEXT} tokens) reached.`);
    if (signal?.aborted) throw new DOMException("aborted", "AbortError");
    const W = this.wgsl, epoch = this.epoch;
    const sp = { temperature: opts.temperature ?? 0.6, topP: opts.topP ?? 0.95 }, key = sp.temperature + "/" + sp.topP;
    if (this.owner !== "wgsl") { W.importOrt(this.kv, this.pos, this.all); this.owner = "wgsl"; this.spec = null; }
    const spec = this.spec; this.spec = null;
    const job = spec && spec.epoch === epoch && spec.pos === this.pos && spec.key === key && spec.token === token ? spec.job
      : W.step(token, this.pos, { readLogits: false, sample: sp });
    let next = null;
    if (this.pos + 2 <= MAX_CONTEXT) {                                   // keep the GPU busy while this token comes back
      next = { job: W.step(null, this.pos + 1, { readLogits: false, sample: sp }), pos: this.pos + 1, epoch, key };
      next.job.catch(() => {});
    }
    this.running = job.catch(() => {});
    let out; try { out = await job; } finally { this.running = null; }
    if (epoch !== this.epoch) throw new DOMException("aborted", "AbortError");
    this.all.push(token); this.pos += 1;
    if (next) { next.token = out.token; this.spec = next; }
    return out.token;
  }

  /** Hand the cache back to ORT (prefill) as tensors re-linearised from the WGSL state. */
  #toOrt() {
    if (this.owner !== "wgsl") return;
    const kv = this.wgsl.exportOrt(this.pos);
    for (const n of this.cacheNames) { if (onGpu(this.kv[n])) this.kv[n].dispose(); this.kv[n] = kv[n]; }
    this.owner = "ort";
  }

  /** Feed tokens (prefill / decode); returns the logits after the last one. The engram rows come from the GPU-resident table when
   *  the WGSL decoder is loaded; otherwise every chunk's pages are requested over HTTP up front, overlapping earlier chunks' GPU work. */
  async feed(tokens, { chunk = this.chunk, signal } = {}) {
    if (this.pos + tokens.length > MAX_CONTEXT) throw new Error(`Context limit (${MAX_CONTEXT} tokens) reached.`);
    const parts = [];
    if (tokens.length === 1 && this.usesWgsl) parts.push([tokens, null]);   // the WGSL decoder gathers its own rows
    else if (this.wgsl) {
      const all = this.wgsl.engramRows(tokens, this.pos, this.all); all.catch(() => {});
      for (let p = 0; p < tokens.length; p += chunk) parts.push([tokens.slice(p, p + chunk), all.then(r => r.slice(p * 2048, (p + chunk) * 2048))]);
    } else {
      const ctx = this.all.slice(-2);                                    // a token's rows depend on the 2 tokens before it
      const rows = pageRows([...ctx, ...tokens], ENGRAM_CFG).slice(ctx.length);
      for (let p = 0; p < tokens.length; p += chunk) parts.push([tokens.slice(p, p + chunk), this.tab.gather(rows.slice(p, p + chunk))]);
    }
    for (const [, g] of parts) g?.catch(() => {});                       // awaited in #step; this only silences an abandoned one
    let logits;
    for (let i = 0; i < parts.length; i++) {
      if (signal?.aborted) throw new DOMException("aborted", "AbortError");
      this.all.push(...parts[i][0]);                                     // only after the abort check: history always matches the cache
      logits = await this.#step(parts[i][0], parts[i][1], i === parts.length - 1);
    }
    return logits;
  }
}

/** Temperature / top-k / top-p sampling from raw logits (the GPU sampler does the same). temperature <= 0 -> greedy. */
export function sample(logits, { temperature = 0.6, topK = 20, topP = 0.95 } = {}) {
  const n = logits.length;
  if (temperature <= 0) { let b = 0; for (let i = 1; i < n; i++) if (logits[i] > logits[b]) b = i; return b; }
  const k = Math.min(topK || n, n);
  // single-pass top-k (k is small): keep a sorted-descending list, insert only values above the current k-th
  const idx = []; let floor = -Infinity;
  for (let i = 0; i < n; i++) {
    const v = logits[i]; if (idx.length === k && v <= floor) continue;
    let j = idx.length; while (j > 0 && logits[idx[j - 1]] < v) j--;
    idx.splice(j, 0, i); if (idx.length > k) idx.pop();
    if (idx.length === k) floor = logits[idx[k - 1]];
  }
  const mx = logits[idx[0]]; let probs = idx.map(i => Math.exp((logits[i] - mx) / temperature));
  const z = probs.reduce((a, b) => a + b, 0); probs = probs.map(p => p / z);
  let cum = 0, keep = 0; for (; keep < probs.length; keep++) { cum += probs[keep]; if (cum >= topP) { keep++; break; } }
  keep = Math.max(1, Math.min(keep, probs.length));
  const zz = probs.slice(0, keep).reduce((a, b) => a + b, 0); let r = Math.random() * zz;
  for (let i = 0; i < keep; i++) { r -= probs[i]; if (r <= 0) return idx[i]; }
  return idx[keep - 1];
}
