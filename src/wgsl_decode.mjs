// Embla 1 decode step in WGSL: one token per step, every layer as a handful of hand-written kernels in one compute pass
// (~440 dispatches), all bind groups built once. Modelled on sevren-ai/embla-metal (MIT), kept in fp32 to match the ONNX graph
// (embla-metal rounds to bf16 like the PyTorch reference). Prefill stays on onnxruntime-web; the KV / conv state moves between
// the two with importOrt() / exportOrt().
//
// Weights: the MLX 4-bit tensors of model.safetensors (w = s*q + b per group of 64; q packed 8 per u32, low nibble first), with
// the bf16 scale and bias of a group packed into one u32. The routed experts reuse the ORT graph's GPU buffers (fcN / projT, see
// experts.mjs). The engram table (1.7 GB) is held on the GPU too, so a decode step needs nothing from the CPU but the position.
// Needs the WebGPU "subgroups" feature (engine.mjs falls back to ORT decode without it). optimizations.md explains the kernels.
import * as ort from "./ort.mjs";

const D = 2048, HD = 128, VOCAB = 65536, E = 96, F = 512, TOPK = 8, SH = 1024;
const EPS32 = 1.1920928955078125e-7;
const WINDOW = 128, RECENT = 128, SPARSE_K = 1024, CONV_TAPS = 4, CONV_DIL = 3, HRING = 12;

const f = (x) => { const s = String(x); return s.includes(".") || s.includes("e") ? s : s + ".0"; };   // WGSL float literal
const quad = (a, i) => `vec4<f32>(${a}[${i}], ${a}[${i} + 1u], ${a}[${i} + 2u], ${a}[${i} + 3u])`;
const NIB = /* wgsl */`
fn nib(w : u32) -> array<vec4<f32>, 2> {
  return array<vec4<f32>, 2>(vec4<f32>(f32(w & 15u), f32((w >> 4u) & 15u), f32((w >> 8u) & 15u), f32((w >> 12u) & 15u)),
                             vec4<f32>(f32((w >> 16u) & 15u), f32((w >> 20u) & 15u), f32((w >> 24u) & 15u), f32(w >> 28u)));
}`;
const INFO = /* wgsl */`struct Info { pos : u32, token : u32, n : u32, pad : u32 };`;
const RED = (n = 256) => /* wgsl */`
var<workgroup> red : array<f32, ${n}>;
fn wsum(t : u32, v : f32) -> f32 {
  red[t] = v; workgroupBarrier();
  for (var s = ${n / 2}u; s > 0u; s = s >> 1u) { if (t < s) { red[t] = red[t] + red[t + s]; } workgroupBarrier(); }
  let r = red[0]; workgroupBarrier(); return r;
}
fn wmax(t : u32, v : f32) -> f32 {
  red[t] = v; workgroupBarrier();
  for (var s = ${n / 2}u; s > 0u; s = s >> 1u) { if (t < s) { red[t] = max(red[t], red[t + s]); } workgroupBarrier(); }
  let r = red[0]; workgroupBarrier(); return r;
}`;

// ------------------------------------------------------------------------------------------------ kernels
/** 4-bit GEMV: one thread per u32 word of a row (8 inputs in registers), the workgroup spans a whole row (K/8 threads) and does RPW
 *  rows, so weight reads are fully coalesced and x is read once. Row sums by subgroupAdd, then across subgroups in workgroup memory.
 *  Scale and bias of a 64-group: one u32 (bf16 | bf16). Variants:
 *    dense  : W [R, K] MLX affine (w = s*q + b), modes store | add | relu2 | softcap
 *    moefc  : rows k*512+f of the 8 chosen experts (fcN), out a[row] = relu(.)^2 * gate_k
 *    moeproj: K = 8 experts x 512 gathered from projT rows, y[o] += . (2 words a thread) */
/** Rows a workgroup of gemv2Src (R must be a multiple); TUNE overrides per variant (rpw_dense / rpw_moefc / rpw_moeproj), for sweeps. */
export const TUNE = {};
const rpwFor = (R, variant = "dense") => TUNE["rpw_" + variant] ?? 4;          // swept on M4 Pro: 4 beats 8 / 16 (6.45 vs 6.7 ms a step)
const gemv2Src = (K, R, mode, variant = "dense") => {
  const WPT = variant === "moeproj" ? 2 : 1;                              // words a thread
  const T = K / 8 / WPT, RPW = rpwFor(R, variant);
  const epi = { store: "y[row] = v;", add: "y[row] = y[row] + v;", relu2: "let r = max(v, 0.0); y[row] = r * r;", softcap: "y[row] = tanh(v / 15.0) * 15.0;" }[mode];
  const head = variant === "dense" ? `
@group(0) @binding(0) var<storage, read> x : array<f32>;
@group(0) @binding(1) var<storage, read> wq : array<u32>;
@group(0) @binding(2) var<storage, read> wsb : array<u32>;
@group(0) @binding(3) var<storage, read_write> y : array<f32>;` : variant === "moefc" ? `
@group(0) @binding(0) var<storage, read> x : array<f32>;
@group(0) @binding(1) var<storage, read> route : array<f32>;
@group(0) @binding(2) var<storage, read> wq : array<u32>;
@group(0) @binding(3) var<storage, read> wsb : array<u32>;
@group(0) @binding(4) var<storage, read_write> y : array<f32>;` : `
@group(0) @binding(0) var<storage, read> route : array<f32>;
@group(0) @binding(1) var<storage, read> wq : array<u32>;
@group(0) @binding(2) var<storage, read> wsb : array<u32>;
@group(0) @binding(3) var<storage, read> x : array<f32>;
@group(0) @binding(4) var<storage, read_write> y : array<f32>;`;
  // per thread: its inputs (registers), and for row r the weight word index, the group index, and the q offset (0 or 8)
  const body = variant === "dense" ? `
  let xv0 = ${quad("x", "t * 8u")}; let xv1 = ${quad("x", "t * 8u + 4u")}; let sx = dot(xv0, vec4<f32>(1.0)) + dot(xv1, vec4<f32>(1.0));
  var wv : array<u32, ${RPW}>; var uv : array<u32, ${RPW}>;            // all loads first: more requests in flight
  for (var r = 0u; r < ${RPW}u; r++) { let row = min(wg.x * ${RPW}u + r, ${R - 1}u); wv[r] = wq[row * ${K / 8}u + t]; uv[r] = wsb[row * ${K / 64}u + t / 8u]; }
  for (var r = 0u; r < ${RPW}u; r++) {
    let q = nib(wv[r]); let u = uv[r];
    acc[r] = SC(u) * (dot(q[0], xv0) + dot(q[1], xv1)) + BI(u) * sx;
  }` : variant === "moefc" ? `
  let xv0 = ${quad("x", "t * 8u")}; let xv1 = ${quad("x", "t * 8u + 4u")}; let sx = dot(xv0, vec4<f32>(1.0)) + dot(xv1, vec4<f32>(1.0));
  let row0 = wg.x * ${RPW}u; let src0 = u32(route[row0 / ${F}u]) * ${F}u + row0 % ${F}u;   // RPW | F: one expert a workgroup
  var wv : array<u32, ${RPW}>; var uv : array<u32, ${RPW}>;
  for (var r = 0u; r < ${RPW}u; r++) { wv[r] = wq[(src0 + r) * 256u + t]; uv[r] = wsb[(src0 + r) * 32u + t / 8u]; }
  for (var r = 0u; r < ${RPW}u; r++) {
    let q = nib(wv[r]); let u = uv[r];
    acc[r] = SC(u) * (dot(q[0], xv0) + dot(q[1], xv1)) + BI(u) * sx;
  }` : `
  let k = t / 32u; let ww = (t % 32u) * 2u;                       // chosen expert k, word ww / ww+1 of its 64
  let e = u32(route[k]); let ai = k * ${F}u + ww * 8u;
  let a0 = ${quad("x", "ai")}; let a1 = ${quad("x", "ai + 4u")}; let a2 = ${quad("x", "ai + 8u")}; let a3 = ${quad("x", "ai + 12u")};
  let sx = dot(a0 + a1 + a2 + a3, vec4<f32>(1.0));
  let wbase = e * 64u + ww; let gb = e * 8u + ww / 8u;
  var wv : array<vec2<u32>, ${RPW}>; var uv : array<u32, ${RPW}>;
  for (var r = 0u; r < ${RPW}u; r++) { let row = wg.x * ${RPW}u + r; wv[r] = vec2<u32>(wq[row * 6144u + wbase], wq[row * 6144u + wbase + 1u]); uv[r] = wsb[row * 768u + gb]; }
  for (var r = 0u; r < ${RPW}u; r++) {
    let q0 = nib(wv[r].x); let q1 = nib(wv[r].y); let u = uv[r];
    acc[r] = SC(u) * (dot(q0[0], a0) + dot(q0[1], a1) + dot(q1[0], a2) + dot(q1[1], a3)) + BI(u) * sx;
  }`;
  const out = variant === "moefc" ? "let rr = max(v, 0.0); y[row] = rr * rr * route[" + TOPK + "u + row / " + F + "u];" : variant === "moeproj" ? "y[row] = y[row] + v;" : epi;
  return /* wgsl */`enable subgroups;
${head}
${NIB}
fn SC(u : u32) -> f32 { return bitcast<f32>(u << 16u); }            // packed bf16 pair: scale (low), bias (high)
fn BI(u : u32) -> f32 { return bitcast<f32>(u & 0xffff0000u); }
var<workgroup> part : array<array<f32, ${RPW}>, ${Math.max(1, T / 4)}>;      // [subgroups (size >= 4)][rows]
@compute @workgroup_size(${T})
fn main(@builtin(workgroup_id) wg : vec3<u32>, @builtin(local_invocation_id) lid : vec3<u32>,
        @builtin(subgroup_invocation_id) sl : u32, @builtin(subgroup_size) sn : u32) {
  let t = lid.x; var acc : array<f32, ${RPW}>;
  ${body}
  let sg = t / sn;
  for (var r = 0u; r < ${RPW}u; r++) { let v = subgroupAdd(acc[r]); if (sl == 0u) { part[sg][r] = v; } }
  workgroupBarrier();
  if (t < ${RPW}u) {
    let row = wg.x * ${RPW}u + t;
    if (row < ${R}u) {
      var v = 0.0;
      for (var i = 0u; i < ${T}u / sn; i++) { v += part[i][t]; }
      ${out}
    }
  }
}`;
};

/** Expert scales for the WGSL kernels: ORT's fp32 (scale s, offset c = b + 8s) -> one u32 a group, bf16 s (low) | bf16 b (high).
 *  b = c - 8s is exact (s, b were bf16 to begin with). */
const packScalesSrc = (n) => /* wgsl */`
@group(0) @binding(0) var<storage, read> s : array<f32>;
@group(0) @binding(1) var<storage, read> c : array<f32>;
@group(0) @binding(2) var<storage, read_write> o : array<u32>;
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) g : vec3<u32>) {
  let i = g.y * 65535u * 256u + g.x;
  if (i < ${n}u) { o[i] = (bitcast<u32>(s[i]) >> 16u) | (bitcast<u32>(c[i] - 8.0 * s[i]) & 0xffff0000u); }
}`;

/** router in one workgroup: 96 logits (fp32 router), sigmoid, top 8 of score + bias, renormalised gates -> route (needs subgroups). */
const routerSrc = () => /* wgsl */`enable subgroups;
@group(0) @binding(0) var<storage, read> lg : array<f32>;      // [96] router logits (dotRows: one workgroup a row)
@group(0) @binding(1) var<storage, read> bias : array<f32>;
@group(0) @binding(2) var<storage, read_write> route : array<f32>;
@compute @workgroup_size(32)
fn main(@builtin(local_invocation_id) lid : vec3<u32>, @builtin(subgroup_invocation_id) sl : u32, @builtin(subgroup_size) sn : u32) {
  let t = lid.x;
  // top 8 of sigmoid + bias: every subgroup does the same (uniform control flow), a lane owns keys sl, sl + sn, ...;
  // max by subgroupMax, ties to the lowest index by subgroupMin
  var taken = 0u; var sum = 0.0; var ci : array<u32, ${TOPK}>; var cs : array<f32, ${TOPK}>;
  for (var k = 0u; k < ${TOPK}u; k++) {
    var bv = -3.0e38; var bi = 0xffffffffu; var j = 0u;
    for (var i = sl; i < ${E}u; i += sn) {
      if ((taken & (1u << j)) == 0u) { let key = 1.0 / (1.0 + exp(-lg[i])) + bias[i]; if (key > bv) { bv = key; bi = i; } }
      j++;
    }
    let m = subgroupMax(bv);
    let w = subgroupMin(select(0xffffffffu, bi, bv == m));
    j = 0u;
    for (var i = sl; i < ${E}u; i += sn) { if (i == w) { taken = taken | (1u << j); } j++; }
    let sc = 1.0 / (1.0 + exp(-lg[w])); ci[k] = w; cs[k] = sc; sum += sc;
  }
  if (t == 0u) { for (var k = 0u; k < ${TOPK}u; k++) { route[k] = f32(ci[k]); route[${TOPK}u + k] = cs[k] / max(sum, 1e-9); } }
}`;

/** fp32 GEMV, one workgroup per row (router: 96 x 2048). */
const dotRowsSrc = (K) => /* wgsl */`
@group(0) @binding(0) var<storage, read> x : array<f32>;
@group(0) @binding(1) var<storage, read> w : array<f32>;
@group(0) @binding(2) var<storage, read_write> y : array<f32>;
${RED()}
@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wg : vec3<u32>, @builtin(local_invocation_id) lid : vec3<u32>) {
  let t = lid.x; var s = 0.0;
  for (var d = t; d < ${K}u; d += 256u) { s += w[wg.x * ${K}u + d] * x[d]; }
  let r = wsum(t, s);
  if (t == 0u) { y[wg.x] = r; }
}`;

/** Block AttnRes: the K mixes of N sources (+ the partial sum x), optionally RMS-normed. One workgroup, 8 dims a thread. The NP
 *  sums (and the output norm) reduce with subgroupAdd + one barrier (an 8-step barrier tree each cost ~1 ms a token). */
const mixSrc = (N, K, HASX, NORM, col, eps) => {
  const NP = (N + (HASX ? 1 : 0)) * (K + 1);
  return /* wgsl */`enable subgroups;
@group(0) @binding(0) var<storage, read> src : array<f32>;       // [9][D]
@group(0) @binding(1) var<storage, read> qs : array<f32>;        // [69][D]
@group(0) @binding(2) var<storage, read> x : array<f32>;         // [D]
@group(0) @binding(3) var<storage, read_write> outp : array<f32>;// [3][D]
var<workgroup> tot : array<f32, ${NP}>;
var<workgroup> sgp : array<f32, ${NP * 32}>;
var<workgroup> nrm : array<f32, 32>;
@compute @workgroup_size(256)
fn main(@builtin(local_invocation_id) lid : vec3<u32>, @builtin(subgroup_invocation_id) sl : u32, @builtin(subgroup_size) sn : u32) {
  let t = lid.x; let sg = t / sn; let nsg = 256u / sn;
  var v : array<array<f32, 8>, ${N}>; var xv : array<f32, 8>; var acc : array<f32, ${NP}>;
  for (var p = 0u; p < ${NP}u; p++) { acc[p] = 0.0; }
  ${HASX ? "" : "_ = x[0];                                       // keep the binding in the auto layout"}
  for (var e = 0u; e < 8u; e++) {
    let d = t + e * 256u;
    var q : array<f32, ${K}>;
    for (var j = 0u; j < ${K}u; j++) { q[j] = qs[(${col}u + j) * ${D}u + d]; }
    for (var n = 0u; n < ${N}u; n++) {
      let a = src[n * ${D}u + d]; v[n][e] = a;
      acc[n * ${K + 1}u] += a * a;
      for (var j = 0u; j < ${K}u; j++) { acc[n * ${K + 1}u + 1u + j] += q[j] * a; }
    }
    ${HASX ? `xv[e] = x[d]; acc[${N * (K + 1)}u] += xv[e] * xv[e];
    for (var j = 0u; j < ${K}u; j++) { acc[${N * (K + 1)}u + 1u + j] += q[j] * xv[e]; }` : ""}
  }
  for (var p = 0u; p < ${NP}u; p++) { let r = subgroupAdd(acc[p]); if (sl == 0u) { sgp[p * 32u + sg] = r; } }
  workgroupBarrier();
  if (t < ${NP}u) { var r = 0.0; for (var i = 0u; i < nsg; i++) { r += sgp[t * 32u + i]; } tot[t] = r; }
  workgroupBarrier();
  for (var j = 0u; j < ${K}u; j++) {
    var lg : array<f32, ${N}>; var mx = -3.0e38;
    for (var n = 0u; n < ${N}u; n++) {
      lg[n] = tot[n * ${K + 1}u + 1u + j] * inverseSqrt(tot[n * ${K + 1}u] / ${f(D)} + ${f(eps)});
      mx = max(mx, lg[n]);
    }
    var se = 0.0;
    for (var n = 0u; n < ${N}u; n++) { se += exp(lg[n] - mx); }
    let lse = mx + log(se);
    ${HASX ? `let lx = tot[${N * (K + 1)}u + 1u + j] * inverseSqrt(tot[${N * (K + 1)}u] / ${f(D)} + ${f(eps)});
    let w = 1.0 / (1.0 + exp(lse - lx));` : ""}
    var o : array<f32, 8>; var ss = 0.0;
    for (var e = 0u; e < 8u; e++) {
      var m = 0.0;
      for (var n = 0u; n < ${N}u; n++) { m += exp(lg[n] - lse) * v[n][e]; }
      ${HASX ? "m = m + w * (xv[e] - m);" : ""}
      o[e] = m; ss += m * m;
    }
    var sc = 1.0;
    ${NORM ? `let rs = subgroupAdd(ss); if (sl == 0u) { nrm[sg] = rs; }
    workgroupBarrier();
    var tss = 0.0; for (var i = 0u; i < nsg; i++) { tss += nrm[i]; }
    workgroupBarrier();
    sc = inverseSqrt(tss / ${f(D)} + ${f(eps)});` : ""}
    for (var e = 0u; e < 8u; e++) { outp[j * ${D}u + t + e * 256u] = o[e] * sc; }
  }
}`;
};

/** copy n floats */
const copySrc = (n) => /* wgsl */`
@group(0) @binding(0) var<storage, read> a : array<f32>;
@group(0) @binding(1) var<storage, read_write> b : array<f32>;
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid : vec3<u32>) { if (gid.x < ${n}u) { b[gid.x] = a[gid.x]; } }`;

/** embedding row (4-bit) -> rms_norm -> source 0. The token comes from tok[0] (the CPU's, or the previous step's sample). */
const embedSrc = (eps) => /* wgsl */`
@group(0) @binding(0) var<storage, read> tok : array<u32>;
@group(0) @binding(1) var<storage, read> wq : array<u32>;
@group(0) @binding(2) var<storage, read> ws : array<f32>;
@group(0) @binding(3) var<storage, read> wb : array<f32>;
@group(0) @binding(4) var<storage, read_write> outp : array<f32>;
${NIB}
${RED()}
@compute @workgroup_size(256)
fn main(@builtin(local_invocation_id) lid : vec3<u32>) {
  let t = lid.x; let row = tok[0];
  let q = nib(wq[row * 256u + t]); let g = row * 32u + t / 8u;
  let s = ws[g]; let b = wb[g];
  let v0 = q[0] * s + vec4<f32>(b); let v1 = q[1] * s + vec4<f32>(b);
  let r = inverseSqrt(wsum(t, dot(v0, v0) + dot(v1, v1)) / ${f(D)} + ${f(eps)});
  for (var i = 0u; i < 4u; i++) { outp[t * 8u + i] = v0[i] * r; outp[t * 8u + 4u + i] = v1[i] * r; }
}`;

/** sliding layers: q / k head norm (gain 1 + w, eps fp32), RoPE (first 64 dims, half split, ORT's cos / -sin tables), and the
 *  new k / v into the ring (slot pos % 128). Workgroup per head: 16 q heads then 4 k heads. */
const ropeAppendSrc = (H, HKV, ropeP) => /* wgsl */`
${INFO}
@group(0) @binding(0) var<uniform> info : Info;
@group(0) @binding(1) var<storage, read_write> q : array<f32>;   // q [H*128] (normed + rotated in place), then k, v [HKV*128] each
@group(0) @binding(2) var<storage, read> nw : array<f32>;        // q_norm [128], k_norm [128]
@group(0) @binding(3) var<storage, read> rope : array<f32>;      // cos [P][32], then -sin [P][32]
@group(0) @binding(4) var<storage, read_write> kc : array<f32>;  // [128 slots][HKV][128]
@group(0) @binding(5) var<storage, read_write> vc : array<f32>;
${RED(128)}
var<workgroup> nv : array<f32, 128>;
@compute @workgroup_size(128)
fn main(@builtin(workgroup_id) wg : vec3<u32>, @builtin(local_invocation_id) lid : vec3<u32>) {
  let d = lid.x; let h = wg.x; let isq = h < ${H}u;
  let kh = h - ${H}u;
  let a = q[h * 128u + d];                                      // q head h, or k head kh (k follows q)
  let gain = 1.0 + select(nw[128u + d], nw[d], isq);
  let r = inverseSqrt(wsum(d, a * a) / 128.0 + ${f(EPS32)});
  nv[d] = a * r * gain;
  workgroupBarrier();
  var o = nv[d];
  let P = info.pos; let i = d % 32u;
  let c = rope[P * 32u + i]; let ns = rope[${ropeP * 32}u + P * 32u + i];
  if (d < 32u) { o = nv[d] * c - nv[d + 32u] * ns; } else if (d < 64u) { o = nv[d] * c + nv[d - 32u] * ns; }
  if (isq) { q[h * 128u + d] = o; }
  else {
    let slot = P % ${WINDOW}u;
    kc[(slot * ${HKV}u + kh) * 128u + d] = o;
    vc[(slot * ${HKV}u + kh) * 128u + d] = q[${(H + HKV) * 128}u + kh * 128u + d];
  }
}`;

/** sliding attention: softmax at 1/sqrt(128) over the last 128 positions (ring); scores with a thread per key (32 vec4 loads), the value sum split over 4 key
 *  quarters (thread = 4 dims x one quarter of the keys) and merged in workgroup memory. Workgroup per q head. */
const attnSliding4Src = (H, HKV) => /* wgsl */`
${INFO}
@group(0) @binding(0) var<uniform> info : Info;
@group(0) @binding(1) var<storage, read> q : array<vec4<f32>>;
@group(0) @binding(2) var<storage, read> kc : array<vec4<f32>>;
@group(0) @binding(3) var<storage, read> vc : array<vec4<f32>>;
@group(0) @binding(4) var<storage, read_write> o : array<vec4<f32>>;
${RED(128)}
var<workgroup> p : array<f32, 128>;
var<workgroup> qh : array<vec4<f32>, 32>;
var<workgroup> part : array<vec4<f32>, 128>;
@compute @workgroup_size(128)
fn main(@builtin(workgroup_id) wg : vec3<u32>, @builtin(local_invocation_id) lid : vec3<u32>) {
  let h = wg.x; let t = lid.x; let kh = h / ${H / HKV}u; let P = info.pos;
  if (t < 32u) { qh[t] = q[h * 32u + t]; }
  workgroupBarrier();
  let j = i32(P) - 127 + i32(t);
  var s = -3.0e38;
  if (j >= 0) {
    let b = ((u32(j) % ${WINDOW}u) * ${HKV}u + kh) * 32u; var acc = 0.0;
    for (var d = 0u; d < 32u; d++) { acc += dot(qh[d], kc[b + d]); }
    s = acc / ${f(Math.sqrt(HD))};
  }
  let mx = wmax(t, s);
  let e = select(0.0, exp(s - mx), j >= 0);
  let sum = wsum(t, e);
  p[t] = e / sum;
  workgroupBarrier();
  let l = t % 32u; let qu = t / 32u;
  var acc = vec4<f32>(0.0);
  for (var k = qu * 32u; k < qu * 32u + 32u; k++) {
    let jj = i32(P) - 127 + i32(k);
    if (jj >= 0) { acc += p[k] * vc[((u32(jj) % ${WINDOW}u) * ${HKV}u + kh) * 32u + l]; }
  }
  part[t] = acc;
  workgroupBarrier();
  if (t < 32u) { o[h * 32u + t] = part[t] + part[t + 32u] + part[t + 64u] + part[t + 96u]; }
}`;

/** full layers: q head norm in place; raw k, normed k and v appended at pos. Workgroup per head: 16 q heads, 2 k heads. */
const fullKvSrc = (H, HKV, maxPos) => /* wgsl */`
${INFO}
@group(0) @binding(0) var<uniform> info : Info;
@group(0) @binding(1) var<storage, read_write> q : array<f32>;   // q [H*128] (normed in place), then k, v [HKV*128] each
@group(0) @binding(2) var<storage, read> nw : array<f32>;        // q_norm, k_norm
@group(0) @binding(3) var<storage, read_write> kraw : array<f32>;// [maxPos][HKV][128]
@group(0) @binding(4) var<storage, read_write> kn : array<f32>;
@group(0) @binding(5) var<storage, read_write> vv : array<f32>;
${RED(128)}
@compute @workgroup_size(128)
fn main(@builtin(workgroup_id) wg : vec3<u32>, @builtin(local_invocation_id) lid : vec3<u32>) {
  let d = lid.x; let h = wg.x; let isq = h < ${H}u; let kh = h - ${H}u; let P = info.pos;
  let a = q[h * 128u + d];
  let gain = 1.0 + select(nw[128u + d], nw[d], isq);
  let r = inverseSqrt(wsum(d, a * a) / 128.0 + ${f(EPS32)});
  if (isq) { q[h * 128u + d] = a * r * gain; }
  else {
    let i = (P * ${HKV}u + kh) * 128u + d;
    kraw[i] = a; kn[i] = a * r * gain; vv[i] = q[${(H + HKV) * 128}u + kh * 128u + d];
  }
}`;

/** sparse layers: q head norm (gain 1 + q_norm), in place. Workgroup per head. */
const qNormSrc = () => /* wgsl */`
@group(0) @binding(0) var<storage, read_write> q : array<f32>;
@group(0) @binding(1) var<storage, read> nw : array<f32>;
${RED(128)}
@compute @workgroup_size(128)
fn main(@builtin(workgroup_id) wg : vec3<u32>, @builtin(local_invocation_id) lid : vec3<u32>) {
  let d = lid.x; let i = wg.x * 128u + d; let a = q[i];
  let r = inverseSqrt(wsum(d, a * a) / 128.0 + ${f(EPS32)});
  q[i] = a * r * (1.0 + nw[d]);
}`;

/** Scores with one thread per key and all 8 query heads of its kv group (each key row read once, as vec4), grid (ceil((pos+1)/128), HKV).
 *  full: a_hj = relu(q_h.kn_j / sqrt(128))^2 for j <= pos, and with SEL also the selection score max_h q_h.kn_j for j <= pos - 128.
 *  sparse: s_hj = q_h.(k_norm-gained kraw_j) * rms(kraw_j)^-1 / sqrt(128) for the recent 128 and the selected keys, else -inf. */
const scores8Src = (kind, H, HKV, maxPos) => {
  const G = H / HKV, full = kind !== "sparse", sel = kind === "fullSel";
  return /* wgsl */`
${INFO}
@group(0) @binding(0) var<uniform> info : Info;
@group(0) @binding(1) var<storage, read> q : array<f32>;
@group(0) @binding(2) var<storage, read> kk : array<vec4<f32>>;      // full: kn, sparse: kraw  [maxPos][HKV][32]
@group(0) @binding(3) var<storage, read_write> sc : array<f32>;     // [H][maxPos]
${full ? (sel ? "@group(0) @binding(4) var<storage, read_write> ss : array<f32>;    // [HKV][maxPos]" : "")
  : `@group(0) @binding(4) var<storage, read> nw : array<f32>;
@group(0) @binding(5) var<storage, read> mask : array<u32>;`}
var<workgroup> qs : array<vec4<f32>, ${G * 32}>;
@compute @workgroup_size(128)
fn main(@builtin(workgroup_id) wg : vec3<u32>, @builtin(local_invocation_id) lid : vec3<u32>) {
  let g = wg.y; let t = lid.x; let P = info.pos;
  for (var i = t; i < ${G * 32}u; i += 128u) {
    let hh = i / 32u; let d = (i % 32u) * 4u; let b = (g * ${G}u + hh) * 128u + d;
    var v = vec4<f32>(q[b], q[b + 1u], q[b + 2u], q[b + 3u]);
    ${full ? "" : "v = v * (vec4<f32>(1.0) + vec4<f32>(nw[128u + d], nw[129u + d], nw[130u + d], nw[131u + d]));"}
    qs[i] = v;
  }
  workgroupBarrier();
  let j = wg.x * 128u + t;
  if (j > P) { return; }
  ${full ? "" : `if (!(j + ${RECENT}u > P || mask[g * ${maxPos}u + j] == 1u)) {
    for (var hh = 0u; hh < ${G}u; hh++) { sc[(g * ${G}u + hh) * ${maxPos}u + j] = -3.0e38; }
    return;
  }`}
  var acc : array<f32, ${G}>; var kq = 0.0;
  let kb = (j * ${HKV}u + g) * 32u;
  for (var d = 0u; d < 32u; d++) {
    let k = kk[kb + d]; ${full ? "" : "kq += dot(k, k);"}
    for (var hh = 0u; hh < ${G}u; hh++) { acc[hh] += dot(qs[hh * 32u + d], k); }
  }
  ${full ? `var m = -3.0e38;
  for (var hh = 0u; hh < ${G}u; hh++) {
    let r = max(acc[hh] / ${f(Math.sqrt(HD))}, 0.0); sc[(g * ${G}u + hh) * ${maxPos}u + j] = r * r; m = max(m, acc[hh]);
  }
  ${sel ? `if (P >= ${RECENT}u && j <= P - ${RECENT}u) { ss[g * ${maxPos}u + j] = m; }` : ""}`
  : `let rn = inverseSqrt(kq / 128.0 + ${f(EPS32)}) / ${f(Math.sqrt(HD))};
  for (var hh = 0u; hh < ${G}u; hh++) { sc[(g * ${G}u + hh) * ${maxPos}u + j] = acc[hh] * rn; }`}
}`;
};

/** top-1024 of the selection scores per kv group -> mask (same result as selMaskSrc: threshold = 1024th largest, ties by lowest
 *  index). 8-bit radix select in 4 passes over keys held in registers (each thread a contiguous run of candidates), histogram in
 *  workgroup atomics; ties resolved with a prefix sum over the threads' tie counts. One workgroup per kv group. */
const selMask8Src = (HKV, maxPos) => /* wgsl */`
${INFO}
@group(0) @binding(0) var<uniform> info : Info;
@group(0) @binding(1) var<storage, read> ss : array<f32>;
@group(0) @binding(2) var<storage, read_write> mask : array<u32>;  // [HKV][maxPos]
var<workgroup> hist : array<atomic<u32>, 256>;
var<workgroup> scan : array<u32, 256>;
var<workgroup> pre : u32;
var<workgroup> rem : u32;
fn key(v : f32) -> u32 { let u = bitcast<u32>(v); return select(u ^ 0x80000000u, ~u, (u >> 31u) == 1u); }
const PERMAX = ${Math.ceil(maxPos / 256)}u;
@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wg : vec3<u32>, @builtin(local_invocation_id) lid : vec3<u32>) {
  let g = wg.x; let t = lid.x; let P = info.pos;
  let C = select(0u, P - ${RECENT - 1}u, P >= ${RECENT}u);            // candidates 0 .. C-1
  let base = g * ${maxPos}u;
  if (C <= ${SPARSE_K}u) {
    for (var j = t; j < C; j += 256u) { mask[base + j] = 1u; }
    return;
  }
  let per = (C + 255u) / 256u; let j0 = t * per; let j1 = min(C, j0 + per);
  var ks : array<u32, PERMAX>;
  for (var i = 0u; i < per; i++) { if (j0 + i < j1) { ks[i] = key(ss[base + j0 + i]); } }
  if (t == 0u) { pre = 0u; rem = ${SPARSE_K}u; }
  var hi = 0u;                                                          // bits fixed so far
  for (var p = 0u; p < 4u; p++) {
    let sh = 24u - 8u * p;
    atomicStore(&hist[t], 0u);
    workgroupBarrier();
    let pf = pre;
    for (var i = 0u; i < per; i++) { if (j0 + i < j1 && (ks[i] & hi) == pf) { atomicAdd(&hist[(ks[i] >> sh) & 255u], 1u); } }
    workgroupBarrier();
    if (t == 0u) {                                                      // the bin holding the rem-th largest of the matching keys
      var cum = 0u; var b = 255i;
      for (; b > 0i; b--) { let h = atomicLoad(&hist[u32(b)]); if (cum + h >= rem) { break; } cum += h; }
      rem = rem - cum; pre = pre | (u32(b) << sh);
    }
    workgroupBarrier();
    hi = hi | (255u << sh);
  }
  let T = pre; let need = rem;                                          // keys > T are in; need more among keys == T
  var ties = 0u;
  for (var i = 0u; i < per; i++) { if (j0 + i < j1 && ks[i] == T) { ties++; } }
  scan[t] = ties; workgroupBarrier();
  for (var o = 1u; o < 256u; o = o << 1u) {                            // inclusive prefix sum
    var v = scan[t]; if (t >= o) { v += scan[t - o]; }
    workgroupBarrier(); scan[t] = v; workgroupBarrier();
  }
  var rank = scan[t] - ties;                                            // ties before this thread's run
  for (var i = 0u; i < per; i++) {
    if (j0 + i < j1) {
      var m = 0u;
      if (ks[i] > T) { m = 1u; } else if (ks[i] == T) { if (rank < need) { m = 1u; } rank++; }
      mask[base + j0 + i] = m;
    }
  }
}`;

/** Attention output split over positions (flash-decoding), for the full (ReLU^2) and sparse (softmax) layers (a workgroup per head
 *  walking every position cost ~16 ms a token at 2K context). Part: grid (ceil(n/64), H), each workgroup 64 positions of one
 *  head -> partial [acc(128), m, s]. Combine: workgroup per head merges the chunks (softmax: log-sum-exp rescaling). */
const ACH = 64, ACMAX = (maxPos) => Math.ceil(maxPos / ACH);
const attnPartSrc = (soft, H, HKV, maxPos) => /* wgsl */`
${INFO}
@group(0) @binding(0) var<uniform> info : Info;
@group(0) @binding(1) var<storage, read> sc : array<f32>;
@group(0) @binding(2) var<storage, read> vv : array<f32>;
@group(0) @binding(3) var<storage, read_write> part : array<f32>;   // [H][chunks][130]
@compute @workgroup_size(128)
fn main(@builtin(workgroup_id) wg : vec3<u32>, @builtin(local_invocation_id) lid : vec3<u32>) {
  let c = wg.x; let h = wg.y; let t = lid.x; let kh = h / ${H / HKV}u; let n = info.pos + 1u;
  let j0 = c * ${ACH}u; let j1 = min(n, j0 + ${ACH}u); let b = h * ${maxPos}u;
  var m = ${soft ? "-3.0e38" : "0.0"};
  ${soft ? "for (var j = j0; j < j1; j++) { m = max(m, sc[b + j]); }" : ""}
  var acc = 0.0; var se = 0.0;
  for (var j = j0; j < j1; j++) {
    let s = sc[b + j];
    ${soft ? "var e = 0.0; if (s > -1.0e38) { e = exp(s - m); }" : "let e = s;"}
    se += e; acc += e * vv[(j * ${HKV}u + kh) * 128u + t];
  }
  let o = (h * ${ACMAX(maxPos)}u + c) * 130u;
  part[o + t] = acc;
  if (t == 0u) { part[o + 128u] = m; part[o + 129u] = se; }
}`;
const attnCombineSrc = (soft, H, maxPos) => /* wgsl */`
${INFO}
@group(0) @binding(0) var<uniform> info : Info;
@group(0) @binding(1) var<storage, read> part : array<f32>;
@group(0) @binding(2) var<storage, read_write> o : array<f32>;
@compute @workgroup_size(128)
fn main(@builtin(workgroup_id) wg : vec3<u32>, @builtin(local_invocation_id) lid : vec3<u32>) {
  let h = wg.x; let t = lid.x; let C = (info.pos + ${ACH}u) / ${ACH}u; let b = h * ${ACMAX(maxPos)}u * 130u;
  var M = -3.0e38;
  ${soft ? "for (var c = 0u; c < C; c++) { M = max(M, part[b + c * 130u + 128u]); }" : "M = 0.0;"}
  var acc = 0.0; var se = 0.0;
  for (var c = 0u; c < C; c++) {
    let w = ${soft ? "exp(part[b + c * 130u + 128u] - M)" : "1.0"};
    acc += part[b + c * 130u + t] * w; se += part[b + c * 130u + 129u] * w;
  }
  o[h * 128u + t] = acc / ${soft ? "se" : `(${f(Math.sqrt(HD))} + se)`};
}`;

/** engram: gate from W_k e and the mix he, gated W_v e, its RMS norm into the conv history, causal conv (4 taps, dilation 3),
 *  -> source 1 = gated + silu(conv). One workgroup, 8 dims a thread. */
const engramSrc = () => /* wgsl */`
${INFO}
@group(0) @binding(0) var<uniform> info : Info;
@group(0) @binding(1) var<storage, read> kv : array<f32>;        // W_k e [D], W_v e [D]
@group(0) @binding(2) var<storage, read> he : array<f32>;
@group(0) @binding(3) var<storage, read> cw : array<f32>;        // conv weight [D][4]
@group(0) @binding(4) var<storage, read_write> hist : array<f32>;// [12][D]
@group(0) @binding(5) var<storage, read_write> outp : array<f32>;
${RED()}
@compute @workgroup_size(256)
fn main(@builtin(local_invocation_id) lid : vec3<u32>) {
  let t = lid.x; let P = info.pos;
  var kk = 0.0; var xx = 0.0; var kx = 0.0;
  for (var e = 0u; e < 8u; e++) { let d = t + e * 256u; let k = kv[d]; let h = he[d]; kk += k * k; xx += h * h; kx += k * h; }
  kk = wsum(t, kk); xx = wsum(t, xx); kx = wsum(t, kx);
  let z = kx * inverseSqrt(kk / ${f(D)} + ${f(EPS32)}) * inverseSqrt(xx / ${f(D)} + ${f(EPS32)}) / ${f(Math.sqrt(D))};
  let g = 1.0 / (1.0 + exp(-(sign(z) * sqrt(abs(z) + 1e-6))));
  var gated : array<f32, 8>; var ss = 0.0;
  for (var e = 0u; e < 8u; e++) { gated[e] = g * kv[${D}u + t + e * 256u]; ss += gated[e] * gated[e]; }
  let r = inverseSqrt(wsum(t, ss) / ${f(D)} + ${f(EPS32)});
  for (var e = 0u; e < 8u; e++) {
    let d = t + e * 256u; let nrm = gated[e] * r;
    var conv = nrm * cw[d * 4u + 3u];
    for (var i = 0u; i < 3u; i++) {
      let back = (3u - i) * ${CONV_DIL}u;
      if (P >= back) { conv += hist[((P - back) % ${HRING}u) * ${D}u + d] * cw[d * 4u + i]; }
    }
    hist[(P % ${HRING}u) * ${D}u + d] = nrm;
    outp[d] = gated[e] + conv / (1.0 + exp(-conv));
  }
}`;

/** top-K selection (value desc, index asc) in one 256-thread workgroup; `scan` calls consider(value, index) for this thread's items.
 *  Leaves topv / topi [K] in workgroup memory, K = 1 when greedy else 20 (opens main; the caller closes it). */
const TOPK_SEL = (scan) => `
var<workgroup> bv : array<f32, 256>;
var<workgroup> bi : array<u32, 256>;
var<workgroup> topv : array<f32, 20>;
var<workgroup> topi : array<u32, 20>;
var<private> cv_ : f32; var<private> ci_ : u32; var<private> pv : f32; var<private> pi : u32; var<private> first : bool;
fn consider(x : f32, j : u32) {
  let below = first || x < pv || (x == pv && j > pi);
  if (below && (x > cv_ || (x == cv_ && j < ci_))) { cv_ = x; ci_ = j; }
}
@compute @workgroup_size(256)
fn main(@builtin(local_invocation_id) lid : vec3<u32>, @builtin(workgroup_id) wg : vec3<u32>) {
  let t = lid.x; pv = 3.4e38; pi = 0u; first = true;
  let K = select(20u, 1u, sp.greedy == 1u);
  for (var k = 0u; k < K; k++) {
    cv_ = -3.4e38; ci_ = 0xffffffffu;
    ${scan}
    bv[t] = cv_; bi[t] = ci_; workgroupBarrier();
    for (var s = 128u; s > 0u; s = s >> 1u) {
      if (t < s) { let a = bv[t + s]; let b = bi[t + s]; if (a > bv[t] || (a == bv[t] && b < bi[t])) { bv[t] = a; bi[t] = b; } }
      workgroupBarrier();
    }
    pv = bv[0]; pi = bi[0]; first = false;
    if (t == 0u) { topv[k] = pv; topi[k] = pi; }
    workgroupBarrier();
  }`;
/** Sampling on the GPU, as engine.mjs's sample(): greedy = first argmax; else the top 20 (value desc, index asc), softmax at the
 *  temperature, the smallest prefix with cumulative probability >= top_p, renormalised, drawn with the CPU's uniform r.
 *  Writes the token to tok[0] (the next step's embedding input). */
const sampleSrc = () => /* wgsl */`
struct SP { temp : f32, topp : f32, r : f32, greedy : u32 };
@group(0) @binding(0) var<storage, read> cv : array<f32>;       // per-chunk top K (samplePart): [64][20] values
@group(0) @binding(1) var<uniform> sp : SP;
@group(0) @binding(2) var<storage, read> ci : array<u32>;       //                                [64][20] token ids
@group(0) @binding(3) var<storage, read_write> tok : array<u32>;
${TOPK_SEL(`for (var c = t; c < 1280u; c += 256u) { if (c % 20u < K) { consider(cv[c], ci[c]); } }`)}
  if (t == 0u) {
    if (sp.greedy == 1u) { tok[0] = topi[0]; return; }
    var p : array<f32, 20>; var z = 0.0;
    for (var k = 0u; k < 20u; k++) { p[k] = exp((topv[k] - topv[0]) / sp.temp); z += p[k]; }
    var cum = 0.0; var keep = 0u;
    for (; keep < 20u; keep++) { cum += p[keep] / z; if (cum >= sp.topp) { keep++; break; } }
    keep = max(1u, min(keep, 20u));
    var zz = 0.0; for (var k = 0u; k < keep; k++) { zz += p[k] / z; }
    var r = sp.r * zz; var pick = topi[keep - 1u];
    for (var k = 0u; k < keep; k++) { r -= p[k] / z; if (r <= 0.0) { pick = topi[k]; break; } }
    tok[0] = pick;
  }
}`;
/** sampling, stage 1: 64 workgroups, each the top K (20, or 1 when greedy) of its 1024 logits. */
const samplePartSrc = () => /* wgsl */`
struct SP { temp : f32, topp : f32, r : f32, greedy : u32 };
@group(0) @binding(0) var<storage, read> lg : array<f32>;
@group(0) @binding(1) var<uniform> sp : SP;
@group(0) @binding(2) var<storage, read_write> cv : array<f32>;
@group(0) @binding(3) var<storage, read_write> ci : array<u32>;
${TOPK_SEL(`for (var m = 0u; m < 4u; m++) { let j = wg.x * 1024u + m * 256u + t; consider(lg[j], j); }`)}
  if (t < K) { cv[wg.x * 20u + t] = topv[t]; ci[wg.x * 20u + t] = topi[t]; }
}`;

// Engram rows on the GPU: hash the token n-grams (engram_hash.mjs, 64-bit arithmetic on u32 pairs) and gather + dequantise the
// 16 rows from the page-packed table held in two GPU buffers (order 2, order 3). Writes eIn [16][128] and records
// the token in a 4-slot ring by position (th), so the next step can hash without the CPU: decode needs no CPU round trip.
const engramGatherSrc = (pages) => /* wgsl */`
@group(0) @binding(0) var<uniform> info : vec4<u32>;
@group(0) @binding(1) var<storage, read> tok : array<u32>;
@group(0) @binding(2) var<storage, read_write> th : array<u32>;
@group(0) @binding(3) var<storage, read> t0 : array<u32>;
@group(0) @binding(4) var<storage, read> t1 : array<u32>;
@group(0) @binding(5) var<storage, read_write> outv : array<f32>;
const K1 = vec2<u32>(0x7F4A7C15u, 0x9E3779B9u);
const K2 = vec2<u32>(0x1CE4E5B9u, 0xBF58476Du);
const K3 = vec2<u32>(0x133111EBu, 0x94D049BBu);
const SEED = vec2<u32>(0xF3BCC909u, 0x6A09E667u);
const SALT = vec2<u32>(0x6659FD93u, 0xD6E8FEB8u);
const CN = vec2<u32>(0x85A308D3u, 0x243F6A88u);
const UL = array<u32, 24>(1u, 3u, 5u, 9u, 11u, 13u, 15u, 17u, 19u, 23u, 25u, 27u, 29u, 31u, 33u, 37u, 39u, 41u, 43u, 45u, 47u, 51u, 53u, 55u);
fn mul64(a : vec2<u32>, b : vec2<u32>) -> vec2<u32> {
  let a0 = a.x & 0xffffu; let a1 = a.x >> 16u; let b0 = b.x & 0xffffu; let b1 = b.x >> 16u;
  let p00 = a0 * b0; let p01 = a0 * b1; let p10 = a1 * b0; let p11 = a1 * b1;
  let mid = (p00 >> 16u) + (p01 & 0xffffu) + (p10 & 0xffffu);
  let lo = (p00 & 0xffffu) | (mid << 16u);
  let hi = p11 + (p01 >> 16u) + (p10 >> 16u) + (mid >> 16u) + a.x * b.y + a.y * b.x;
  return vec2<u32>(lo, hi);
}
fn shr(a : vec2<u32>, n : u32) -> vec2<u32> {
  if (n >= 32u) { return vec2<u32>(a.y >> (n - 32u), 0u); }
  return vec2<u32>((a.x >> n) | (a.y << (32u - n)), a.y >> n);
}
fn fin(h0 : vec2<u32>) -> vec2<u32> {
  var h = mul64(h0 ^ shr(h0, 30u), K2); h = mul64(h ^ shr(h, 27u), K3); return h ^ shr(h, 31u);
}
fn mod64(a : vec2<u32>, m : u32) -> u32 {                      // m < 2^24
  var r = 0u;
  for (var i = 0u; i < 4u; i++) { r = (r * 256u + ((a.y >> (24u - 8u * i)) & 255u)) % m; }
  for (var i = 0u; i < 4u; i++) { r = (r * 256u + ((a.x >> (24u - 8u * i)) & 255u)) % m; }
  return r;
}
var<workgroup> pg : array<u32, 4>;
var<workgroup> pa : array<u32, 4>;
var<workgroup> ps : array<u32, 4>;
@compute @workgroup_size(256)
fn main(@builtin(local_invocation_id) lid : vec3<u32>) {
  let t = lid.x; let pos = info.x;
  if (t < 4u) {
    let k = t / 2u; let g = t % 2u; let n = k + 2u;               // orders [2, 3], heads / pageHeads = 2 groups
    var r = SEED;
    for (var L = i32(n) - 1; L >= 0; L--) {
      let l = u32(L);
      var id = 65536u;                                           // pad
      if (pos >= l) { if (l == 0u) { id = tok[0]; } else { id = th[(pos - l) % 4u]; } }
      r = mul64(r ^ vec2<u32>(id, 0u), K1); r = r ^ shr(r, 29u);
    }
    let hn = fin(r ^ mul64(CN, vec2<u32>(n + 1u, 0u)));
    let hp = fin(hn ^ mul64(K1, vec2<u32>(1u + 64u * n + g, 0u)));
    pg[t] = mod64(shr(hp, 1u), ${pages}u);
    let hs = fin(hp ^ SALT);
    pa[t] = mod64(shr(hs, 1u), 56u); ps[t] = UL[mod64(shr(hs, 40u), 24u)];
  }
  workgroupBarrier();
  if (t == 0u) { th[pos % 4u] = tok[0]; }
  for (var idx = t; idx < 2048u; idx += 256u) {
    let row = idx / 128u; let e = idx % 128u; let kg = row / 4u; let j = row % 4u;
    let slot = (pa[kg] + j * ps[kg]) % 56u; let base = pg[kg] * 1008u;
    var w : u32; var sc : u32; var bi : u32;
    if (kg < 2u) { w = t0[base + slot * 16u + e / 8u]; sc = t0[base + 896u + slot]; bi = t0[base + 952u + slot]; }
    else { w = t1[base + slot * 16u + e / 8u]; sc = t1[base + 896u + slot]; bi = t1[base + 952u + slot]; }
    let q = f32((w >> (4u * (e % 8u))) & 15u);
    let s2 = unpack2x16float(sc); let b2 = unpack2x16float(bi);
    let hi = e >= 64u;
    outv[idx] = q * select(s2.x, s2.y, hi) + select(b2.x, b2.y, hi);
  }
}`;

// ------------------------------------------------------------------------------------------------ engine
export class WgslDecoder {
  /** device: ORT's GPUDevice (must have "subgroups"). fetchTensor(key) -> {t, u8}: a model.safetensors tensor. experts: the ORT
   *  graph's expert tensors (fcN / projT). rope: Float32Array, the graph's cos table [P][32] then its -sin table [P][32].
   *  cfg: config.json. engram: {pages, chunk(k, page0, n) -> Promise<Uint8Array n x 4032>}, the page-packed table of order k. */
  static async create({ device, fetchTensor, experts, rope, cfg, engram, maxPos = 8192 + 64, onStatus = () => {} }) {
    if (!device.features.has("subgroups")) throw new Error("WGSL decoder: the GPU device has no \"subgroups\" feature");
    if (rope.length / 64 < maxPos) throw new Error(`WGSL decoder: the graph's RoPE table covers ${rope.length / 64} positions, fewer than ${maxPos}`);
    const dec = new WgslDecoder(device, cfg, maxPos);
    await dec.#init(fetchTensor, experts, rope, engram, onStatus);
    return dec;
  }
  constructor(device, cfg, maxPos) {
    this.device = device; this.cfg = cfg; this.maxPos = maxPos; this.eps = cfg.rms_norm_eps;
    this.L = cfg.num_hidden_layers; this.types = cfg.layer_types; this.mlpTypes = cfg.mlp_layer_types;
    this.pipes = new Map();
  }
  #buf(size, usage = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC) {
    return this.device.createBuffer({ size: Math.max(16, Math.ceil(size / 4) * 4), usage });
  }
  #upload(data) { const u8 = new Uint8Array(data.buffer, data.byteOffset, data.byteLength); const b = this.#buf(u8.byteLength); this.device.queue.writeBuffer(b, 0, u8); return b; }
  #pipe(key, code) {
    let p = this.pipes.get(key);
    if (!p) {
      const module = this.device.createShaderModule({ code, label: key });
      (this.modules ||= []).push([key, module]);
      p = this.device.createComputePipeline({ layout: "auto", label: key, compute: { module, entryPoint: "main" } }); this.pipes.set(key, p);
    }
    return p;
  }
  #bg(pipe, bufs) {
    return this.device.createBindGroup({ layout: pipe.getBindGroupLayout(0), entries: bufs.map((b, binding) =>
      ({ binding, resource: b.buffer ? { buffer: b.buffer, offset: b.offset || 0, size: b.size } : { buffer: b } })) });
  }
  /** a dispatch recorded once: [pipeline, bind group, groups (number | () => [x, y])] */
  #op(list, key, code, bufs, groups) { const p = this.#pipe(key, code); list.push([p, this.#bg(p, bufs), groups]); }

  async #init(fetchTensor, experts, rope, engram, onStatus) {
    const cfg = this.cfg, dev = this.device, MP = this.maxPos;
    const bf = (u8) => { const u16 = new Uint16Array(u8.buffer, u8.byteOffset, u8.byteLength / 2), o = new Uint32Array(u16.length);
      for (let k = 0; k < u16.length; k++) o[k] = u16[k] << 16; return new Float32Array(o.buffer); };
    const get = async (key) => (await fetchTensor(key)).u8;
    const pack = (s, b) => { const S = new Uint16Array(s.buffer, s.byteOffset, s.byteLength / 2), B = new Uint16Array(b.buffer, b.byteOffset, b.byteLength / 2), o = new Uint32Array(S.length);
      for (let k = 0; k < S.length; k++) o[k] = S[k] | (B[k] << 16); return o; };           // bf16 scale (low) | bf16 bias (high), as gemv2Src reads them
    const q4w = (w, s, b, sep) => ({ q: this.#upload(new Uint32Array(w.buffer, w.byteOffset, w.byteLength / 4)), sb: this.#upload(pack(s, b)),
      ...(sep ? { s: this.#upload(bf(s)), b: this.#upload(bf(b)) } : {}) });          // sep: fp32 scales / biases too (embedding)
    const q4 = async (base, sep = false) => { const [w, s, b] = await Promise.all([get(base + ".weight"), get(base + ".scales"), get(base + ".biases")]); return q4w(w, s, b, sep); };
    const vecf = async (key) => this.#upload(bf(await get(key)));
    // q, k, v of one layer as one weight (rows concatenated): one GEMV writes q | k | v into this.qkv
    const q4cat = async (bases) => {
      const parts = await Promise.all(bases.map(b => Promise.all([get(b + ".weight"), get(b + ".scales"), get(b + ".biases")])));
      const cat = (arrs) => { const o = new Uint8Array(arrs.reduce((n, a) => n + a.byteLength, 0)); let k = 0; for (const a of arrs) { o.set(a, k); k += a.byteLength; } return o; };
      return q4w(cat(parts.map(p => p[0])), cat(parts.map(p => p[1])), cat(parts.map(p => p[2])));
    };
    let done = 0; const tick = () => onStatus(`Loading the WGSL decoder… ${++done}`);

    // weights
    this.wte = await q4("transformer.wte", true); tick();
    this.lmHead = await q4("lm_head"); tick();
    this.engram = { wk: await q4("engram.w_k"), wv: await q4("engram.w_v"), conv: await vecf("engram.conv_weight") }; tick();
    // pseudo-queries in the reference order: per layer attn (+ bridged kv_res queries), mlp; engram; final
    const bridges = {}; for (const [j, s] of Object.entries(cfg.kv_sources)) if (+j !== s) (bridges[s] ||= []).push(+j);
    for (const s in bridges) bridges[s].sort((a, b) => a - b);
    this.bridges = bridges;
    const qrows = [], cols = {};
    for (let i = 0; i < this.L; i++) {
      cols["attn" + i] = qrows.length; qrows.push(`transformer.h.${i}.attn_res_query`);
      for (const j of bridges[i] || []) qrows.push(`transformer.h.${j}.kv_res_query`);
      cols["mlp" + i] = qrows.length; qrows.push(`transformer.h.${i}.mlp_res_query`);
    }
    cols.engram = qrows.length; qrows.push("engram_res_query"); cols.final = qrows.length; qrows.push("final_res_query");
    const Q = new Float32Array(qrows.length * D);
    for (let r = 0; r < qrows.length; r++) Q.set(bf(await get(qrows[r])), r * D);
    this.queries = this.#upload(Q); this.cols = cols;
    this.layers = [];
    for (let i = 0; i < this.L; i++) {
      const p = `transformer.h.${i}.`, type = this.types[i], Lw = { i, type };
      Lw.cp = await q4(p + "attn.c_proj");
      if (type === "sparse_attention" || (type === "full_attention" && cfg.kv_sources[i] !== i)) {
        Lw.cq = await q4(p + "attn.c_q");
        if (type !== "sparse_attention") { Lw.ck = await q4(p + "attn.c_k"); Lw.cv = await q4(p + "attn.c_v"); }
      } else Lw.qkv = await q4cat(["c_q", "c_k", "c_v"].map(n => p + "attn." + n));
      const nw = new Float32Array(256); nw.set(bf(await get(p + "attn.q_norm.weight")), 0); nw.set(bf(await get(p + "attn.k_norm.weight")), 128);
      Lw.nw = this.#upload(nw);
      if (this.mlpTypes[i] === "dense") { Lw.fc = await q4(p + "mlp.c_fc"); Lw.proj = await q4(p + "mlp.c_proj"); }
      else {
        Lw.sfc = await q4(p + "mlp.shared_fc"); Lw.sproj = await q4(p + "mlp.shared_proj");
        Lw.router = await vecf(p + "mlp.router.weight"); Lw.ebias = await vecf(p + "mlp.expert_bias");
        const g = (n) => experts[`e${i}_${n}`].gpuBuffer;
        Lw.ex = { fq: g("fcN_q4"), pq: g("projT_q4") };
        const enc = dev.createCommandEncoder(), pass = enc.beginComputePass();   // packed scale + bias copies of ORT's fp32 ones
        for (const [s_, c_, k] of [[g("fcN_sc"), g("fcN_c"), "sbF"], [g("projT_sc"), g("projT_c"), "sbP"]]) {
          const n = s_.size / 4, o = this.#buf(n * 4, GPUBufferUsage.STORAGE), p_ = this.#pipe(`packScales${n}`, packScalesSrc(n));
          pass.setPipeline(p_); pass.setBindGroup(0, this.#bg(p_, [s_, c_, o])); pass.dispatchWorkgroups(Math.min(65535, Math.ceil(n / 256)), Math.ceil(n / 256 / 65535));
          Lw.ex[k] = o;
        }
        pass.end(); dev.queue.submit([enc.finish()]);
      }
      this.layers.push(Lw); tick();
    }

    // state
    const f4 = (n) => this.#buf(n * 4);
    this.info = dev.createBuffer({ size: 16, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    this.tok = this.#buf(16); this.sp = dev.createBuffer({ size: 16, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    this.th = this.#buf(16); this.stagePool = [];                       // token ring for the GPU engram; token readback buffers (one per step in flight)
    this.src = f4(9 * D); this.x = f4(D); this.hs = f4(3 * D); this.taps = {}; for (const j of [20, 24, 28]) this.taps[j] = f4(D);
    this.qkv = f4(24 * HD); this.qb = { buffer: this.qkv, offset: 0, size: 16 * HD * 4 }; this.ob = f4(16 * HD);
    this.scores = f4(16 * MP); this.selS = f4(2 * MP); this.mask = f4(2 * MP);
    this.rl = f4(E); this.route = f4(16); this.sh = f4(SH); this.act = f4(TOPK * F); this.hmlp = f4(D);
    this.eIn = f4(D); this.ekv = f4(2 * D); this.hist = f4(HRING * D);
    this.logits = f4(VOCAB); this.staging = dev.createBuffer({ size: VOCAB * 4, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
    this.rope = this.#upload(rope); this.ropeP = rope.length / 64;     // cos [P][32], then -sin [P][32]
    this.cache = {};
    for (let i = 0; i < this.L; i++) {
      const t = this.types[i];
      if (t === "sliding_attention") this.cache[i] = { k: f4(WINDOW * 4 * HD), v: f4(WINDOW * 4 * HD) };
      if (t === "full_attention") this.cache[i] = { k: f4(MP * 2 * HD), kn: f4(MP * 2 * HD), v: f4(MP * 2 * HD) };
    }
    await this.#loadEngramTable(engram.pages, engram.chunk, onStatus);
    dev.pushErrorScope("validation");
    this.#record();
    const err = await dev.popErrorScope();
    this.initError = err ? err.message : null;
  }

  /** Shader compile errors and the first validation error of building the step (null when all is well). */
  async diagnostics() {
    const msgs = [];
    for (const [key, m] of this.modules || []) for (const x of (await m.getCompilationInfo()).messages) if (x.type === "error") msgs.push(`${key}:${x.lineNum}:${x.linePos} ${x.message}`);
    return { compile: msgs, init: this.initError };
  }

  /** Record every dispatch of a step once (static bind groups; only pos / token change, through the uniform). */
  #record() {
    const ops = [], cfg = this.cfg, eps = this.eps, MP = this.maxPos;
    const view = (buffer, i, n = D) => ({ buffer, offset: i * n * 4, size: n * 4 });
    const rpw = rpwFor;
    const gemv = (list, Wt, K, R, mode, x, y) => this.#op(list, `gemv2_${K}x${R}${mode}_${rpw(R)}`, gemv2Src(K, R, mode), [x, Wt.q, Wt.sb, y], Math.ceil(R / rpw(R)));
    const copy = (list, a, b, n = D) => this.#op(list, `copy${n}`, copySrc(n), [a, b], Math.ceil(n / 256));
    const starts = new Set(); for (let s = cfg.attn_res_block; s < this.L; s += cfg.attn_res_block) starts.add(s);
    const blockFull = (i) => { for (let j = i - 1; j >= 0; j--) if (this.types[j] === "full_attention") return j; };
    const selects = (i) => this.types.some((t, j) => j > i && t === "sparse_attention" && blockFull(j) === i);
    const mix = (list, N, K, hasX, norm, col) => this.#op(list, `mix${N}_${K}_${+hasX}_${+norm}_${col}`, mixSrc(N, K, hasX, norm, col, eps),
      [{ buffer: this.src, offset: 0, size: 9 * D * 4 }, this.queries, this.x, this.hs], 1);
    const hs = (j) => view(this.hs, j);
    this.apart = this.apart || this.#buf(16 * ACMAX(MP) * 130 * 4);
    const attnOut = (soft, v) => {
      this.#op(ops, soft ? "sparsePart" : "fullPart", attnPartSrc(soft, 16, 2, MP), [this.info, this.scores, v, this.apart], () => [Math.ceil((this.pos + 1) / ACH), 16]);
      this.#op(ops, soft ? "sparseCombine" : "fullCombine", attnCombineSrc(soft, 16, MP), [this.info, this.apart, this.ob], 16);
    };

    this.#op(ops, "embed", embedSrc(eps), [this.tok, this.wte.q, this.wte.s, this.wte.b, view(this.src, 0)], 1);
    this.#op(ops, "engramGather", engramGatherSrc(this.etab.pages), [this.info, this.tok, this.th, this.etab.b[0], this.etab.b[1], this.eIn], 1);
    let N = 1, xNone = true;
    for (let i = 0; i < this.L; i++) {
      const Lw = this.layers[i], t = this.types[i], C = this.cache[i];
      if (starts.has(i)) { copy(ops, this.x, view(this.src, N)); N++; xNone = true; }
      const Kq = 1 + (this.bridges[i] || []).length;
      mix(ops, N, Kq, !xNone, true, this.cols["attn" + i]);
      for (let j = 1; j < Kq; j++) copy(ops, hs(j), this.taps[this.bridges[i][j - 1]]);
      if (t === "sliding_attention") {
        gemv(ops, Lw.qkv, D, 24 * HD, "store", hs(0), { buffer: this.qkv, offset: 0, size: 24 * HD * 4 });
        this.#op(ops, "rope16_4", ropeAppendSrc(16, 4, this.ropeP), [this.info, { buffer: this.qkv, offset: 0, size: 24 * HD * 4 }, Lw.nw, this.rope, C.k, C.v], 20);
        this.#op(ops, "attnSliding4", attnSliding4Src(16, 4), [this.info, this.qb, C.k, C.v, this.ob], 16);
      } else if (t === "full_attention") {
        const srcIn = cfg.kv_sources[i] === i ? hs(0) : this.taps[i];
        const kvv = (j) => ({ buffer: this.qkv, offset: (16 + 2 * j) * HD * 4, size: 2 * HD * 4 });
        if (Lw.qkv) gemv(ops, Lw.qkv, D, 20 * HD, "store", hs(0), { buffer: this.qkv, offset: 0, size: 20 * HD * 4 });
        else { gemv(ops, Lw.cq, D, 16 * HD, "store", hs(0), this.qb); gemv(ops, Lw.ck, D, 2 * HD, "store", srcIn, kvv(0)); gemv(ops, Lw.cv, D, 2 * HD, "store", srcIn, kvv(1)); }
        this.#op(ops, "fullKv", fullKvSrc(16, 2, MP), [this.info, { buffer: this.qkv, offset: 0, size: 20 * HD * 4 }, Lw.nw, C.k, C.kn, C.v], 18);
        const sel = selects(i), g128 = () => [Math.ceil((this.pos + 1) / 128), 2];
        this.#op(ops, sel ? "fullScoresSel" : "fullScores8", scores8Src(sel ? "fullSel" : "full", 16, 2, MP), [this.info, this.qb, C.kn, this.scores, ...(sel ? [this.selS] : [])], g128);
        attnOut(false, C.v);
        if (selects(i)) {
          this.#op(ops, "selMask8", selMask8Src(2, MP), [this.info, this.selS, this.mask], 2);
        }
      } else {                                                           // sparse
        const F_ = this.cache[blockFull(i)];
        gemv(ops, Lw.cq, D, 16 * HD, "store", hs(0), this.qb);
        this.#op(ops, "qNorm", qNormSrc(), [this.qb, Lw.nw], 16);
        this.#op(ops, "sparseScores8", scores8Src("sparse", 16, 2, MP), [this.info, this.qb, F_.k, this.scores, Lw.nw, this.mask], () => [Math.ceil((this.pos + 1) / 128), 2]);
        attnOut(true, F_.v);
      }
      gemv(ops, Lw.cp, 16 * HD, D, xNone ? "store" : "add", this.ob, this.x);
      xNone = false;
      // MLP
      mix(ops, N, 1, true, true, this.cols["mlp" + i]);
      if (this.mlpTypes[i] === "dense") {
        gemv(ops, Lw.fc, D, D, "relu2", hs(0), this.hmlp);
        gemv(ops, Lw.proj, D, D, "add", this.hmlp, this.x);
      } else {
        this.#op(ops, "dotRows2048", dotRowsSrc(D), [hs(0), Lw.router, this.rl], E);
        this.#op(ops, "router", routerSrc(), [this.rl, Lw.ebias, this.route], 1);
        gemv(ops, Lw.sfc, D, SH, "relu2", hs(0), this.sh);
        gemv(ops, Lw.sproj, SH, D, "add", this.sh, this.x);
        this.#op(ops, `moeFc2_${rpw(TOPK * F, "moefc")}`, gemv2Src(D, TOPK * F, "store", "moefc"), [hs(0), this.route, Lw.ex.fq, Lw.ex.sbF, this.act], TOPK * F / rpw(TOPK * F, "moefc"));
        this.#op(ops, `moeProj2_${rpw(D, "moeproj")}`, gemv2Src(TOPK * F, D, "add", "moeproj"), [this.route, Lw.ex.pq, Lw.ex.sbP, this.act, this.x], D / rpw(D, "moeproj"));
      }
      if (i === cfg.engram_layer) {
        mix(ops, N, 1, true, false, this.cols.engram);
        gemv(ops, this.engram.wk, D, D, "store", this.eIn, view(this.ekv, 0));
        gemv(ops, this.engram.wv, D, D, "store", this.eIn, view(this.ekv, 1));
        this.#op(ops, "engram", engramSrc(), [this.info, this.ekv, hs(0), this.engram.conv, this.hist, view(this.src, N)], 1);
        N++;
      }
    }
    mix(ops, N, 1, true, true, this.cols.final);
    gemv(ops, this.lmHead, D, VOCAB, "softcap", hs(0), this.logits);
    this.ops = ops;
    this.cand = this.cand || [this.#buf(64 * 20 * 4), this.#buf(64 * 20 * 4)];
    const sampler = []; this.#op(sampler, "samplePart", samplePartSrc(), [this.logits, this.sp, this.cand[0], this.cand[1]], VOCAB / 1024);
    this.#op(sampler, "sample", sampleSrc(), [this.cand[0], this.sp, this.cand[1], this.tok], 1); this.sampleOps = sampler;
  }

  /** One decode token at position pos. token: the input id, or null when the previous step's sample is already in tok[0].
   *  Returns {logits} (readLogits) and / or {token} (sample: {temperature, topP}). Everything is submitted before the first await,
   *  so a caller can queue the next step before this one's result comes back. */
  async step(token, pos, { readLogits = true, sample = null } = {}) {
    if (pos + 1 > this.maxPos) throw new Error("WGSL decoder: context limit");
    const dev = this.device; this.pos = pos;
    dev.queue.writeBuffer(this.info, 0, new Uint32Array([pos, 0, 0, 0]));
    if (token != null) dev.queue.writeBuffer(this.tok, 0, new Uint32Array([token, 0, 0, 0]));
    if (sample) {
      const greedy = !(sample.temperature > 0);
      const b = new ArrayBuffer(16); new Float32Array(b, 0, 3).set([greedy ? 1 : sample.temperature, sample.topP ?? 0.95, Math.random()]); new Uint32Array(b, 12, 1)[0] = greedy ? 1 : 0;
      dev.queue.writeBuffer(this.sp, 0, b);
    }
    const enc = dev.createCommandEncoder(), pass = enc.beginComputePass();
    const run = ([p, bg, g]) => { pass.setPipeline(p); pass.setBindGroup(0, bg); const gg = typeof g === "function" ? g() : [g, 1]; pass.dispatchWorkgroups(gg[0], gg[1]); };
    for (const op of this.ops) if (!this.skip?.test(op[0].label)) run(op);   // skip: ablation timing only (the output is wrong)
    if (sample) this.sampleOps.forEach(run);
    pass.end();
    if (readLogits) enc.copyBufferToBuffer(this.logits, 0, this.staging, 0, VOCAB * 4);
    const stage = sample ? this.stagePool.pop() || dev.createBuffer({ size: 16, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST }) : null;
    if (sample) enc.copyBufferToBuffer(this.tok, 0, stage, 0, 16);
    dev.queue.submit([enc.finish()]);
    const out = {};
    if (readLogits) { await this.staging.mapAsync(GPUMapMode.READ); out.logits = new Float32Array(this.staging.getMappedRange().slice(0)); this.staging.unmap(); }
    if (sample) { await stage.mapAsync(GPUMapMode.READ); out.token = new Uint32Array(stage.getMappedRange())[0]; stage.unmap(); this.stagePool.push(stage); }
    return out;
  }

  /** The engram table on the GPU: two buffers (n-gram order 2, 3) of `pages` pages x 4032 B (56 rows: 64 B codes each, then 56
   *  fp16 scale pairs, then 56 fp16 bias pairs). chunk(k, page0, n) supplies n pages of order k. */
  async #loadEngramTable(pages, chunk, onStatus, chunkPages = 16384, parallel = 4) {
    const b = [0, 1].map(() => this.device.createBuffer({ size: pages * 4032, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST }));
    const jobs = []; for (let k = 0; k < 2; k++) for (let p = 0; p < pages; p += chunkPages) jobs.push([k, p, Math.min(chunkPages, pages - p)]);
    let next = 0, done = 0;
    const worker = async () => { while (next < jobs.length) { const [k, p, n] = jobs[next++]; const u8 = await chunk(k, p, n);
      if (u8.byteLength !== n * 4032) throw new Error(`engram table chunk ${k}/${p}: ${u8.byteLength} B, expected ${n * 4032}`);
      this.device.queue.writeBuffer(b[k], p * 4032, u8); onStatus(`Uploading the engram table to the GPU ${++done}/${jobs.length}…`); } };
    try { await Promise.all(Array.from({ length: parallel }, worker)); } catch (e) { b.forEach(x => x.destroy()); throw e; }
    this.etab = { pages, b };
  }
  /** Rebuild the dispatch list (after changing TUNE). */
  rerecord() { this.#record(); }
  /** Seed the GPU token ring with the tokens before position pos (all: every token fed so far). */
  setTokens(all, pos) {
    const r = new Uint32Array(4); for (let p = Math.max(0, pos - 3); p < pos; p++) r[p % 4] = all[p];
    this.device.queue.writeBuffer(this.th, 0, r);
  }
  /** Engram rows for `tokens` at positions pos0, pos0+1, ... (history: every token before pos0) -> Float32Array [T*16*128], computed
   *  from the GPU-resident table: prefill's input to the ORT graph, with no HTTP requests. One small submit a token (queue order
   *  carries each token's position through the shared uniform). */
  async engramRows(tokens, pos0, history) {
    const dev = this.device, T = tokens.length, out = dev.createBuffer({ size: T * D * 4, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
    const op = this.ops.find(([p]) => p.label === "engramGather");
    this.setTokens(history, pos0);
    for (let t = 0; t < T; t++) {
      dev.queue.writeBuffer(this.info, 0, new Uint32Array([pos0 + t, 0, 0, 0])); dev.queue.writeBuffer(this.tok, 0, new Uint32Array([tokens[t], 0, 0, 0]));
      const enc = dev.createCommandEncoder(), pass = enc.beginComputePass();
      pass.setPipeline(op[0]); pass.setBindGroup(0, op[1]); pass.dispatchWorkgroups(1); pass.end();
      enc.copyBufferToBuffer(this.eIn, 0, out, t * D * 4, D * 4); dev.queue.submit([enc.finish()]);
    }
    await out.mapAsync(GPUMapMode.READ); const r = new Float32Array(out.getMappedRange().slice(0)); out.unmap(); out.destroy();
    return r;
  }
  /** Test: the engram rows the GPU computes for `token` at pos (after setTokens). Clobbers the ring slot of pos. */
  async testEngram(token, pos) {
    const dev = this.device; dev.queue.writeBuffer(this.info, 0, new Uint32Array([pos, 0, 0, 0])); dev.queue.writeBuffer(this.tok, 0, new Uint32Array([token, 0, 0, 0]));
    const op = this.ops.find(([p]) => p.label === "engramGather"), enc = dev.createCommandEncoder(), pass = enc.beginComputePass();
    pass.setPipeline(op[0]); pass.setBindGroup(0, op[1]); pass.dispatchWorkgroups(1); pass.end(); dev.queue.submit([enc.finish()]);
    return this.read(this.eIn, D * 4);
  }

  /** GPU time per kernel type (timestamp queries; every dispatch in its own pass, so the sum is a bit above a real step). */
  async profileKernels(pos, steps = 5) {
    const dev = this.device, n = this.ops.length;
    const qs = dev.createQuerySet({ type: "timestamp", count: 2 * n });
    const res = dev.createBuffer({ size: 16 * n, usage: GPUBufferUsage.QUERY_RESOLVE | GPUBufferUsage.COPY_SRC });
    const rd = dev.createBuffer({ size: 16 * n, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
    const tot = new Map();
    for (let s = 0; s < steps; s++) {
      this.pos = pos + s; dev.queue.writeBuffer(this.info, 0, new Uint32Array([pos + s, 0, 0, 0]));
      const enc = dev.createCommandEncoder();
      this.ops.forEach(([p, bg, g], k) => {
        const pass = enc.beginComputePass({ timestampWrites: { querySet: qs, beginningOfPassWriteIndex: 2 * k, endOfPassWriteIndex: 2 * k + 1 } });
        pass.setPipeline(p); pass.setBindGroup(0, bg); const gg = typeof g === "function" ? g() : [g, 1]; pass.dispatchWorkgroups(gg[0], gg[1]); pass.end();
      });
      enc.resolveQuerySet(qs, 0, 2 * n, res, 0); enc.copyBufferToBuffer(res, 0, rd, 0, 16 * n); dev.queue.submit([enc.finish()]);
      await rd.mapAsync(GPUMapMode.READ); const t = new BigUint64Array(rd.getMappedRange().slice(0)); rd.unmap();
      this.ops.forEach(([p], k) => { const key = (p.label || "?").replace(/_\d+$/, ""); const ns = Number(t[2 * k + 1] - t[2 * k]); const a = tot.get(key) || [0, 0]; a[0] += ns; a[1]++; tot.set(key, a); });
    }
    qs.destroy(); res.destroy(); rd.destroy();
    const rows = [...tot].map(([k, [ns, c]]) => [k, ns / steps / 1e6, c / steps]).sort((a, b) => b[1] - a[1]);
    return { totalMs: rows.reduce((a, r) => a + r[1], 0), rows };
  }

  /** Profiling: encode-only time, and GPU time from back-to-back steps without readback (state advances; reset after). */
  async profile(pos, n = 40) {
    const dev = this.device;
    let enc = 0;
    for (let k = 0; k < 5; k++) await this.step(0, pos + k);                       // warm
    await dev.queue.onSubmittedWorkDone();
    const t0 = performance.now();
    for (let k = 0; k < n; k++) { const a = performance.now(); await this.step(0, pos + 5 + k, { readLogits: false }); enc += performance.now() - a; }
    const t1 = performance.now(); await dev.queue.onSubmittedWorkDone(); const t2 = performance.now();
    const full = []; for (let k = 0; k < 10; k++) { const a = performance.now(); await this.step(0, pos + 5 + n + k); full.push(performance.now() - a); }
    return { encodeMs: enc / n, gpuMsPerStep: (t2 - t0) / n, stepWithReadbackMs: full.reduce((a, b) => a + b) / full.length, dispatches: this.ops.length };
  }

  async read(buf, bytes, offset = 0) {
    const s = this.device.createBuffer({ size: bytes, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
    const enc = this.device.createCommandEncoder(); enc.copyBufferToBuffer(buf, offset, s, 0, bytes); this.device.queue.submit([enc.finish()]);
    await s.mapAsync(GPUMapMode.READ); const out = new Float32Array(s.getMappedRange().slice(0)); s.unmap(); s.destroy(); return out;
  }

  /** Take over the ORT cache tensors (name_in -> tensor) at position pos; tokens: every token fed so far (for the GPU engram). */
  importOrt(kv, pos, tokens) {
    if (tokens) this.setTokens(tokens, pos);                            // the GPU engram hashes the tokens before pos
    const enc = this.device.createCommandEncoder(), B = 4;
    for (let i = 0; i < this.L; i++) {
      const t = this.types[i], C = this.cache[i];
      if (t === "sliding_attention") {
        const L = Math.min(pos, WINDOW - 1), row = 4 * HD * B;
        for (const [nm, dst] of [[`sw${i}_k_in`, C.k], [`sw${i}_v_in`, C.v]]) {
          const src = kv[nm]; if (!L || !src?.gpuBuffer) continue;
          for (let r = 0; r < L; r++) enc.copyBufferToBuffer(src.gpuBuffer, r * row, dst, ((pos - L + r) % WINDOW) * row, row);
        }
      } else if (t === "full_attention") {
        for (const [nm, dst] of [[`fl${i}_k_in`, C.k], [`fl${i}_kn_in`, C.kn], [`fl${i}_v_in`, C.v]]) {
          const src = kv[nm]; if (pos && src?.gpuBuffer) enc.copyBufferToBuffer(src.gpuBuffer, 0, dst, 0, pos * 2 * HD * B);
        }
      }
    }
    const conv = kv.conv_in;                                            // [9][D]: positions pos-9 .. pos-1
    if (conv?.gpuBuffer) for (let r = 0; r < 9; r++) { const p = pos - 9 + r; if (p >= 0) enc.copyBufferToBuffer(conv.gpuBuffer, r * D * B, this.hist, (p % HRING) * D * B, D * B); }
    else if (conv) this.device.queue.writeBuffer(this.hist, 0, new Float32Array(HRING * D));
    this.device.queue.submit([enc.finish()]);
  }

  /** The state as ORT cache tensors (name_in -> tensor) at position pos (new GPU buffers, freed when the tensor is disposed). */
  exportOrt(pos) {
    const enc = this.device.createCommandEncoder(), B = 4, out = {};
    const mk = (bytes, dims) => { const b = this.#buf(bytes); return [b, ort.Tensor.fromGpuBuffer(b, { dataType: "float32", dims, dispose: () => b.destroy() })]; };
    for (let i = 0; i < this.L; i++) {
      const t = this.types[i], C = this.cache[i];
      if (t === "sliding_attention") {
        const L = Math.min(pos, WINDOW - 1), row = 4 * HD * B;
        for (const [nm, src] of [[`sw${i}_k_in`, C.k], [`sw${i}_v_in`, C.v]]) {
          if (!L) { out[nm] = new ort.Tensor("float32", new Float32Array(0), [1, 0, 4, HD]); continue; }
          const [b, tt] = mk(L * row, [1, L, 4, HD]);
          for (let r = 0; r < L; r++) enc.copyBufferToBuffer(src, ((pos - L + r) % WINDOW) * row, b, r * row, row);
          out[nm] = tt;
        }
      } else if (t === "full_attention") {
        for (const [nm, src] of [[`fl${i}_k_in`, C.k], [`fl${i}_kn_in`, C.kn], [`fl${i}_v_in`, C.v]]) {
          if (!pos) { out[nm] = new ort.Tensor("float32", new Float32Array(0), [1, 0, 2, HD]); continue; }
          const [b, tt] = mk(pos * 2 * HD * B, [1, pos, 2, HD]); enc.copyBufferToBuffer(src, 0, b, 0, pos * 2 * HD * B); out[nm] = tt;
        }
      }
    }
    const [cb, ct] = mk(9 * D * B, [1, 9, D]);
    const zero = new Float32Array(D);
    for (let r = 0; r < 9; r++) { const p = pos - 9 + r; if (p >= 0) enc.copyBufferToBuffer(this.hist, (p % HRING) * D * B, cb, r * D * B, D * B); else this.device.queue.writeBuffer(cb, r * D * B, zero); }
    out.conv_in = ct;
    this.device.queue.submit([enc.finish()]);
    return out;
  }
}
