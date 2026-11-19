"""Replace the gathered-expert MoE with two 4-bit MatMulNBits over ALL experts (graph surgery on a patch_syncs.py output).

Before, per MoE layer and token: GatherBlockQuantized dequantizes the 8 chosen experts to fp32 ([8, 512, 2048] = 32 MB), adds the
affine offset (another 64 MB read+write), then a generic Einsum reads it again; twice (fc, proj). ~8 GB of fp32 traffic per token,
~60% of decode time, and prefill cost grows with T x 8 expert slices.

After (exact, only the summation order changes):
    h   = relu(x @ Wfc_all^T + groupsum(x) @ Cfc^T)^2           Wfc_all = all 96 experts stacked: [96*512, 2048], 4-bit
    hg  = h * gate_dense                                         gate_dense[n, e] = routing weight if e chosen else 0
    y   = hg @ Wproj_all^T + groupsum(hg) @ Cproj^T             Wproj_all = experts concatenated along K: [2048, 96*512]
4-bit reads of all experts (~100 MB/layer) instead of fp32 materialisation, with ORT's tuned MatMulNBits kernel, and the cost no
longer grows with the chunk length (prefill).
New expert inputs: e<i>_fcN_* (all experts' fc rows stacked, a plain reshape of MLX's bytes) and e<i>_projT_* ([out, expert*in],
which the browser loader, src/experts.mjs, builds by transposing MLX's [expert, out, in] while uploading).

usage: uv run python tools/patch_moe_dense.py in/embla_step.onnx out/embla_step.onnx
"""
import collections, os, re, sys
import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

E, F, D, G = 96, 512, 2048, 64                     # experts, moe_intermediate_size, hidden, quant group
src, dst = sys.argv[1], sys.argv[2]
m = onnx.load(src, load_external_data=False)
g = m.graph
prod = {o: n for n in g.node for o in n.output}
cons = collections.defaultdict(list)
for n in g.node:
    for i in n.input: cons[i].append(n)
init = {t.name: t for t in g.initializer}
uid = iter(range(10**6))
nm = lambda base: f"{base}__dense{next(uid)}"
def const(name, arr):
    if name not in init:
        t = numpy_helper.from_array(arr, name); g.initializer.append(t); init[name] = t
    return name
I64 = lambda *v: np.array(v, np.int64)

new_nodes = collections.defaultdict(list)          # insert before node name
rewire = {}                                         # old tensor -> new tensor
new_inputs, drop_inputs = [], set()
layers = 0
for e5 in [n for n in g.node if n.op_type == "Einsum" and n.attribute[0].s == b"nkf,nkdf->nkd"]:
    (mul,) = cons[e5.output[0]]; (rs,) = cons[mul.output[0]]
    gates = prod[[x for x in mul.input if x != e5.output[0]][0]].input[0]           # Unsqueeze(div_70) -> div_70 [N, 8]
    relu = prod[prod[e5.input[0]].input[0]]; e4 = prod[relu.input[0]]; x2 = e4.input[0]
    gbq = prod[prod[prod[prod[e4.input[1]].input[0]].input[0]].input[0]]              # Reshape <- Add <- Reshape <- GBQ
    assert gbq.op_type == "GatherBlockQuantized", gbq.op_type
    li = re.match(r"e(\d+)_fc_q4$", gbq.input[0]).group(1)
    topk_idx = prod[prod[gbq.input[1]].input[0]].input[1]                               # Reshape <- Gather(arange, idx)
    assert prod[topk_idx].op_type == "TopK"
    # new expert inputs, uploaded by the loader in the shapes MatMulNBits wants (WebGPU can't Reshape uint8 in-graph: it would
    # copy every expert tensor to the CPU). fcN = all experts' fc rows stacked; projT = proj reordered to [out, expert*in].
    kf, kp, ST = "fcN", "projT", TensorProto.FLOAT
    fq, fs, fcc = f"e{li}_{kf}_q4", f"e{li}_{kf}_sc", f"e{li}_{kf}_c"
    pq, ps, pc = f"e{li}_{kp}_q4", f"e{li}_{kp}_sc", f"e{li}_{kp}_c"
    drop_inputs.update({f"e{li}_{w}_{p}" for w in ("fc", "proj") for p in ("q4", "sc", "c")})
    new_inputs += [helper.make_tensor_value_info(fq, TensorProto.UINT8, [E * F, D // G, G // 2]),
                   helper.make_tensor_value_info(fs, ST, [E * F, D // G]),
                   helper.make_tensor_value_info(fcc, ST, [E * F, D // G]),
                   helper.make_tensor_value_info(pq, TensorProto.UINT8, [D, E * F // G, G // 2]),
                   helper.make_tensor_value_info(ps, ST, [D, E * F // G]),
                   helper.make_tensor_value_info(pc, ST, [D, E * F // G])]
    t = {k: nm(k) for k in "idxf idx3 eq oh g3 wg gd h0 x3 gsx off1 h r a a3 gd3 ag ag2 y0 ag3 gsa off2 y".split()}
    ar = const("dense_arange_experts", np.arange(E, dtype=np.float32))
    N = lambda op, ins, out, **kw: helper.make_node(op, ins, [t[out]], name=nm(op), **kw)
    new_nodes[rs.name] += [
        # dense routing weights [N, E] from top-k indices + normalised gates (one-hot in float: no int64 ops on the GPU)
        N("Gather", [ar, topk_idx], "idxf", axis=0),
        N("Unsqueeze", [t["idxf"], const("dense_ax_last", I64(-1))], "idx3"),
        N("Equal", [t["idx3"], ar], "eq"),
        N("Cast", [t["eq"]], "oh", to=TensorProto.FLOAT),
        N("Unsqueeze", [gates, const("dense_ax_last", I64(-1))], "g3"),
        N("Mul", [t["oh"], t["g3"]], "wg"),
        N("ReduceSum", [t["wg"], const("dense_ax1", I64(1))], "gd", keepdims=0),
    ]
    mm = lambda a, b, c, out, K_, N_: helper.make_node("MatMulNBits", [a, b, c], [t[out]], domain="com.microsoft", name=nm("mmnb"), K=K_, N=N_, bits=4, block_size=G)
    new_nodes[rs.name] += [
        N("Reshape", [x2, const("dense_shape_gx", I64(-1, D // G, G))], "x3"),
        N("ReduceSum", [t["x3"], const("dense_ax_last", I64(-1))], "gsx", keepdims=0),
    ]
    new_nodes[rs.name] += [mm(x2, fq, fs, "h0", D, E * F), N("Gemm", [t["gsx"], fcc], "off1", transB=1), N("Add", [t["h0"], t["off1"]], "h")]
    new_nodes[rs.name] += [
        N("Relu", [t["h"]], "r"),
        N("Mul", [t["r"], t["r"]], "a"),
        # gate (zero for unchosen experts), then proj over all experts (concatenated along K)
        N("Reshape", [t["a"], const("dense_shape_ef", I64(-1, E, F))], "a3"),
        N("Unsqueeze", [t["gd"], const("dense_ax_last", I64(-1))], "gd3"),
        N("Mul", [t["a3"], t["gd3"]], "ag"),
        N("Reshape", [t["ag"], const("dense_shape_k", I64(-1, E * F))], "ag2"),
    ]
    new_nodes[rs.name] += [                        # proj over all experts (concatenated along K)
        mm(t["ag2"], pq, ps, "y0", E * F, D),
        N("Reshape", [t["ag2"], const("dense_shape_gk", I64(-1, E * F // G, G))], "ag3"),
        N("ReduceSum", [t["ag3"], const("dense_ax_last", I64(-1))], "gsa", keepdims=0),
        N("Gemm", [t["gsa"], pc], "off2", transB=1),
        N("Add", [t["y0"], t["off2"]], "y")]
    rewire[rs.output[0]] = t["y"]
    layers += 1

for n in g.node:
    for k, x in enumerate(n.input):
        if x in rewire: n.input[k] = rewire[x]
nodes = []
for n in g.node:
    nodes += new_nodes.get(n.name, []); nodes.append(n)
outs = {o.name for o in g.output}
while True:                                         # drop the old gathered-expert path
    used = {i for n in nodes for i in n.input} | outs
    live = [n for n in nodes if any(o in used for o in n.output)]
    if len(live) == len(nodes): break
    nodes = live
del g.node[:]; g.node.extend(nodes)
used = {i for n in nodes for i in n.input}
assert not (drop_inputs & used)
keep = [i for i in g.input if i.name not in drop_inputs]
del g.input[:]; g.input.extend(keep + new_inputs)
dead_init = [t for t in g.initializer if t.name not in used]
for t in dead_init: g.initializer.remove(t)
if not any(o.domain == "com.microsoft" for o in m.opset_import): m.opset_import.append(helper.make_opsetid("com.microsoft", 1))

os.makedirs(os.path.dirname(dst) or ".", exist_ok=True)
onnx.save(m, dst)
data_dst = os.path.join(os.path.dirname(dst), "embla_step.onnx.data")
if not os.path.exists(data_dst): os.link(os.path.join(os.path.dirname(src), "embla_step.onnx.data"), data_dst)
print(f"dense MoE layers: {layers}, nodes {len(nodes)}, unused initializers dropped {len(dead_init)} -> {dst}")
