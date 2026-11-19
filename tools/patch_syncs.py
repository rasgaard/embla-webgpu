"""Remove the GPU->CPU syncs from the exported step graph (no re-export, graph surgery only).

ORT-web's WebGPU EP has no int64 kernels for Reshape/Where/Concat and none for IsInf/Sign, so those nodes run on the CPU and
ORT inserts a MemcpyToHost in front of them. Each one drains the GPU queue mid-run: 40 of them per step (one per MoE layer for
the routing indices, two per sparse-attention top-k, one Sign). This rewrites them into int32 / float ops the EP has:
  * TopK indices -> int32 via Gather(arange_int32, idx) (WebGPU can't Cast int64, but Gather takes int64 indices) before the Reshape (MoE) and Where (sparse attention) that consume them;
  * IsInf(x) -> Less(x, -1e30)  (top-k scores are finite or -inf padding);
  * the Where's int64 fill constant -> int32, and the Concat partner `expand` -> Cast(int32) (Concat then feeds a Cast to int32 anyway);
  * Sign(x) -> float(x > 0) - float(x < 0).
External weights are untouched: the new .onnx points at the same .data file (hard-linked next to it).

usage: uv run python tools/patch_syncs.py in/embla_step.onnx out/embla_step.onnx
"""
import collections, os, sys
import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

src, dst = sys.argv[1], sys.argv[2]
m = onnx.load(src, load_external_data=False)
g = m.graph
prod = {o: n for n in g.node for o in n.output}
cons = collections.defaultdict(list)
for n in g.node:
    for i in n.input: cons[i].append(n)
init = {t.name: t for t in g.initializer}
touched = set()                                   # tensors whose dtype changes: drop their stale value_info
new_nodes = collections.defaultdict(list)         # node name -> nodes to insert before it
uid = iter(range(10**6))
nm = lambda base: f"{base}__nosync{next(uid)}"

def const(name, arr):
    if name not in init:
        t = numpy_helper.from_array(arr, name); g.initializer.append(t); init[name] = t
    return name

def cast_before(node, inp):                       # plain Cast (for CPU-side int64 tensors: ORT uploads the int32 result)
    out = nm(inp + "_i32")
    new_nodes[node.name].append(helper.make_node("Cast", [inp], [out], to=TensorProto.INT32, name=nm("cast")))
    return out

def gpu_i32(node, inp, n):                        # int64 GPU tensor -> int32 on the GPU. WebGPU has no int64 Cast/Reshape/Where,
    out = nm(inp + "_i32")                        # but Gather takes int64 *indices*: arange_i32[idx] == idx.
    table = const(f"nosync_arange{n}", np.arange(n, dtype=np.int32))
    new_nodes[node.name].append(helper.make_node("Gather", [table, inp], [out], axis=0, name=nm("i64to32")))
    return out

N_EXPERTS, MAX_KEYS = 96, 8192 + 64                # expert ids; key positions (full-layer cache dims go to 8192, + chunk)
stats = collections.Counter()
for n in list(g.node):
    if n.op_type != "TopK": continue
    idx = n.output[1]
    for c in cons[idx]:
        if c.op_type == "Reshape":                                          # MoE routing indices -> GatherBlockQuantized / Gather
            c.input[0] = gpu_i32(c, idx, N_EXPERTS); touched.update(c.output); stats["moe reshape"] += 1
            for c2 in cons[c.output[0]]:
                if c2.op_type == "Reshape": touched.update(c2.output)
        elif c.op_type == "Where":                                          # sparse attention: where(isinf(score), -1, idx)
            cond, fill, _ = c.input
            isinf = prod[cond]; assert isinf.op_type == "IsInf", isinf.op_type
            isinf.op_type = "Less"; del isinf.attribute[:]
            isinf.input.append(const("nosync_neg_big", np.array(-1e30, np.float32)))
            assert init[fill].data_type == TensorProto.INT64
            c.input[1] = const(fill + "_i32", numpy_helper.to_array(init[fill]).astype(np.int32))
            c.input[2] = gpu_i32(c, idx, MAX_KEYS); touched.update(c.output)
            for cat in cons[c.output[0]]:
                assert cat.op_type == "Concat", cat.op_type
                for k, x in enumerate(cat.input):
                    if x != c.output[0]: cat.input[k] = cast_before(cat, x)
                touched.update(cat.output)
                for nxt in cons[cat.output[0]]: assert nxt.op_type == "Cast" and nxt.attribute[0].i == TensorProto.INT32, nxt
            stats["sparse where"] += 1
        else:
            assert c.op_type == "GatherElements", c.op_type                  # stays on the GPU as is

# Sparse attention, second half: idx (int32, [B,Hkv,Tq,K]) -> valid = idx >= 0, ii = clamp_min(idx, 0) -> int64 -> GatherND(k, (b, g, ii)).
# WebGPU has no int32 GreaterOrEqual/Clip and GatherND only takes int64 indices, so do it in float (exact below 2^24) and
# replace each GatherND by a flat Gather:  k[0, g, ii] == flatten(k)[g*Tk + ii]   (B == 1).
for c68 in [n for n in g.node if n.op_type == "Cast" and n.input[0] in touched and n.attribute[0].i == TensorProto.INT32]:
    users = list(cons[c68.output[0]])
    f = nm(c68.output[0] + "_f")
    new_nodes[users[0].name].append(helper.make_node("Cast", [c68.input[0]], [f], to=TensorProto.FLOAT, name=nm("idxf")))
    zf = const("nosync_zero", np.array(0, np.float32))
    for u in users:
        if u.op_type == "GreaterOrEqual": u.input[:] = [f, zf]; stats["valid mask"] += 1; continue
        assert u.op_type == "Clip", u.op_type
        u.input[:] = [f, zf]; clampf = u.output[0]; touched.add(clampf)
        (c71,) = cons[clampf]; assert c71.op_type == "Cast" and c71.attribute[0].i == TensorProto.INT64
        (unsq,) = cons[c71.output[0]]; assert unsq.op_type == "Unsqueeze"
        for cat in cons[unsq.output[0]]:
            assert cat.op_type == "Concat" and len(cat.input) == 3 and cat.input[2] == unsq.output[0]
            grid_b, grid_g = init[cat.input[0]], init[cat.input[1]]
            assert grid_b.dims[0] == 1, "flat-gather rewrite assumes batch 1"
            hkv = grid_g.dims[1]
            for gnd in cons[cat.output[0]]:
                assert gnd.op_type == "GatherND" and gnd.input[1] == cat.output[0]
                data = gnd.input[0]
                shp, tk, tkf, off, flatf, flati, flat = (nm(x) for x in ("shp", "tk", "tkf", "off", "flatf", "flati", "flat"))
                gridf = const(f"nosync_grid{hkv}", np.arange(hkv, dtype=np.float32).reshape(1, hkv, 1, 1))
                new_nodes[gnd.name] += [
                    helper.make_node("Shape", [data], [shp], start=2, end=3, name=nm("tk")),                       # Tk, on the CPU (no sync: shape only)
                    helper.make_node("Cast", [shp], [tkf], to=TensorProto.FLOAT, name=nm("tk")),
                    helper.make_node("Mul", [gridf, tkf], [off], name=nm("tk")),                               # [1,Hkv,1,1] = g*Tk
                    helper.make_node("Add", [clampf, off], [flatf], name=nm("flat")),
                    helper.make_node("Cast", [flatf], [flati], to=TensorProto.INT32, name=nm("flat")),
                    helper.make_node("Flatten", [data], [flat], axis=3, name=nm("flat")),                      # [B*Hkv*Tk, D]
                ]
                gnd.op_type = "Gather"; gnd.input[:] = [flat, flati]; del gnd.attribute[:]; gnd.attribute.append(helper.make_attribute("axis", 0))
                stats["gathernd -> gather"] += 1

for n in list(g.node):
    if n.op_type != "Sign": continue
    x, y = n.input[0], n.output[0]
    zero = const("nosync_zero", np.array(0, np.float32))
    gt, lt, gtf, ltf = nm("gt"), nm("lt"), nm("gtf"), nm("ltf")
    new_nodes[n.name] += [helper.make_node("Greater", [x, zero], [gt], name=nm("sign")), helper.make_node("Less", [x, zero], [lt], name=nm("sign")),
                          helper.make_node("Cast", [gt], [gtf], to=TensorProto.FLOAT, name=nm("sign")), helper.make_node("Cast", [lt], [ltf], to=TensorProto.FLOAT, name=nm("sign"))]
    n.op_type = "Sub"; n.input[:] = [gtf, ltf]; stats["sign"] += 1

nodes = []
for n in g.node:
    nodes += new_nodes.get(n.name, []); nodes.append(n)
outs = {o.name for o in g.output}                 # drop nodes left dead by the rewrites (they would still run, on the CPU)
while True:
    used = {i for n in nodes for i in n.input} | outs
    live = [n for n in nodes if any(o in used for o in n.output)]
    if len(live) == len(nodes): break
    stats["dead nodes removed"] += len(nodes) - len(live); nodes = live
del g.node[:]; g.node.extend(nodes)
keep = [v for v in g.value_info if v.name not in touched]
del g.value_info[:]; g.value_info.extend(keep)

os.makedirs(os.path.dirname(dst) or ".", exist_ok=True)
onnx.save(m, dst)
data_src = os.path.join(os.path.dirname(src), "embla_step.onnx.data"); data_dst = os.path.join(os.path.dirname(dst), "embla_step.onnx.data")
if not os.path.exists(data_dst): os.link(data_src, data_dst)
print(dict(stats), "nodes", len(g.node), "->", dst)
