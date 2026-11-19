// Where the model files come from.
//   weights: sevren-ai/Embla1-4bit (the MLX 4-bit checkpoint, engram table, tokenizer, config), pinned to the revision this was built on
//   graph:   the ONNX step graph tools/build_graph.sh produces (prefill runs on it), on the Hugging Face Hub
// On a local server with models/ populated (tools/download.py, tools/build_graph.sh) the local copies are used instead.
// Override with ?source=hub or ?source=local.
export const WEIGHTS = { repo: "sevren-ai/Embla1-4bit", revision: "4d23eb7de21fc5b55c3051021c7550af1fdce0e8" };
export const GRAPH = { repo: "rasgaard/Embla1-4bit-webgpu", revision: "d9139298e9cf71a85e6888b703d6c30c89f19f70" };

const hub = ({ repo, revision }) => `https://huggingface.co/${repo}/resolve/${revision}/`;

/** -> {weightsUrl, graphUrl, tokenizer: {model, revision, localPath?}, source} */
export async function modelSources(params = new URLSearchParams(location.search)) {
  let source = params.get("source");
  if (!source) {
    try { source = (await fetch("models/graph/embla_step.onnx", { method: "HEAD" })).ok ? "local" : "hub"; } catch { source = "hub"; }
  }
  if (source === "local") return { source, weightsUrl: new URL("models/Embla1-4bit/", location.href).href, graphUrl: new URL("models/graph/", location.href).href,
    tokenizer: { model: "Embla1-4bit", localPath: new URL("models/", location.href).href } };
  return { source, weightsUrl: hub(WEIGHTS), graphUrl: hub(GRAPH), tokenizer: { model: WEIGHTS.repo, revision: WEIGHTS.revision } };
}
