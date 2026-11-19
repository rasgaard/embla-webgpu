// onnxruntime-web 1.26.0 (WebGPU build) from jsDelivr, so the repo carries no 50 MB of wasm. Everything imports ORT from here.
const DIST = "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.26.0/dist/";
import * as ort from "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.26.0/dist/ort.webgpu.bundle.min.mjs";
ort.env.wasm.wasmPaths = DIST;
export * from "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.26.0/dist/ort.webgpu.bundle.min.mjs";
