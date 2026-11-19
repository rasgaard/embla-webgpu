// Persistent weight cache on top of the Cache API (works in Chromium, Firefox and Safari; survives reloads).
// One cache store; every entry key carries the content tag (size + ETag/Last-Modified) of the file it came from, so a
// re-published graph only invalidates the graph entries (the 3.5 GB of experts stay), and prune() drops entries a load
// no longer used. Older stores (one per whole-model signature) are deleted on open.
const PREFIX = "embla1-";
const NAME = PREFIX + "v2";
const CHUNK = 64 * 1024 * 1024;

const hash = (s) => { let h = 2166136261; for (let i = 0; i < s.length; i++) { h ^= s.charCodeAt(i); h = Math.imul(h, 16777619); } return (h >>> 0).toString(36); };

/** Cheap identity of a remote file without downloading it. */
export async function remoteSignature(url) {
  const r = await fetch(url, { method: "HEAD" });
  if (!r.ok) throw new Error(`HEAD ${url} -> ${r.status}`);
  return `${url}|${r.headers.get("content-length")}|${r.headers.get("etag") || r.headers.get("last-modified") || ""}`;
}

/** Content identity of a remoteSignature() without the URL (hard-linked copies of one file in two folders share it). */
export const contentTag = (sig) => hash(sig.split("|").slice(1).join("|"));

export class WeightCache {
  static async open() {
    if (!("caches" in globalThis)) return new WeightCache(null, "unavailable");
    for (const n of await caches.keys()) if (n.startsWith(PREFIX) && n !== NAME) await caches.delete(n);   // pre-v2 stores
    return new WeightCache(await caches.open(NAME), NAME);
  }
  /** Delete every entry whose key is not in `keep` (keys as passed to get/put). */
  async prune(keep) {
    if (!this.c) return 0; let n = 0;
    for (const req of await this.c.keys()) {
      const key = decodeURIComponent(req.url.split("/").pop()).replace(/#\d+$/, "");
      if (!keep.has(key)) { await this.c.delete(req); n++; }
    }
    return n;
  }
  constructor(cache, name) { this.c = cache; this.name = name; this.writeError = null; this.disabled = false; }
  get available() { return !!this.c; }
  #req(key) { return new Request(`${location.origin}/__embla_weight_cache__/${this.name}/${encodeURIComponent(key)}`); }

  // Chromium's Cache API rejects single entries of a few hundred MB ("UnknownError"), so big values are stored as
  // 64 MB chunks plus a small manifest entry that carries the chunk count.
  async get(key) {
    if (!this.c) return null;
    const r = await this.c.match(this.#req(key));
    if (!r) return null;
    const n = Number(r.headers.get("x-embla-chunks") || 0);
    if (!n) return new Uint8Array(await r.arrayBuffer());
    const out = new Uint8Array(Number(r.headers.get("x-embla-size"))); let off = 0;
    for (let i = 0; i < n; i++) {
      const c = await this.c.match(this.#req(`${key}#${i}`));
      if (!c) { await this.#drop(key, n); return null; }                      // partial entry (e.g. evicted): treat as a miss
      const u8 = new Uint8Array(await c.arrayBuffer()); out.set(u8, off); off += u8.byteLength;
    }
    return off === out.byteLength ? out : null;
  }
  async #drop(key, n) { await this.c.delete(this.#req(key)); for (let i = 0; i < n; i++) await this.c.delete(this.#req(`${key}#${i}`)); }

  /** Returns false (and remembers why) if the browser refuses, e.g. quota exceeded: that entry is simply not cached. */
  async put(key, u8) {
    if (!this.c || this.disabled) return false;
    const n = u8.byteLength > CHUNK ? Math.ceil(u8.byteLength / CHUNK) : 0;
    try {
      if (!n) { await this.c.put(this.#req(key), new Response(u8)); return true; }
      for (let i = 0; i < n; i++) await this.c.put(this.#req(`${key}#${i}`), new Response(u8.subarray(i * CHUNK, (i + 1) * CHUNK)));
      await this.c.put(this.#req(key), new Response(null, { headers: { "x-embla-chunks": String(n), "x-embla-size": String(u8.byteLength) } }));   // manifest last: a crash mid-write leaves no manifest
      return true;
    } catch (e) {
      this.writeError = e; if (e?.name === "QuotaExceededError") this.disabled = true;          // out of space: stop trying
      try { await this.#drop(key, n); } catch {}
      return false;
    }
  }
  async count() { return this.c ? (await this.c.keys()).length : 0; }
  async clear() { if (this.c) { await caches.delete(this.name); this.c = await caches.open(this.name); this.writeError = null; this.disabled = false; } }

  static async clearAll() { for (const n of await caches.keys()) if (n.startsWith(PREFIX)) await caches.delete(n); }
  static async usage() {
    const e = (await navigator.storage?.estimate?.()) || {};
    return { used: e.usage ?? NaN, quota: e.quota ?? NaN, persisted: (await navigator.storage?.persisted?.()) ?? false };
  }
  /** Ask the browser not to evict the weights under storage pressure (must be called from a user gesture in some browsers). */
  static async requestPersistence() { try { return (await navigator.storage?.persist?.()) ?? false; } catch { return false; } }
}

/** fetch a whole file with progress; returns Uint8Array. */
export async function fetchBytes(url, onBytes) {
  const r = await fetch(url); if (!r.ok) throw new Error(`${url} -> ${r.status}`);
  const total = Number(r.headers.get("content-length")) || 0;
  if (!r.body || !total) { const u8 = new Uint8Array(await r.arrayBuffer()); onBytes?.(u8.byteLength); return u8; }
  const out = new Uint8Array(total); let n = 0; const reader = r.body.getReader();
  for (;;) { const { done, value } = await reader.read(); if (done) break; out.set(value, n); n += value.byteLength; onBytes?.(value.byteLength); }
  return out;
}
