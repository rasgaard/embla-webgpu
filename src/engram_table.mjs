// Reader for engram_table.safetensors (23.5M rows x 128, MLX affine 4-bit, group 64, fp16 scales / biases) over HTTP Range requests.
//   gather(): the rows of a prompt (prefill on ORT), fetched by 56-row page and cached for the session;
//   pages():  whole pages in the GPU layout the WGSL decoder holds (codes, then scales, then biases, 4032 B a page).
const f16 = (h) => { const s = h & 0x8000 ? -1 : 1, e = (h >> 10) & 31, m = h & 1023;
  return e === 0 ? s * m * 2 ** -24 : e === 31 ? (m ? NaN : s * Infinity) : s * (1 + m / 1024) * 2 ** (e - 15); };
export const PAGE_ROWS = 56, PAGE_BYTES = PAGE_ROWS * 72;

export class EngramTable {
  constructor(url, hdr, base) { this.url = url; this.t = hdr; this.base = base; this.cache = new Map(); this.S = PAGE_ROWS; this.bytesFetched = 0; this.requests = 0;
    this.inflight = 0; this.waiting = []; }
  static MAX_INFLIGHT = 24;                // a long prompt touches thousands of pages: unbounded fetch() fails (ERR_INSUFFICIENT_RESOURCES)
  static async open(url) {
    const get = async (a, b) => new Uint8Array(await (await fetch(url, { headers: { Range: `bytes=${a}-${b}` } })).arrayBuffer());
    const n = Number(new DataView((await get(0, 7)).buffer).getBigUint64(0, true));
    return new EngramTable(url, JSON.parse(new TextDecoder().decode(await get(8, 8 + n - 1))), 8 + n);
  }
  get rows() { return this.t.weight.shape[0]; }
  async _range(name, rowStart, rows, rowBytes) {
    const o = this.base + this.t[name].data_offsets[0] + rowStart * rowBytes, len = rows * rowBytes;
    this.requests++; this.bytesFetched += len;
    if (this.inflight >= EngramTable.MAX_INFLIGHT) await new Promise(r => this.waiting.push(r));
    this.inflight++;
    try {
      const r = await fetch(this.url, { headers: { Range: `bytes=${o}-${o + len - 1}` } });
      if (!r.ok) throw new Error(`engram table: HTTP ${r.status}`);
      const u8 = new Uint8Array(await r.arrayBuffer());
      if (u8.byteLength !== len) throw new Error(`engram table: ${u8.byteLength} B for a ${len} B range (no Range support?)`);
      return u8;
    } finally { this.inflight--; this.waiting.shift()?.(); }
  }
  /** n pages from page p0 (global page = order * pagesPerOrder + page), in the GPU layout: per page 56 x 64 B codes, 56 x 2 fp16
   *  scales, 56 x 2 fp16 biases. Three Range requests. */
  async pages(p0, n) {
    const S = PAGE_ROWS, r0 = p0 * S, rows = n * S;
    const [w, sc, bi] = await Promise.all([this._range("weight", r0, rows, 64), this._range("scales", r0, rows, 4), this._range("biases", r0, rows, 4)]);
    const out = new Uint8Array(n * PAGE_BYTES);
    for (let p = 0; p < n; p++) {
      out.set(w.subarray(p * S * 64, (p + 1) * S * 64), p * PAGE_BYTES);
      out.set(sc.subarray(p * S * 4, (p + 1) * S * 4), p * PAGE_BYTES + S * 64);
      out.set(bi.subarray(p * S * 4, (p + 1) * S * 4), p * PAGE_BYTES + S * 68);
    }
    return out;
  }
  async _page(p) {                       // p = floor(row / S); returns dequantised Float32Array [S, 128]
    if (this.cache.has(p)) return this.cache.get(p);
    const S = this.S, start = p * S;
    const job = (async () => {
      const [w, sc, bi] = await Promise.all([this._range("weight", start, S, 64), this._range("scales", start, S, 4), this._range("biases", start, S, 4)]);
      const sv = new DataView(sc.buffer, sc.byteOffset, sc.byteLength), bv = new DataView(bi.buffer, bi.byteOffset, bi.byteLength);
      const out = new Float32Array(S * 128);
      for (let r = 0; r < S; r++) for (let g = 0; g < 2; g++) {
        const s = f16(sv.getUint16((r * 2 + g) * 2, true)), b = f16(bv.getUint16((r * 2 + g) * 2, true));
        for (let j = 0; j < 32; j++) { const byte = w[r * 64 + g * 32 + j];
          out[r * 128 + g * 64 + 2 * j] = (byte & 15) * s + b; out[r * 128 + g * 64 + 2 * j + 1] = (byte >> 4) * s + b; }
      }
      return out;
    })();
    job.catch(() => this.cache.delete(p));   // a failed fetch must not poison the page for the rest of the session
    this.cache.set(p, job); return job;
  }
  /** rows: number[T][16] global row ids -> Float32Array [T*16*128] (the graph's engram_rows input, flattened). */
  async gather(rows) {
    const T = rows.length, K = rows[0].length, out = new Float32Array(T * K * 128);
    await Promise.all(rows.flatMap((row, t) => row.map(async (r, k) => {
      const page = await this._page(Math.floor(r / this.S)); const slot = r % this.S;
      out.set(page.subarray(slot * 128, slot * 128 + 128), (t * K + k) * 128);
    })));
    return out;
  }
}
