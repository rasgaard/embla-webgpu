// Something to watch while the weights download: Embla's 31 mixture-of-experts layers x 96 experts as a grid.
// Cells light up as the download progresses; every key typed (or click) sends a "token" down through the layers, flashing the
// 8 experts it is routed to in each one. The routing here is a hash of the key, not the model's router: it only shows the shape
// of the computation (8 of 96 experts a layer, so ~1B of the 10B parameters run for a token).
const LAYERS = 31, EXPERTS = 96, TOPK = 8, STEP_MS = 42;

const hash = (a, b) => { let h = Math.imul(a ^ 0x9e3779b9, 0x85ebca6b) ^ Math.imul(b + 0x632be5ab, 0xc2b2ae35);
  h = Math.imul(h ^ (h >>> 15), 0x2c1b3c6d); h = Math.imul(h ^ (h >>> 12), 0x297a2d39); return (h ^ (h >>> 15)) >>> 0; };
const route = (seed, layer) => { const s = new Set(); for (let i = 0; s.size < TOPK; i++) s.add(hash(seed, layer * 131 + i) % EXPERTS); return s; };
const rgb = (hex) => { const m = /^#?([\da-f]{2})([\da-f]{2})([\da-f]{2})$/i.exec(hex.trim()); return m ? m.slice(1).map(x => parseInt(x, 16)) : [128, 128, 128]; };
const mix = (a, b, t) => a.map((v, i) => Math.round(v + (b[i] - v) * t));

export function moeToy(canvas, { onToken = () => {} } = {}) {
  const ctx = canvas.getContext("2d");
  const reduce = matchMedia("(prefers-reduced-motion: reduce)").matches;
  const heat = new Float32Array(LAYERS * EXPERTS);
  const tokens = [];
  let progress = 0, shown = 0, last = 0, lastDraw = 0, idle = 900, raf = 0, running = true, seq = 0, colors;

  const theme = () => {
    const css = getComputedStyle(document.documentElement), v = (n) => css.getPropertyValue(n);
    const a = rgb(v("--accent")), b = rgb(v("--accent-2"));
    colors = { rows: Array.from({ length: LAYERS }, (_, l) => mix(a, b, l / (LAYERS - 1))), off: rgb(v("--line")), ink: v("--muted").trim() };
  };
  const resize = () => {
    const dpr = devicePixelRatio || 1, w = canvas.clientWidth, h = canvas.clientHeight;
    canvas.width = Math.round(w * dpr); canvas.height = Math.round(h * dpr); ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  };
  theme(); resize();
  const ro = new ResizeObserver(() => { resize(); if (reduce) draw(); }); ro.observe(canvas);
  const mq = matchMedia("(prefers-color-scheme: dark)"), onScheme = () => { theme(); if (reduce) draw(); }; mq.addEventListener("change", onScheme);

  function send(label) {
    tokens.push({ label, seed: hash(label.codePointAt(0) || 0, ++seq), layer: -1, t: 0 });
    if (tokens.length > 16) tokens.shift();
    onToken(label);
    if (reduce) { const tk = tokens.at(-1); for (let l = 0; l < LAYERS; l++) for (const e of route(tk.seed, l)) heat[l * EXPERTS + e] = 1; draw(); }
  }

  function draw() {
    lastDraw = performance.now();
    const W = canvas.clientWidth, H = canvas.clientHeight, pad = 16, cw = (W - pad) / EXPERTS, ch = H / LAYERS;
    ctx.clearRect(0, 0, W, H);
    const lit = shown * LAYERS * EXPERTS;
    for (let l = 0; l < LAYERS; l++) {
      const [r, g, b] = colors.rows[l], [or, og, ob] = colors.off;
      for (let e = 0; e < EXPERTS; e++) {
        const n = l * EXPERTS + e, on = n < lit, hot = heat[n];
        if (hot > 0.02) ctx.fillStyle = `rgba(${r},${g},${b},${(on ? 0.55 : 0.3) + (on ? 0.45 : 0.7) * hot})`;
        else ctx.fillStyle = on ? `rgba(${r},${g},${b},0.55)` : `rgba(${or},${og},${ob},0.85)`;
        ctx.fillRect(pad + e * cw + 0.5, l * ch + 0.5, Math.max(1, cw - 1), Math.max(1, ch - 1));
      }
    }
    ctx.fillStyle = colors.ink; ctx.font = `600 ${Math.min(12, ch * 1.6)}px system-ui, sans-serif`; ctx.textAlign = "center"; ctx.textBaseline = "middle";
    for (const tk of tokens) if (tk.layer >= 0 && tk.layer < LAYERS) ctx.fillText(tk.label, pad / 2, (tk.layer + 0.5) * ch);
  }

  function frame(now) {
    if (!running) return;
    const real = now - (last || now), dt = Math.min(64, real); last = now;   // frames are rare while the weights upload: catch up in real time
    shown += (progress - shown) * (1 - Math.exp(-real / 250));
    if ((idle += dt) > 1600) { idle = 0; send("·"); }
    for (const tk of tokens) {
      tk.t += dt;
      while (tk.t >= STEP_MS) { tk.t -= STEP_MS; tk.layer++; if (tk.layer < LAYERS) for (const e of route(tk.seed, tk.layer)) heat[tk.layer * EXPERTS + e] = 1; }
    }
    for (let i = tokens.length - 1; i >= 0; i--) if (tokens[i].layer > LAYERS + 4) tokens.splice(i, 1);
    const decay = Math.exp(-dt / 420); for (let i = 0; i < heat.length; i++) heat[i] *= decay;
    draw();
    raf = requestAnimationFrame(frame);
  }

  const onKey = (e) => {
    if (e.metaKey || e.ctrlKey || e.altKey || /^(INPUT|TEXTAREA|SELECT|BUTTON)$/.test(e.target.tagName)) return;
    if (e.key.length === 1 && e.key !== " ") { idle = 0; send(e.key); }
  };
  const onClick = () => { idle = 0; send("•"); };
  addEventListener("keydown", onKey); canvas.addEventListener("pointerdown", onClick);
  if (reduce) draw(); else raf = requestAnimationFrame(frame);

  return {
    /** 0..1: share of the weights downloaded / uploaded so far. */
    setProgress(p) {
      progress = Math.max(0, Math.min(1, p));
      const now = performance.now();                    // animation frames starve while the CPU converts weights: draw from here too
      if (reduce || now - lastDraw > 100) { shown = Math.max(shown, progress - 0.01); draw(); }
    },
    stop() { running = false; cancelAnimationFrame(raf); ro.disconnect(); mq.removeEventListener("change", onScheme);
      removeEventListener("keydown", onKey); canvas.removeEventListener("pointerdown", onClick); },
  };
}
