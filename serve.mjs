// Local static server with HTTP Range support (the engine reads model files by range; python's http.server can't).
// usage: node serve.mjs [port=8000]   then open http://localhost:8000/
import http from "http"; import fs from "fs"; import path from "path";
const root = path.dirname(new URL(import.meta.url).pathname), port = +process.argv[2] || 8000;
const types = { ".html": "text/html", ".mjs": "text/javascript", ".js": "text/javascript", ".json": "application/json", ".md": "text/markdown" };
http.createServer((req, res) => {
  let p = path.join(root, decodeURIComponent(req.url.split("?")[0])), st;
  try { st = fs.statSync(p); if (st.isDirectory()) { p = path.join(p, "index.html"); st = fs.statSync(p); } } catch { res.writeHead(404).end(); return; }
  if (!p.startsWith(root)) { res.writeHead(403).end(); return; }
  const h = { "Content-Type": types[path.extname(p)] || "application/octet-stream", "Accept-Ranges": "bytes", "ETag": `W/"${st.size}-${Math.round(st.mtimeMs)}"`,
    "Last-Modified": st.mtime.toUTCString(), "Cache-Control": "no-cache" };
  const m = /bytes=(\d*)-(\d*)/.exec(req.headers.range || "");
  if (m) { const a = m[1] ? +m[1] : 0, b = m[2] ? Math.min(+m[2], st.size - 1) : st.size - 1;
    res.writeHead(206, { ...h, "Content-Range": `bytes ${a}-${b}/${st.size}`, "Content-Length": b - a + 1 }); if (req.method === "HEAD") return res.end(); fs.createReadStream(p, { start: a, end: b }).pipe(res); }
  else { res.writeHead(200, { ...h, "Content-Length": st.size }); if (req.method === "HEAD") return res.end(); fs.createReadStream(p).pipe(res); }
}).listen(port, () => console.log(`http://localhost:${port}/`));
