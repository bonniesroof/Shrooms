// Shrooms client: connects to the server and renders the live sim.
// With ?bench, it renders a synthetic dense world without a server and
// measures frame rate (Gate B: 60 fps with ~2,000 entities).

import { Application, Container } from "pixi.js";
import { Hud } from "./hud";
import { Connection, decodeFields, type Field, type Flow } from "./net";
import { CELL, World } from "./world";

const params = new URLSearchParams(location.search);

// No top-level await: Pixi's init() dynamically imports a chunk that imports
// this module back, so awaiting init() at module level deadlocks the import.
async function start(): Promise<void> {
  const app = new Application();
  // WebGL by default: robust everywhere, and WebGPU init can stall on some drivers.
  const preference = params.get("renderer") === "webgpu" ? "webgpu" : "webgl";
  await app.init({ resizeTo: document.getElementById("map")!, background: "#0d0f0c", antialias: params.get("aa") !== "0", preference });

  document.getElementById("map")!.appendChild(app.canvas);

  const world = new World(app);
  const camera = new Container();
  camera.addChild(world.root);
  app.stage.addChild(camera);

  function fit(): void {
    const s = Math.min(app.screen.width, app.screen.height) / (64 * CELL);
    camera.scale.set(s);
    camera.position.set((app.screen.width - 64 * CELL * s) / 2, (app.screen.height - 64 * CELL * s) / 2);
  }
  fit();
  app.renderer.on("resize", fit);

  let fpsAvg = 60;
  app.ticker.add((t) => {
    world.animate(t.deltaMS);
    fpsAvg = fpsAvg * 0.95 + (1000 / Math.max(t.deltaMS, 1)) * 0.05;
  });

  if (params.has("bench")) {
    runBench();
  } else {
    const conn = new Connection();
    const hud = new Hud(conn);
    hud.onView = (v) => world.setView(v);
    hud.onForecast = (head, horizon) => {
      world.forecastHead = head;
      world.forecastHorizon = horizon;
      world.drawForecast();
    };
    conn.onHello = (h) => hud.setHello(h);
    conn.onReply = (r) => { if (r.type === "error") hud.toast(r.error ?? "error", true); };
    conn.onStatus = (s) => { if (s === "closed") hud.toast("Disconnected — reconnecting…", true); };
    conn.onFrame = (f) => {
      world.update(decodeFields(f), f.flows, f.scenario?.site ?? null);
      hud.update(f);
      const made = world.forecast?.available ? world.forecast.made_day : -1;
      world.forecast = f.forecast;
      if (!f.forecast.available || f.forecast.made_day !== made) world.drawForecast();
    };
    setInterval(() => hud.perf(fpsAvg, world.entityCount()), 500);

    const forecastText = (x: number, y: number): string => {
      const p = world.forecastAt(x, y);
      if (p === null) return "";
      const h = world.forecastHorizon, head = world.forecastHead;
      const pct = (d: number) => `${d >= 0 ? "+" : ""}${((Math.exp(d) - 1) * 100).toFixed(0)}%`;
      const text = head === "mortality" ? `${(p * 100).toFixed(0)}% die-back risk`
        : head === "moisture" ? `moisture ${p >= 0 ? "+" : ""}${(p * 100).toFixed(0)} pts`
        : `${head} ${pct(p)}`;
      return ` · forecast ${h} d: ${text}`;
    };
    const cellAt = (e: { global: { x: number; y: number } }) => {
      const p = camera.toLocal(e.global);
      const x = Math.floor(p.x / CELL), y = Math.floor(p.y / CELL);
      return x >= 0 && y >= 0 && x < 64 && y < 64 ? [x, y] : null;
    };
    app.stage.eventMode = "static";
    app.stage.hitArea = app.screen;
    app.stage.on("pointermove", (e) => {
      const c = cellAt(e);
      world.cursor(c?.[0] ?? null, c?.[1] ?? null, hud.tool);
      if (!c) return hud.hover("");
      const [x, y] = c, v = (n: string) => world.value(n, x, y);
      hud.hover(`(${x},${y}) plants ${v("plant").toFixed(0)} g · fungi ${v("mycorrhiza").toFixed(1)} · bacteria ${v("bacteria").toFixed(1)} · contaminant ${v("contaminant").toFixed(1)} g · moisture ${(v("moisture") * 100).toFixed(0)}%` + forecastText(x, y));
    });
    app.stage.on("pointertap", (e) => {
      const c = cellAt(e);
      if (c && hud.tool) conn.send({ cmd: "tool", tool: hud.tool, x: c[0], y: c[1] });
    });
    conn.connect();
  }

  function runBench(): void {
    // A dense synthetic world: every cell planted, hyphae everywhere, insects and pulses.
    const n = 64 * 64, t0 = performance.now();
    const field = (fn: (x: number, y: number, t: number) => number, t: number): Field => {
      const f = new Float32Array(n);
      for (let i = 0; i < n; i++) f[i] = fn(i % 64, Math.floor(i / 64), t);
      return f;
    };
    const flows: Flow[] = Array.from({ length: 12 }, (_, k) => ({
      kind: "shuttle_nutrients", from: `r${k % 8}c${(k * 3) % 8}`, to: `r${(k + 2) % 8}c${(k * 5) % 8}`,
      element: k % 2 ? "n" : "p", tick: 0,
    }));
    const regen = () => {
      const t = (performance.now() - t0) / 1000;
      const wave = (x: number, y: number) => 0.5 + 0.5 * Math.sin(x * 0.3 + t) * Math.cos(y * 0.25 - t * 0.7);
      world.update({
        plant: field((x, y) => 200 + 700 * wave(x, y), t), moisture: field((x, y) => wave(y, x), t),
        contaminant: field((x, y) => 20 * Math.exp(-((x - 32) ** 2 + (y - 32) ** 2) / 60), t),
        mycorrhiza: field((x, y) => 3 + 15 * wave(x + 3, y), t), bacteria: field(() => 8, t),
        litter: field(() => 150, t), som: field(() => 1600, t), insects: field(() => 1.2, t),
      }, flows, [16, 16, 48, 48]);
    };
    regen();
    const view = params.get("view");
    if (view === "underground") world.setView("underground");
    // Rebuild as often as the server sends frames (10 Hz), unless ?regen=0.
    const regenTimer = setInterval(() => { regen(); updates.push(world.lastUpdateMs); },
                                   params.get("regen") === "0" ? 1e9 : 100);
    const samples: number[] = [], updates: number[] = [], animMs: number[] = [];
    const timeAnim = (t: { deltaMS: number }) => { const a = performance.now(); world.animate(t.deltaMS); animMs.push(performance.now() - a); };
    let last = performance.now();
    const sample = () => { const now = performance.now(); samples.push(now - last); last = now; };
    setTimeout(() => {
      app.ticker.add(sample); app.ticker.add(timeAnim);
      setTimeout(() => {
        app.ticker.remove(sample); app.ticker.remove(timeAnim); clearInterval(regenTimer);
        const sorted = [...samples].sort((a, b) => a - b);
        const mean = samples.reduce((a, b) => a + b, 0) / samples.length;
        const avg = (a: number[]) => (a.length ? a.reduce((x, y) => x + y, 0) / a.length : 0);
        const result = { view: world.view, fps: 1000 / mean, p95_ms: sorted[Math.floor(sorted.length * 0.95)],
                         frames: samples.length, entities: world.entityCount(),
                         update_ms: avg(updates), animate_ms: avg(animMs) };
        (window as unknown as { __bench: unknown }).__bench = result;
        document.getElementById("perf")!.textContent = JSON.stringify(result);
      }, 5000);
    }, 1000);
  }

}

void start();
