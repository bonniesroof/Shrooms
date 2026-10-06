// The map: a surface view (ground, plants, insects, contamination) and an
// underground view (soil, bacteria, contaminant plume, and the glowing hyphal
// network with nutrient pulses from the keystone agent's actions).

import { Application, Container, Graphics, Sprite, Texture } from "pixi.js";
import type { Field, Flow } from "./net";

export const CELL = 10; // world px per grid cell
const PATCH = 8; // cells per patch (matches the sim's network patches)
const NETWORK_MIN = 1.2; // g fungal C per cell to draw a hypha

type View = "surface" | "underground";

function hash(i: number): number { // deterministic 0..1 jitter per cell
  const x = Math.sin(i * 127.1 + 311.7) * 43758.5453;
  return x - Math.floor(x);
}

function canvasTexture(w: number, h: number): { tex: Texture; ctx: CanvasRenderingContext2D; img: ImageData } {
  const c = document.createElement("canvas");
  c.width = w; c.height = h;
  const ctx = c.getContext("2d")!;
  const tex = Texture.from(c);
  tex.source.scaleMode = "linear";
  return { tex, ctx, img: ctx.createImageData(w, h) };
}

export class World {
  readonly root = new Container();
  private surface = new Container();
  private under = new Container();
  private plantLayer = new Container(); // static between frames: cached as a texture
  private netLayer = new Container();   // likewise: thousands of strands, one quad per frame
  private netBuiltAt = -1e9;
  private netDirty = true;
  lastUpdateMs = 0;
  private groundTex; private soilTex;
  private plants: Sprite[] = [];
  private insects: { s: Sprite; x: number; y: number; vx: number; vy: number }[] = [];
  private glow = new Graphics(); private core = new Graphics();
  private pulses: { s: Sprite; ax: number; ay: number; bx: number; by: number; t: number; speed: number }[] = [];
  private overlay = new Graphics();
  private site = new Graphics();
  private dotTex: Texture; private bugTex: Texture; private sparkTex: Texture;
  private fields: Record<string, Field> = {};
  private w = 64; private h = 64;
  view: View = "surface";
  segments = 0;

  constructor(private app: Application, w = 64, h = 64) {
    this.w = w; this.h = h;
    this.dotTex = app.renderer.generateTexture(new Graphics().circle(0, 0, 8).fill(0xffffff));
    this.bugTex = app.renderer.generateTexture(new Graphics().ellipse(0, 0, 3, 2).fill(0xffffff));
    this.sparkTex = app.renderer.generateTexture(
      new Graphics().circle(0, 0, 14).fill({ color: 0xffffff, alpha: 0.12 }).circle(0, 0, 8)
        .fill({ color: 0xffffff, alpha: 0.3 }).circle(0, 0, 4).fill(0xffffff));
    this.groundTex = canvasTexture(w, h);
    this.soilTex = canvasTexture(w, h);

    const ground = new Sprite(this.groundTex.tex); ground.scale.set(CELL);
    const soil = new Sprite(this.soilTex.tex); soil.scale.set(CELL);
    this.surface.addChild(ground, this.plantLayer);
    for (let i = 0; i < w * h; i++) {
      const s = new Sprite(this.dotTex);
      s.anchor.set(0.5);
      s.position.set(((i % w) + 0.2 + 0.6 * hash(i)) * CELL, (Math.floor(i / w) + 0.2 + 0.6 * hash(i + 9e3)) * CELL);
      s.visible = false;
      this.plants.push(s); this.plantLayer.addChild(s);
    }
    for (let i = 0; i < 600; i++) {
      const s = new Sprite(this.bugTex); s.anchor.set(0.5); s.tint = 0x2a1d10; s.visible = false;
      this.insects.push({ s, x: Math.random() * w, y: Math.random() * h, vx: 0, vy: 0 });
      this.surface.addChild(s);
    }
    this.netLayer.blendMode = "add";
    this.netLayer.addChild(this.glow, this.core);
    this.under.addChild(soil, this.netLayer);
    for (let i = 0; i < 80; i++) {
      const s = new Sprite(this.sparkTex); s.anchor.set(0.5); s.blendMode = "add"; s.visible = false;
      this.pulses.push({ s, ax: 0, ay: 0, bx: 0, by: 0, t: 0, speed: 0 });
      this.under.addChild(s);
    }
    this.root.addChild(this.surface, this.under, this.site, this.overlay);
    this.plantLayer.cacheAsTexture(true);
    this.netLayer.cacheAsTexture(true);
    this.setView("surface");
  }

  setView(v: View): void {
    this.view = v;
    if (v === "underground" && this.netDirty) this.buildNetwork();
    this.surface.visible = v === "surface";
    this.under.visible = v === "underground";
  }

  /** Entities drawn in the current view: sprites plus hyphal strands. */
  entityCount(): number {
    const vis = (a: { visible: boolean }[]) => a.filter((s) => s.visible).length;
    return this.view === "surface"
      ? vis(this.plants) + vis(this.insects.map((i) => i.s))
      : vis(this.pulses.map((p) => p.s)) + this.segments;
  }

  value(name: string, x: number, y: number): number {
    const f = this.fields[name];
    return f ? f[y * this.w + x] : 0;
  }

  update(fields: Record<string, Field>, flows: Flow[], site: [number, number, number, number] | null): void {
    const t0 = performance.now();
    this.fields = fields;
    const { w, h } = this;
    const plant = fields.plant, moist = fields.moisture, contam = fields.contaminant;
    const litter = fields.litter, myco = fields.mycorrhiza, bact = fields.bacteria, som = fields.som;

    // Surface ground: dry soil -> damp soil, litter mulch, oily contamination.
    const g = this.groundTex.img.data;
    for (let i = 0; i < w * h; i++) {
      const m = moist ? moist[i] : 0.5, l = litter ? Math.min(1, litter[i] / 600) : 0;
      const c = contam ? Math.min(1, contam[i] / 30) : 0;
      let r = 150 - 70 * m - 30 * l, gg = 118 - 50 * m - 15 * l, b = 82 - 40 * m - 20 * l;
      r = r * (1 - c) + 70 * c; gg = gg * (1 - c) + 40 * c; b = b * (1 - c) + 70 * c;
      g[i * 4] = r; g[i * 4 + 1] = gg; g[i * 4 + 2] = b; g[i * 4 + 3] = 255;
    }
    this.groundTex.ctx.putImageData(this.groundTex.img, 0, 0); this.groundTex.tex.source.update();

    // Plants: one sprite per cell, sized by biomass, yellowed by toxicity.
    for (let i = 0; i < w * h; i++) {
      const s = this.plants[i], p = plant ? plant[i] : 0;
      s.visible = p > 25;
      if (!s.visible) continue;
      s.scale.set(0.18 + 0.5 * Math.sqrt(Math.min(1, p / 900)));
      const tox = contam ? Math.min(1, contam[i] / 25) : 0;
      const gr = Math.round(150 + 60 * hash(i) - 60 * tox), rr = Math.round(40 + 140 * tox);
      s.tint = (rr << 16) | (gr << 8) | 50;
    }

    // Insects: as many bugs as the meadow holds, re-homed toward dense cells.
    const ins = fields.insects;
    let total = 0;
    if (ins) for (let i = 0; i < w * h; i++) total += ins[i];
    const want = Math.min(this.insects.length, Math.round(total * 0.6));
    this.insects.forEach((b, k) => {
      b.s.visible = k < want;
      if (b.s.visible && ins && Math.random() < 0.15) {
        for (let tries = 0; tries < 8; tries++) {
          const i = Math.floor(Math.random() * w * h);
          if (Math.random() < ins[i] / 1.0) { b.x = (i % w) + Math.random(); b.y = Math.floor(i / w) + Math.random(); break; }
        }
      }
    });

    // Underground: soil organic matter, bacteria glow, contaminant plume.
    const u = this.soilTex.img.data;
    for (let i = 0; i < w * h; i++) {
      const o = som ? Math.min(1, som[i] / 2500) : 0.5, bb = bact ? Math.min(1, bact[i] / 20) : 0;
      const c = contam ? Math.min(1, contam[i] / 30) : 0;
      u[i * 4] = 18 + 30 * o + 60 * bb + 150 * c; u[i * 4 + 1] = 12 + 20 * o + 30 * bb + 20 * c;
      u[i * 4 + 2] = 10 + 12 * o + 10 * c; u[i * 4 + 3] = 255;
    }
    this.soilTex.ctx.putImageData(this.soilTex.img, 0, 0); this.soilTex.tex.source.update();

    // Hyphal network: rebuilt at most twice a second, and only while visible.
    this.netDirty = true;
    if (this.view === "underground" && performance.now() - this.netBuiltAt > 500) this.buildNetwork();

    // Nutrient pulses along the keystone agent's recent shuttles and relocations.
    const centre = (pid: string | null): [number, number] | null => {
      const m = pid && /^r(\d+)c(\d+)$/.exec(pid);
      return m ? [(+m[2] * PATCH + PATCH / 2) * CELL, (+m[1] * PATCH + PATCH / 2) * CELL] : null;
    };
    let k = 0;
    for (const f of flows) {
      const a = centre(f.from), b = centre(f.to);
      if (!a || !b) continue;
      const color = f.element === "p" ? 0xff7ad9 : f.element === "n" ? 0x7af4ff : 0x9dff7a;
      for (let n = 0; n < 6 && k < this.pulses.length; n++, k++) {
        const p = this.pulses[k];
        Object.assign(p, { ax: a[0], ay: a[1], bx: b[0], by: b[1], t: n / 6, speed: 0.25 + 0.1 * hash(k) });
        p.s.tint = color; p.s.visible = true;
      }
    }
    for (; k < this.pulses.length; k++) this.pulses[k].s.visible = false;

    this.plantLayer.updateCacheTexture();
    this.site.clear();
    if (site) {
      const [x0, y0, x1, y1] = site;
      this.site.rect(x0 * CELL, y0 * CELL, (x1 - x0) * CELL, (y1 - y0) * CELL)
        .stroke({ width: 1.5, color: 0xffd36b, alpha: 0.7 });
    }
    this.lastUpdateMs = performance.now() - t0;
  }

  /** Cell-to-cell strands where fungi are dense enough, in three strengths. */
  private buildNetwork(): void {
    const myco = this.fields.mycorrhiza, { w, h } = this;
    this.glow.clear(); this.core.clear(); this.segments = 0;
    if (myco) {
      const node = (i: number): [number, number] =>
        [((i % w) + 0.15 + 0.7 * hash(i)) * CELL, (Math.floor(i / w) + 0.15 + 0.7 * hash(i + 777)) * CELL];
      const buckets: [number, number, number, number][][] = [[], [], []];
      for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) {
        const i = y * w + x;
        if (myco[i] < NETWORK_MIN) continue;
        const links = [x + 1 < w ? i + 1 : -1, y + 1 < h ? i + w : -1,
                       x + 1 < w && y + 1 < h && hash(i + 5) > 0.75 ? i + w + 1 : -1];
        for (let d = 0; d < links.length; d++) {
          const j = links[d];
          if (j < 0 || myco[j] < NETWORK_MIN || hash(i * 3 + d + 11) < 0.35) continue; // prune: organic, not a grid
          const strength = Math.min(myco[i], myco[j]);
          const k = strength > 12 ? 2 : strength > 5 ? 1 : 0;
          const [ax, ay] = node(i), [bx, by] = node(j);
          buckets[k].push([ax, ay, bx, by]);
        }
      }
      const colors = [0x2f7d68, 0x46b48c, 0x8fe3b8];
      buckets.forEach((segs, k) => {
        if (!segs.length) return;
        for (const [ax, ay, bx, by] of segs) { this.glow.moveTo(ax, ay).lineTo(bx, by); this.core.moveTo(ax, ay).lineTo(bx, by); }
        this.glow.stroke({ width: 3 + 2 * k, color: colors[k], alpha: 0.05 + 0.03 * k });
        this.core.stroke({ width: 0.5 + 0.35 * k, color: colors[k], alpha: 0.35 + 0.15 * k });
        this.segments += segs.length;
      });
    }

    this.netLayer.updateCacheTexture();
    this.netBuiltAt = performance.now();
    this.netDirty = false;
  }

  animate(dtMs: number): void {
    const dt = dtMs / 1000;
    if (this.view === "surface") {
      for (const b of this.insects) {
        if (!b.s.visible) continue;
        b.vx += (Math.random() - 0.5) * 6 * dt; b.vy += (Math.random() - 0.5) * 6 * dt;
        b.vx *= 0.96; b.vy *= 0.96;
        b.x = Math.min(this.w - 0.01, Math.max(0, b.x + b.vx * dt));
        b.y = Math.min(this.h - 0.01, Math.max(0, b.y + b.vy * dt));
        b.s.position.set(b.x * CELL, b.y * CELL);
        b.s.rotation = Math.atan2(b.vy, b.vx);
      }
    } else {
      for (const p of this.pulses) {
        if (!p.s.visible) continue;
        p.t = (p.t + p.speed * dt) % 1;
        p.s.position.set(p.ax + (p.bx - p.ax) * p.t, p.ay + (p.by - p.ay) * p.t);
        p.s.alpha = Math.sin(p.t * Math.PI);
      }
    }
  }

  cursor(x: number | null, y: number | null, tool: string | null): void {
    this.overlay.clear();
    if (x === null || y === null || !tool) return;
    const r = { inoculate_bacteria: 4, inoculate_fungi: 4, inoculate_mycorrhiza: 4, compost: 5, seed: 5,
                irrigate: 5, spill: 3 }[tool];
    const cx = (x + 0.5) * CELL, cy = (y + 0.5) * CELL;
    if (r) this.overlay.circle(cx, cy, r * CELL * 1.5).stroke({ width: 1.5, color: 0xffffff, alpha: 0.8 });
    else {
      const half = tool === "excavate" ? 4 : 3;
      this.overlay.rect((x - half) * CELL, (y - half) * CELL, 2 * half * CELL, 2 * half * CELL)
        .stroke({ width: 1.5, color: 0xffffff, alpha: 0.8 });
    }
  }
}
