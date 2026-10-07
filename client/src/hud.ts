// The HUD: time controls, view and mode switches, tools and budget, scenario
// objectives, the inspector (observation -> intent -> rationale), the whisper
// box, and the feed. Plain DOM over the canvas.

import type { Connection, Frame, Hello } from "./net";

const $ = <T extends HTMLElement>(id: string) => document.getElementById(id) as T;
const esc = (s: unknown) => String(s ?? "").replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" })[c]!);
const SPEED_LABELS: Record<number, string> = { 0: "Pause", 6: "¼ d/s", 24: "1 d/s", 96: "4 d/s", 336: "2 wk/s" };

function describe(i: Record<string, unknown>): string {
  const k = String(i.kind);
  if (k === "shuttle_nutrients") return `shuttle ${i.amount_g} g ${String(i.element).toUpperCase()} ${i.from_patch}→${i.to_patch}`;
  if (k === "relocate_hyphae") return `relocate ${Math.round(Number(i.fraction) * 100)}% hyphae ${i.from_patch}→${i.to_patch}`;
  if (k === "set_trade_bias") return `trade bias ${i.bias} in ${i.patch}`;
  return k;
}

export class Hud {
  tool: string | null = null;
  private inspectorId = -1;
  private toolsBuilt = false;
  private hello: Hello | null = null;
  onView: (v: "surface" | "underground") => void = () => {};
  onForecast: (head: string, horizon: number) => void = () => {};
  private fcHead = "biomass";
  private fcHorizon = 0;
  private lastForecast: Frame["forecast"] | null = null;

  constructor(private conn: Connection) {
    $("view").onclick = () => {
      const under = $("view").dataset.v !== "underground";
      $("view").dataset.v = under ? "underground" : "surface";
      $("view").textContent = under ? "⬆ Surface" : "⬇ Underground";
      this.onView(under ? "underground" : "surface");
    };
    $("mode").onclick = () => conn.send({ cmd: "mode", value: $("mode").dataset.m === "game" ? "research" : "game" });
    $("step").onclick = () => conn.send({ cmd: "step", days: 1 });
    $("newgame").onclick = () => conn.send({ cmd: "new_game" });
    const whisper = () => {
      const t = $<HTMLInputElement>("whisper-text");
      if (t.value.trim()) { conn.send({ cmd: "whisper", text: t.value }); t.value = ""; }
    };
    $("whisper-send").onclick = whisper;
    $<HTMLInputElement>("whisper-text").onkeydown = (e) => { if (e.key === "Enter") whisper(); };
    $("fc-horizon").querySelectorAll<HTMLButtonElement>("button").forEach((b) => (b.onclick = () => {
      this.fcHorizon = Number(b.dataset.h);
      $("fc-horizon").querySelectorAll("button").forEach((x) => x.classList.toggle("on", x === b));
      this.onForecast(this.fcHead, this.fcHorizon);
      this.forecastNote();
    }));
    $<HTMLSelectElement>("fc-head").onchange = (e) => {
      this.fcHead = (e.target as HTMLSelectElement).value;
      this.onForecast(this.fcHead, this.fcHorizon);
      this.forecastNote();
    };
    window.addEventListener("keydown", (e) => {
      if ((e.target as HTMLElement).tagName === "INPUT") return;
      if (e.key === " ") { e.preventDefault(); $("speeds").querySelector<HTMLButtonElement>(this.paused ? "[data-s='24']" : "[data-s='0']")?.click(); }
      if (e.key === "Tab") { e.preventDefault(); $("view").click(); }
      if (e.key === "Escape") this.select(null);
    });
  }

  private paused = true;

  setHello(h: Hello): void {
    this.hello = h;
    $("speeds").innerHTML = h.speeds.map((s) => `<button data-s="${s}">${SPEED_LABELS[s] ?? s}</button>`).join("");
    $("speeds").querySelectorAll<HTMLButtonElement>("button").forEach((b) => {
      b.onclick = () => this.conn.send({ cmd: "speed", value: Number(b.dataset.s) });
    });
    this.toolsBuilt = false;
  }

  select(tool: string | null): void {
    this.tool = this.tool === tool ? null : tool;
    $("tools").querySelectorAll<HTMLButtonElement>("button").forEach((b) => b.classList.toggle("on", b.dataset.t === this.tool));
  }

  toast(text: string, bad = false): void {
    const t = $("toast");
    t.textContent = text; t.className = bad ? "show bad" : "show";
    clearTimeout((t as unknown as { _h: number })._h);
    (t as unknown as { _h: number })._h = window.setTimeout(() => (t.className = ""), 2600);
  }

  perf(fps: number, entities: number): void {
    $("perf").textContent = `${fps.toFixed(0)} fps · ${entities} entities`;
  }

  hover(text: string): void { $("hover").textContent = text; }


  private forecastNote(): void {
    const fc = this.lastForecast;
    if (!fc) return;
    const note = $("fc-note");
    if (!fc.available) note.textContent = `Forecast unavailable: ${fc.reason}`;
    else if (!this.fcHorizon) note.textContent = `Made on day ${fc.made_day + 1}. Pick a horizon to show it.`;
    else {
      const skill = fc.skill[`${this.fcHead}@${this.fcHorizon}d`];
      const legend = this.fcHead === "links" ? "green: forms · red: breaks · faint: stays"
        : this.fcHead === "mortality" ? "red: share of plants likely to die back"
        : "green: good news · red: bad news";
      note.textContent = `${legend}. Skill vs “no change” on unseen worlds: ${skill === undefined ? "–" : (skill * 100).toFixed(0) + "%"}.`;
    }
  }

  update(f: Frame): void {
    this.paused = f.speed === 0;
    $("date").textContent = f.date;
    $("speeds").querySelectorAll<HTMLButtonElement>("button").forEach((b) => b.classList.toggle("on", Number(b.dataset.s) === f.speed));
    $("mode").dataset.m = f.mode;
    $("mode").textContent = f.mode === "game" ? "🎮 Game" : "🔬 Research";
    $("thinking").textContent = f.thinking ? `🍄 ${f.thinking} is thinking…` : "";
    const st = f.stats;
    $("stats").innerHTML = [
      `🌡 ${st.temp_c ?? "–"}°C${st.raining ? " 🌧" : ""}`, `💧 ${Math.round(Number(st.moisture) * 100)}%`,
      `🌿 ${st.plant_c} gC/m²`, `🍄 ${st.fungal_c}`, `🦠 ${st.bacteria_c}`, `☣ ${st.contaminant_kg} kg`,
    ].map((s) => `<span>${s}</span>`).join("");

    // Tools (rebuilt when the mode changes, since some are research-only).
    if (this.hello && (!this.toolsBuilt || $("tools").dataset.mode !== f.mode)) {
      $("tools").dataset.mode = f.mode;
      $("tools").innerHTML = Object.entries(this.hello.tools)
        .filter(([, t]) => !t.research_only || f.mode === "research")
        .map(([k, t]) => `<button data-t="${k}" title="${esc(t.hint)}">${esc(t.label)}<small>${f.mode === "game" && f.scenario ? t.cost : "free"}</small></button>`)
        .join("");
      $("tools").querySelectorAll<HTMLButtonElement>("button").forEach((b) => (b.onclick = () => this.select(b.dataset.t!)));
      this.select(null); this.select(this.tool);
      this.toolsBuilt = true;
    }
    $("budget").textContent = f.scenario ? (f.mode === "game" ? `Budget ${f.budget}` : "Research mode: tools are free") : "";

    // Scenario objectives.
    const sc = f.scenario;
    if (sc) {
      const fmt = (o: { key: string; value: number; target: number }) =>
        o.key === "network_patches" ? `${o.value}/${o.target}` : `${Math.round(o.value * 100)}% (goal ${o.key === "contaminant_left" ? "≤" : "≥"} ${Math.round(o.target * 100)}%)`;
      $("objectives").innerHTML = `<h3>${esc(sc.title)}</h3><div class="days">${sc.days_left} days left${f.research_used ? " · unranked" : ""}</div>` +
        sc.objectives.map((o) => `<div class="obj ${o.done ? "done" : ""}"><div>${esc(o.label)} <b>${fmt(o)}</b></div><div class="bar"><i style="width:${Math.round(o.progress * 100)}%"></i></div></div>`).join("");
      const banner = $("banner");
      banner.className = sc.status === "playing" ? "" : `show ${sc.status}`;
      banner.innerHTML = sc.status === "won" ? `🌱 The site is alive again — won on day ${sc.day}.<br><button id="again">New game</button>`
        : sc.status === "lost" ? `⌛ The grant ran out. Contamination and bare ground remain.<br><button id="again">Try again</button>` : "";
      document.getElementById("again")?.addEventListener("click", () => this.conn.send({ cmd: "new_game" }));
    }

    // Inspector: the keystone agent's latest decision.
    const ins = f.inspector;
    if (ins && ins.id !== this.inspectorId) {
      this.inspectorId = ins.id;
      const net = ins.observation.network ?? {};
      const needy = (ins.observation.needy_partners ?? []).map((n) => `${esc(n.patch)} (C:N ${esc(n.plant_cn)})`).join(", ") || "none";
      const market = ins.observation.market?.active ? "open" : "closed";
      $("inspector").innerHTML = `
        <div class="when">Decision ${ins.id} · ${esc(ins.date)}</div>
        ${ins.whisper ? `<div class="whisper">🗣 you whispered: “${esc(ins.whisper)}”</div>` : ""}
        <h4>Observation</h4><div>Network on ${esc(net.patches_on_network)}/${esc(net.patches_total)} patches · market ${market} · hungriest: ${needy}</div>
        <h4>Reasoning</h4><div class="thought">${esc(ins.deliberation)}</div>
        <h4>Intents</h4>${ins.accepted.length ? ins.accepted.map((a) => `<div class="ok">✔ ${esc(describe(a))}<small>${esc(a.rationale)}</small></div>`).join("") : "<div>— none —</div>"}
        ${ins.rejected.map((r) => `<div class="no">✖ ${esc(describe(r.proposal))}<small>${esc(r.violations.join("; "))}</small></div>`).join("")}`;
    }

    $("feed").innerHTML = [...f.feed].reverse().slice(0, 25)
      .map((m) => `<div class="${esc(m.who)}"><span>${esc(m.date)}</span> <b>${esc(m.who)}</b> ${esc(m.text)}</div>`).join("") +
      [...f.events].reverse().slice(0, 10).map((e) => `<div class="event"><span>${esc(e.date)}</span> ${esc(e.message)}</div>`).join("");
    this.lastForecast = f.forecast;
    this.forecastNote();
    const j = f.journal[f.journal.length - 1];
    if (j?.text) $("journal").innerHTML = `<h4>Field journal · ${esc(j.period)}</h4><div>${esc(j.text)}</div>`;
  }
}
