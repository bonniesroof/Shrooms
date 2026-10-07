// WebSocket connection to the game server, and frame decoding.

export type Field = Float32Array; // H*W values, row-major, in real units

export interface ToolInfo { label: string; cost: number; hint: string; research_only: boolean }
export interface Hello { type: "hello"; sim_version: string; speeds: number[]; tools: Record<string, ToolInfo> }
export interface Objective { key: string; label: string; value: number; target: number; progress: number; done: boolean }
export interface Flow { kind: string; from: string | null; to: string | null; element: string | null; tick: number }
export interface FeedItem { tick: number; date: string; who: string; text: string }

export interface Frame {
  type: "frame";
  tick: number;
  date: string;
  speed: number;
  mode: "game" | "research";
  research_used: boolean;
  grid: [number, number];
  fields: Record<string, string>;
  scales: Record<string, number>;
  stats: Record<string, number | boolean | null>;
  scenario: null | {
    title: string; site: [number, number, number, number]; objectives: Objective[];
    progress: number; day: number; days_left: number; status: "playing" | "won" | "lost";
  };
  budget: number;
  flows: Flow[];
  events: { date: string; kind: string; message: string }[];
  feed: FeedItem[];
  journal: { period?: string; text?: string }[];
  inspector: null | {
    id: number; date: string; whisper: string | null; deliberation: string;
    observation: { network?: Record<string, number>; market?: Record<string, unknown>;
                   needy_partners?: Record<string, unknown>[]; contaminated?: Record<string, unknown>[] };
    accepted: Record<string, unknown>[]; rejected: { proposal: Record<string, unknown>; violations: string[] }[];
  };
  thinking: string | null;
  forecast: Forecast;
}

export type Forecast =
  | { available: false; reason: string }
  | {
      available: true; made_day: number; horizons: number[]; grid: [number, number];
      heads: Record<string, Record<string, number[]>>; links_now: number[]; pairs: [number, number][];
      skill: Record<string, number>;
    };

export function decodeFields(frame: Frame): Record<string, Field> {
  const out: Record<string, Field> = {};
  for (const [name, b64] of Object.entries(frame.fields)) {
    const bin = atob(b64);
    const scale = frame.scales[name] / 255;
    const f = new Float32Array(bin.length);
    for (let i = 0; i < bin.length; i++) f[i] = bin.charCodeAt(i) * scale;
    out[name] = f;
  }
  return out;
}

export class Connection {
  private ws: WebSocket | null = null;
  onHello: (h: Hello) => void = () => {};
  onFrame: (f: Frame) => void = () => {};
  onReply: (r: { type: string; error?: string; cmd?: string }) => void = () => {};
  onStatus: (s: "open" | "closed") => void = () => {};

  connect(): void {
    const proto = location.protocol === "https:" ? "wss" : "ws";
    this.ws = new WebSocket(`${proto}://${location.host}/ws`);
    this.ws.onopen = () => this.onStatus("open");
    this.ws.onclose = () => { this.onStatus("closed"); setTimeout(() => this.connect(), 1500); };
    this.ws.onmessage = (ev) => {
      const msg = JSON.parse(ev.data);
      if (msg.type === "hello") this.onHello(msg);
      else if (msg.type === "frame") this.onFrame(msg);
      else this.onReply(msg);
    };
  }

  send(cmd: Record<string, unknown>): void {
    if (this.ws?.readyState === WebSocket.OPEN) this.ws.send(JSON.stringify(cmd));
  }
}
