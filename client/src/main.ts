// Placeholder PixiJS scene. The real surface/underground views land in Phase 3;
// until then the debug viewer is `uv run python -m sim.run` (matplotlib PNGs).
import { Application, Text } from "pixi.js";

const app = new Application();
await app.init({ resizeTo: window, background: "#111" });
document.body.appendChild(app.canvas);

const label = new Text({
  text: "Shrooms client: Phase 3",
  style: { fill: "#9c6", fontFamily: "monospace", fontSize: 20 },
});
label.position.set(24, 24);
app.stage.addChild(label);
