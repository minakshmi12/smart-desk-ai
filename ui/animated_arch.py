"""Animated architecture view (self-contained HTML/SVG/JS, no external requests)."""

ANIMATED_HTML = """
<style>
  #wrap{font-family:system-ui,sans-serif;background:#fff;color:#0f172a;border-radius:12px;padding:12px}
  #bar{display:flex;gap:10px;align-items:center;flex-wrap:wrap;margin-bottom:8px}
  #bar select,#bar button{font-size:14px;padding:4px 8px}
  #stats{display:flex;gap:8px;flex-wrap:wrap;margin-bottom:8px}
  .card{border:1px solid #cbd5e1;border-radius:8px;padding:6px 10px;min-width:92px}
  .card b{font-size:22px;display:block}
  .card span{font-size:11px;color:#475569}
  #caption{min-height:44px;margin-top:6px;font-size:14px;background:#f1f5f9;border-radius:8px;padding:8px 10px}
  svg{width:100%;height:auto}
  text{font-size:12px;fill:#0f172a;text-anchor:middle;pointer-events:none}
  .tag{font-size:9px;fill:#475569}
  .edge{fill:none;stroke:#94a3b8;stroke-width:1.6;stroke-dasharray:6 6;animation:flow 1s linear infinite}
  .edge.active{stroke:#2563eb;stroke-width:3}
  @keyframes flow{to{stroke-dashoffset:-12}}
  @media (prefers-reduced-motion: reduce){.edge{animation:none}}
  .n-llm rect{fill:#dbeafe;stroke:#2563eb;stroke-width:1.5}
  .n-code rect{fill:#f1f5f9;stroke:#64748b;stroke-width:1.5}
  .n-store rect{fill:#fef3c7;stroke:#d97706;stroke-width:1.5}
  .n-actor rect{fill:#dcfce7;stroke:#16a34a;stroke-width:1.5}
  .visited rect{stroke-width:3}
  .current rect{stroke:#ea580c;stroke-width:4}
  #boundary{fill:none;stroke:#7c3aed;stroke-width:2;stroke-dasharray:3 5}
  #dot{fill:#ea580c;filter:drop-shadow(0 0 6px #ea580c)}
  .legend{font-size:12px;color:#475569;margin-top:6px}
</style>

<div id="wrap">
  <div id="bar">
    <label>Scenario <select id="scn"></select></label>
    <button id="replay">Replay</button>
    <label><input type="checkbox" id="loop"> Auto-loop</label>
  </div>
  <div id="stats"></div>
  <svg id="svg" viewBox="0 0 860 620"></svg>
  <div id="caption"></div>
  <div class="legend">Blue = node that calls an LLM &middot; Grey = plain code &middot;
    Yellow = data store / external system &middot; Purple dashed = the single agent</div>
</div>

<script>
const AGENTS = 1;
const NODES = {
  emp:    {x:55,  y:305, w:100, label:"Employee",      type:"actor", tag:"",
           info:"The employee sends a message."},
  api:    {x:185, y:305, w:100, label:"FastAPI",       type:"code",  tag:"lock + thread_id",
           info:"Takes the per-conversation lock and passes only the new message to the agent."},
  router: {x:335, y:305, w:100, label:"Router",        type:"llm",   tag:"code first, then LLM", agent:true,
           info:"If a question is pending, code reads yes / no / email. Otherwise GPT-4o classifies the intent."},
  kb:     {x:540, y:60,  label:"kb_answer",     type:"llm",  tag:"LLM + RAG", agent:true,
           info:"Retrieves passages (semantic, keyword or hybrid), checks distance, then GPT-4o answers with citations or says it does not know."},
  offer:  {x:540, y:130, label:"offer_ticket",  type:"code", tag:"code", agent:true,
           info:"Asks for consent and saves the ticket draft. Nothing is created yet."},
  chitchat:{x:540,y:200, label:"chitchat",      type:"llm",  tag:"LLM", agent:true,
           info:"Short friendly reply. No search, no ticket offer."},
  start:  {x:540, y:270, label:"start_ticket",  type:"llm",  tag:"LLM", agent:true,
           info:"GPT-4o drafts a ticket title and summary, then asks the employee to confirm."},
  email:  {x:540, y:340, label:"ask_email",     type:"code", tag:"code", agent:true,
           info:"Asks for the work email, only when it is not already known."},
  create: {x:540, y:410, label:"create_ticket", type:"code", tag:"code + tool", agent:true,
           info:"Creates the ticket, only after an explicit yes and a valid email."},
  status: {x:540, y:480, label:"ticket_status", type:"code", tag:"code + tool", agent:true,
           info:"Looks up tickets by email. Someone else's ticket looks exactly like a missing one."},
  cancel: {x:540, y:550, label:"cancel_ticket", type:"code", tag:"code", agent:true,
           info:"Ends a declined or abandoned ticket flow and clears the draft."},
  chroma: {x:780, y:60,  w:100, label:"Chroma",   type:"store", tag:"vector store",
           info:"Returns the document chunks closest to the question."},
  jira:   {x:780, y:445, w:100, label:"Jira",     type:"store", tag:"ticket platform",
           info:"Creates the issue, or returns its current status."},
  state:  {x:185, y:480, w:100, label:"State DB", type:"store", tag:"SQLite / Postgres",
           info:"Restores the conversation: pending question, ticket draft, known email."},
};
// [from, to, sideFrom, sideTo]  (r=right, l=left, t=top, b=bottom)
const EDGES = [
  ["emp","api","r","l"], ["api","router","r","l"], ["api","state","b","t"],
  ["router","kb","r","l"], ["router","chitchat","r","l"], ["router","start","r","l"],
  ["router","email","r","l"], ["router","create","r","l"], ["router","status","r","l"],
  ["router","cancel","r","l"], ["kb","offer","b","t"], ["kb","chroma","r","l"],
  ["create","jira","r","l"], ["status","jira","r","l"],
];
const PRE = ["emp","api","state","api","router"], POST = ["router","api","emp"];
const SCENARIOS = {
  "Answered from documents": {desc:"A question the documents can answer.",
    mid:["kb","chroma","kb"]},
  "Cannot answer: ticket offered": {desc:"No good passage found, so the agent offers a ticket instead of guessing.",
    mid:["kb","chroma","kb","offer","kb"]},
  "Yes + email: ticket created": {desc:"The employee confirmed and gave an email, so the ticket is written to Jira.",
    mid:["create","jira","create"]},
  "Ticket status": {desc:"The employee asks about a ticket they raised earlier.",
    mid:["status","jira","status"]},
  "Greeting": {desc:"Small talk skips retrieval entirely.", mid:["chitchat"]},
};

const NS = "http://www.w3.org/2000/svg";
const svg = document.getElementById("svg");
function el(name, attrs, parent) {
  const e = document.createElementNS(NS, name);
  for (const k in attrs) e.setAttribute(k, attrs[k]);
  (parent || svg).appendChild(e);
  return e;
}
function anchor(id, side) {
  const n = NODES[id], w = n.w || 120, h = 40;
  return side === "r" ? [n.x + w/2, n.y] : side === "l" ? [n.x - w/2, n.y]
       : side === "t" ? [n.x, n.y - h/2] : [n.x, n.y + h/2];
}
function pathD(a, b, sa, sb) {
  const [x1,y1] = anchor(a, sa), [x2,y2] = anchor(b, sb);
  if (sa === "r" || sa === "l") { const mx = (x1+x2)/2; return `M${x1},${y1} C${mx},${y1} ${mx},${y2} ${x2},${y2}`; }
  const my = (y1+y2)/2; return `M${x1},${y1} C${x1},${my} ${x2},${my} ${x2},${y2}`;
}

// ---- stats (computed from the same data that draws the diagram)
const agentNodes = Object.values(NODES).filter(n => n.agent);
const llmCount = agentNodes.filter(n => n.type === "llm").length;
const storeCount = Object.values(NODES).filter(n => n.type === "store").length;
[[AGENTS,"agent (one LangGraph graph)"],[agentNodes.length,"nodes in the agent"],
 [llmCount,"nodes that call an LLM"],[agentNodes.length-llmCount,"plain-code nodes"],
 [storeCount,"data stores / systems"]].forEach(([v,t]) => {
  const d = document.createElement("div"); d.className = "card";
  d.innerHTML = `<b>${v}</b><span>${t}</span>`;
  document.getElementById("stats").appendChild(d);
});

// ---- draw: boundary, edges, nodes, packet
el("rect", {id:"boundary", x:270, y:30, width:355, height:562, rx:14});
el("text", {x:447, y:22, style:"font-weight:600;fill:#7c3aed"}, null).textContent =
  `1 AGENT  -  ${agentNodes.length} nodes`;
EDGES.forEach(e => {
  const p = el("path", {class:"edge", d:pathD(...e)});
  e.el = p; e.len = p.getTotalLength();
});
for (const id in NODES) {
  const n = NODES[id], w = n.w || 120;
  const g = el("g", {class:"n-" + n.type});
  el("rect", {x:n.x - w/2, y:n.y - 20, width:w, height:40, rx:8}, g);
  el("text", {x:n.x, y:n.y - 2}, g).textContent = n.label;
  el("text", {x:n.x, y:n.y + 13, class:"tag"}, g).textContent = n.tag;
  n.g = g;
}
const dot = el("circle", {id:"dot", r:7, cx:0, cy:0});

// ---- animation
const caption = document.getElementById("caption");
const sel = document.getElementById("scn"), loop = document.getElementById("loop");
const reduce = matchMedia("(prefers-reduced-motion: reduce)").matches;
loop.checked = !reduce;
Object.keys(SCENARIOS).forEach(k => sel.add(new Option(k, k)));
let token = 0, cur = null;
const sleep = ms => new Promise(r => setTimeout(r, ms));

function edgeFor(a, b) {
  for (const e of EDGES) {
    if (e[0] === a && e[1] === b) return {e, fwd:true};
    if (e[0] === b && e[1] === a) return {e, fwd:false};
  }
  throw new Error("No edge between " + a + " and " + b);
}
function reset() {
  document.querySelectorAll(".active").forEach(x => x.classList.remove("active"));
  for (const id in NODES) NODES[id].g.classList.remove("current", "visited");
  cur = null;
}
function arrive(id, step, total) {
  if (cur) { cur.g.classList.remove("current"); cur.g.classList.add("visited"); }
  cur = NODES[id]; cur.g.classList.add("current");
  caption.textContent = `Step ${step}/${total} - ${cur.label}: ${cur.info}`;
}
function move(a, b, my) {
  const {e, fwd} = edgeFor(a, b);
  e.el.classList.add("active");
  const dur = Math.max(350, e.len / 0.3), t0 = performance.now();
  return new Promise(res => {
    function tick(now) {
      if (my !== token) return res();
      const p = Math.min(1, (now - t0) / dur);
      const pt = e.el.getPointAtLength((fwd ? p : 1 - p) * e.len);
      dot.setAttribute("transform", `translate(${pt.x},${pt.y})`);
      p < 1 ? requestAnimationFrame(tick) : res();
    }
    requestAnimationFrame(tick);
  });
}
async function play() {
  const my = ++token, sc = SCENARIOS[sel.value];
  const path = [...PRE, ...sc.mid, ...POST];
  reset();
  dot.setAttribute("transform", `translate(${NODES[path[0]].x},${NODES[path[0]].y})`);
  arrive(path[0], 1, path.length);
  caption.textContent = sc.desc + "  " + caption.textContent;
  await sleep(700);
  for (let i = 0; i < path.length - 1; i++) {
    if (my !== token) return;
    await move(path[i], path[i+1], my);
    if (my !== token) return;
    arrive(path[i+1], i + 2, path.length);
    await sleep(500);
  }
  if (loop.checked && my === token) { await sleep(1500); if (my === token) play(); }
}
document.getElementById("replay").onclick = play;
sel.onchange = play;
if (!reduce) play(); else { arrive("emp", 1, 1); caption.textContent = "Press Replay to see the flow."; }
</script>
"""