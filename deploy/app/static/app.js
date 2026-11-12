const H = { "X-Requested-With": "bwm" };
const $ = (id) => document.getElementById(id);
let lastReport = null;

function cell(tr, text, cls) {
  const td = document.createElement("td");
  if (cls) { const s = document.createElement("span"); s.className = cls; s.textContent = text; td.appendChild(s); }
  else td.textContent = text;
  tr.appendChild(td);
}

async function me() {
  const r = await fetch("/api/me", { headers: H });
  if (r.status === 401) { location.href = "/login"; return; }
  const j = await r.json();
  $("who").textContent = `${j.user} (${j.role}) · model ${j.model || "?"}`;
}

$("logout").onclick = async () => { await fetch("/logout", { method: "POST", headers: H }); location.href = "/login"; };

$("form").onsubmit = async (ev) => {
  ev.preventDefault();
  $("error").classList.add("hidden");
  $("go").disabled = true;
  const t0 = Date.now();
  const tick = setInterval(() => { $("status").textContent = `Processing… ${Math.round((Date.now() - t0) / 1000)} s (usually 1–2 minutes)`; }, 1000);
  try {
    const r = await fetch("/api/analyze", { method: "POST", headers: H, body: new FormData($("form")) });
    const j = await r.json();
    if (!r.ok) throw new Error(j.detail || `error ${r.status}`);
    show(j);
    $("status").textContent = `Done in ${j.seconds} s`;
  } catch (e) {
    $("error").textContent = e.message;
    $("error").classList.remove("hidden");
    $("status").textContent = "";
  } finally {
    clearInterval(tick);
    $("go").disabled = false;
  }
};

function show(j) {
  lastReport = j;
  $("pred").innerHTML = "";
  $("pred").append(j.prediction);
  const sm = document.createElement("small");
  sm.textContent = { CN: "cognitively normal", MCI: "mild cognitive impairment", AD: "Alzheimer’s disease" }[j.prediction] || "";
  $("pred").appendChild(sm);
  $("bars").innerHTML = "";
  for (const [k, v] of Object.entries(j.probabilities)) {
    const row = document.createElement("div"); row.className = "bar";
    const a = document.createElement("span"); a.textContent = k;
    const t = document.createElement("div"); t.className = "track";
    const f = document.createElement("div"); f.className = "fill"; f.style.width = `${(v * 100).toFixed(1)}%`;
    t.appendChild(f);
    const b = document.createElement("span"); b.textContent = `${(v * 100).toFixed(0)}%`;
    row.append(a, t, b); $("bars").appendChild(row);
  }
  $("fig").src = j.figure;
  const words = { ok: "OK", warn: "Warning", fail: "Failed" };
  $("checks").innerHTML = "";
  for (const c of j.checks) {
    const tr = document.createElement("tr");
    cell(tr, c.check); cell(tr, words[c.status] || c.status, `tag ${c.status}`); cell(tr, c.detail);
    $("checks").appendChild(tr);
  }
  $("forecast").innerHTML = "";
  for (const f of j.forecast) {
    const tr = document.createElement("tr");
    cell(tr, `+${f.years} y`); cell(tr, f.predicted_atrophy.toFixed(4)); cell(tr, f.uncertainty.toFixed(3));
    $("forecast").appendChild(tr);
  }
  $("bage").textContent = `${j.brain_age} years`;
  $("input").innerHTML = "";
  const inp = Object.assign({}, j.input, { "file SHA-256": j.sha256, "report id": j.id });
  for (const [k, v] of Object.entries(inp)) {
    const tr = document.createElement("tr"); cell(tr, k); cell(tr, v === null || v === undefined ? "—" : String(v));
    $("input").appendChild(tr);
  }
  $("sig-out").textContent = `HMAC-SHA256: ${j.signature}`;
  for (const id of ["result", "details", "report"]) $(id).classList.remove("hidden");
}

$("download").onclick = () => {
  if (!lastReport) return;
  const blob = new Blob([JSON.stringify(lastReport, null, 1)], { type: "application/json" });
  const a = document.createElement("a");
  a.href = URL.createObjectURL(blob); a.download = `report-${lastReport.id}.json`; a.click();
};

$("verify-file").onchange = async (ev) => {
  const file = ev.target.files[0]; if (!file) return;
  try {
    const body = await file.text();
    const r = await fetch("/api/verify_report", { method: "POST", headers: Object.assign({ "Content-Type": "application/json" }, H), body });
    const j = await r.json();
    $("verify-status").textContent = j.valid ? "Valid: the report has not been changed." : "INVALID: the report was altered or not issued by this server.";
  } catch (e) { $("verify-status").textContent = "Could not read that file."; }
};

me();
