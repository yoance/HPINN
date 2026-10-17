/* H-PINN playground frontend. Vanilla JS, no dependencies. */

const $ = (id) => document.getElementById(id);
const tooltip = $("tooltip");

let problems = {};
let runs = [];          // [{key, label, color, request, job_id, job}]
let pollTimer = null;
let shownSolution = null;

const TERMINAL = ["completed", "failed", "cancelled"];

/* ------------------------------------------------------------------ */
/* Color ramps (sequential, light -> dark; from the reference palette) */
/* ------------------------------------------------------------------ */

const BLUE_RAMP = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"];
const AQUA_RAMP = ["#d9f7ea", "#a9ecd2", "#79ddb6", "#4cc898", "#1baf7a", "#107252", "#073c2c"];

function hexToRgb(h) {
  return [parseInt(h.slice(1, 3), 16), parseInt(h.slice(3, 5), 16), parseInt(h.slice(5, 7), 16)];
}

function rampColor(ramp, v) {
  // v in [0, 1] -> interpolated [r, g, b]
  const t = Math.min(1, Math.max(0, v)) * (ramp.length - 1);
  const i = Math.min(ramp.length - 2, Math.floor(t));
  const f = t - i;
  const a = hexToRgb(ramp[i]), b = hexToRgb(ramp[i + 1]);
  return [0, 1, 2].map((k) => Math.round(a[k] + (b[k] - a[k]) * f));
}

function cssVar(name) {
  return getComputedStyle(document.documentElement).getPropertyValue(name).trim();
}

/* ------------------------------------------------------------------ */
/* Runs: what gets trained, and how each one is labelled and colored.  */
/* Color follows the entity (method or scheduler), never the run order. */
/* ------------------------------------------------------------------ */

const METHODS = {
  forward: { label: "PINN", color: "--series-1" },
  pinn: { label: "PINN", color: "--series-1" },
  hpinn: { label: "H-PINN", color: "--series-2" },
};
const SCHEDULERS = {
  constant: { label: "Constant", color: "--series-1" },
  exponential: { label: "Exponential", color: "--series-2" },
  combined: { label: "CombinedDecay", color: "--series-3" },
};

function isHpinn(req) {
  return req.mode === "inverse" && req.method === "hpinn";
}

function planRuns(base, compare) {
  if (compare === "models" && base.mode === "inverse") {
    return ["pinn", "hpinn"].map((m) => ({
      key: m, ...METHODS[m], request: { ...base, method: m },
    }));
  }
  if (compare === "schedulers") {
    return Object.keys(SCHEDULERS).map((s) => ({
      key: s, ...SCHEDULERS[s], request: { ...base, lr_schedule: s },
    }));
  }
  const m = base.mode === "forward" ? "forward" : base.method;
  return [{ key: m, ...METHODS[m], request: base }];
}

/* ------------------------------------------------------------------ */
/* Learning-rate schedules — mirrors the backend (Keras ExponentialDecay */
/* and hpinn.CombinedDecay) so the preview matches what training uses.  */
/* ------------------------------------------------------------------ */

function lrAt(step, req) {
  const exp = (s) => req.learning_rate * Math.pow(req.decay_rate, s / req.decay_steps);
  if (req.lr_schedule === "constant") return req.learning_rate;
  if (req.lr_schedule === "exponential") return exp(step);
  const b = req.boundaries;
  if (step < b[0]) return exp(step);
  // values[0] = exp(b[0]); values[k+1] = values[k] / values_steps;
  // between boundary k-1 and k (exclusive, inclusive) -> values[k]
  const k = b.filter((v) => v < step).length;
  return exp(b[0]) / Math.pow(req.values_steps, k);
}

function lrSeries(req, maxPoints = 400) {
  // Each optimizer has its own step counter, so with H-PINN the schedule
  // restarts at the start of the λ-search phase.
  const phases = isHpinn(req) ? [req.epochs, req.param_epochs] : [req.epochs];
  const xs = [], ys = [];
  let offset = 0;
  const total = phases.reduce((a, b) => a + b, 0);
  const stride = Math.max(1, Math.floor(total / maxPoints));
  for (const n of phases) {
    const steps = new Set();
    for (let s = 0; s < n; s += stride) steps.add(s);
    steps.add(n - 1);
    if (req.lr_schedule === "combined") {
      // keep both sides of every jump sharp
      for (const bd of req.boundaries) {
        if (bd < n) { steps.add(bd); steps.add(Math.min(n - 1, bd + 1)); }
      }
    }
    for (const s of [...steps].sort((a, b) => a - b)) {
      xs.push(offset + s);
      ys.push(lrAt(s, req));
    }
    offset += n;
  }
  return { xs, ys };
}

/* ------------------------------------------------------------------ */
/* Form setup                                                          */
/* ------------------------------------------------------------------ */

async function loadProblems() {
  const res = await fetch("/api/problems");
  const list = await res.json();
  const sel = $("problem");
  sel.innerHTML = "";
  for (const p of list) {
    problems[p.key] = p;
    const opt = document.createElement("option");
    opt.value = p.key;
    opt.textContent = p.name;
    sel.appendChild(opt);
  }
  updateProblemUI();
}

function updateProblemUI() {
  const p = problems[$("problem").value];
  if (!p) return;
  $("problem-desc").textContent =
    `${p.description} Domain: t ∈ [${p.lb[0]}, ${p.ub[0]}], x ∈ [${p.lb[1]}, ${p.ub[1]}].`;
  const wrap = $("lambda-inputs");
  wrap.innerHTML = "";
  p.lambda_names.forEach((name, i) => {
    const label = document.createElement("label");
    label.append(`Initial ${name}`);
    const input = document.createElement("input");
    input.type = "number";
    input.step = "any";
    input.value = "0";
    input.dataset.lambdaIndex = i;
    label.appendChild(input);
    wrap.appendChild(label);
  });
}

const METHOD_HINTS = {
  pinn: "Trains the network weights and λ together on data misfit + PDE residual.",
  hpinn: "Phase 1 fits the network to the measurements only; phase 2 then searches λ " +
         "with the fitted network held fixed. Converges even from distant initial guesses.",
};
const COMPARE_HINTS = {
  none: "",
  models: "Trains PINN and H-PINN on the same data, seed and initial λ, one after the other.",
  schedulers: "Trains the same model three times, once per scheduler, sharing the settings above.",
};
const LR_HINTS = {
  constant: "The rate stays at its initial value.",
  exponential: "rate · decay_rate^(step / decay_steps)",
  combined: "Exponential decay until the first boundary, then the rate is divided at every boundary.",
};

function updateFormUI() {
  const inverse = $("mode").value === "inverse";
  const compareSel = $("compare");
  compareSel.querySelector('option[value="models"]').disabled = !inverse;
  if (!inverse && compareSel.value === "models") compareSel.value = "none";
  const compare = compareSel.value;

  $("forward-fields").hidden = inverse;
  $("inverse-fields").hidden = !inverse;
  $("method-field").hidden = compare === "models";
  $("method-hint").textContent = compare === "models"
    ? "Both methods are trained — see Compare below."
    : METHOD_HINTS[$("method").value];

  const hpinnUsed = inverse && (compare === "models" || $("method").value === "hpinn");
  $("param-epochs-field").hidden = !hpinnUsed;
  $("epochs-label").textContent = hpinnUsed ? "Epochs (data fit)" : "Epochs";

  const sched = $("lr_schedule").value;
  const schedCompare = compare === "schedulers";
  $("lr-schedule-field").hidden = schedCompare;
  document.querySelectorAll(".decay-field").forEach((n) => {
    n.hidden = !schedCompare && sched === "constant";
  });
  document.querySelectorAll(".combined-field").forEach((n) => {
    n.hidden = !schedCompare && sched !== "combined";
  });
  $("lr-hint").textContent = schedCompare
    ? "All three schedulers use the parameters above."
    : LR_HINTS[sched];
  $("compare-hint").textContent = COMPARE_HINTS[compare];

  renderLrPreview();
}

function parseBoundaries() {
  const parts = $("boundaries").value.split(/[\s,;]+/).filter(Boolean);
  const nums = parts.map(Number);
  if (!nums.length || nums.some((v) => !Number.isInteger(v) || v < 1)) return null;
  if (nums.some((v, i) => i && v <= nums[i - 1])) return null;
  return nums;
}

function buildRequest() {
  const req = {
    problem: $("problem").value,
    mode: $("mode").value,
    method: $("method").value,
    num_hidden_layers: +$("num_hidden_layers").value,
    num_neurons_per_layer: +$("num_neurons_per_layer").value,
    epochs: +$("epochs").value,
    param_epochs: +$("param_epochs").value,
    seed: +$("seed").value,
    learning_rate: +$("learning_rate").value,
    lr_schedule: $("lr_schedule").value,
    decay_steps: +$("decay_steps").value,
    decay_rate: +$("decay_rate").value,
    boundaries: parseBoundaries(),
    values_steps: +$("values_steps").value,
    n_collocation: +$("n_collocation").value,
    n_initial: +$("n_initial").value,
    n_boundary: +$("n_boundary").value,
    n_data: +$("n_data").value,
    noise: +$("noise").value,
  };
  if (req.mode === "inverse") {
    req.initial_lambda = [...$("lambda-inputs").querySelectorAll("input")].map((i) => +i.value);
  }
  return req;
}

function validate(base, planned) {
  const usesCombined = planned.some((r) => r.request.lr_schedule === "combined");
  if (usesCombined && !base.boundaries) {
    return "Boundaries must be positive whole numbers in increasing order, e.g. 1000, 1500.";
  }
  if (!(base.learning_rate > 0)) return "The initial learning rate must be positive.";
  if (!(base.decay_steps >= 1)) return "Decay steps must be at least 1.";
  if (!(base.decay_rate > 0 && base.decay_rate <= 1)) return "Decay rate must be in (0, 1].";
  if (!(base.values_steps >= 1)) return "“Divide by” must be at least 1.";
  return null;
}

function currentPlan() {
  const base = buildRequest();
  const planned = planRuns(base, $("compare").value);
  return { base, planned, error: validate(base, planned) };
}

function applyNotebookPreset() {
  // lr_comb_param from the H-PINN notebooks
  $("learning_rate").value = 0.1;
  $("decay_steps").value = 150;
  $("decay_rate").value = 0.9;
  $("boundaries").value = "3000, 7000, 14000";
  $("values_steps").value = 2;
  updateFormUI();
}

/* ------------------------------------------------------------------ */
/* Job lifecycle                                                       */
/* ------------------------------------------------------------------ */

async function startTraining(e) {
  e.preventDefault();
  const { planned, error } = currentPlan();
  if (error) return;

  if (pollTimer) clearInterval(pollTimer);
  runs = [];
  shownSolution = null;
  for (const run of planned) {
    const res = await fetch("/api/train", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(run.request),
    });
    if (!res.ok) {
      const detail = await res.json().catch(() => ({}));
      showFormError("Could not start training: " + formatDetail(detail.detail || res.status));
      await cancelTraining();
      return;
    }
    run.job_id = (await res.json()).job_id;
    run.job = null;
    runs.push(run);
  }

  $("empty-state").hidden = true;
  $("summary").hidden = true;
  $("solution-card").hidden = true;
  $("lambda-card").hidden = true;
  $("loss-card").hidden = true;
  $("status-card").hidden = false;
  $("train-btn").disabled = true;
  $("cancel-btn").hidden = false;
  poll();
  pollTimer = setInterval(poll, 700);
}

function formatDetail(d) {
  if (Array.isArray(d)) return d.map((x) => `${(x.loc || []).slice(1).join(".")}: ${x.msg}`).join("; ");
  return typeof d === "string" ? d : JSON.stringify(d);
}

function showFormError(msg) {
  const n = $("form-error");
  n.textContent = msg || "";
  n.hidden = !msg;
}

async function cancelTraining() {
  await Promise.all(runs
    .filter((r) => !r.job || !TERMINAL.includes(r.job.status))
    .map((r) => fetch(`/api/train/${r.job_id}`, { method: "DELETE" })));
}

async function poll() {
  const pending = runs.filter((r) => !r.job || !TERMINAL.includes(r.job.status));
  await Promise.all(pending.map(async (r) => {
    const res = await fetch(`/api/train/${r.job_id}`);
    if (res.ok) r.job = await res.json();
  }));
  const withJob = runs.filter((r) => r.job);
  if (!withJob.length) return;

  renderStatus();
  renderLossChart(withJob);
  renderLambdaCharts(withJob);
  renderSummary();
  renderSolution();

  if (runs.every((r) => r.job && TERMINAL.includes(r.job.status))) {
    clearInterval(pollTimer);
    pollTimer = null;
    $("train-btn").disabled = false;
    $("cancel-btn").hidden = true;
  }
}

const STATUS_ICONS = { queued: "⏳", running: "⚙", completed: "✓", failed: "✕", cancelled: "⊘" };

function phaseText(run) {
  const req = run.job.request;
  if (!isHpinn(req) || run.job.status !== "running") return "";
  return run.job.progress.iteration < req.epochs ? "phase 1 · data fit · " : "phase 2 · λ search · ";
}

function renderStatus() {
  const wrap = $("runs-status");
  wrap.replaceChildren();
  for (const run of runs) {
    const job = run.job;
    const row = document.createElement("div");
    row.className = "run-status";
    const status = job ? job.status : "queued";
    const it = job ? job.progress.iteration : 0;
    const total = job ? job.progress.total : 1;
    const last = job ? job.progress.loss_history.at(-1) : null;
    let text;
    if (status === "failed") text = job.error;
    else if (status === "completed") text = `${total.toLocaleString()} iterations` +
      (last != null ? ` · final loss ${last.toExponential(2)}` : "");
    else text = (job ? phaseText(run) : "") + `iteration ${it.toLocaleString()} / ${total.toLocaleString()}` +
      (last != null ? ` · loss ${last.toExponential(2)}` : "");
    const pct = status === "completed" ? 100 : Math.min(100, (it / total) * 100);
    row.innerHTML = `
      <div class="status-row">
        <span class="run-name"><span class="swatch" style="background:${cssVar(run.color)}"></span>${run.label}</span>
        <span class="status-chip ${status}">${STATUS_ICONS[status] || ""} ${status}</span>
        <span class="status-text"></span>
      </div>
      <div class="progress-track"><div class="progress-fill" style="width:${pct}%;background:${cssVar(run.color)}"></div></div>`;
    row.querySelector(".status-text").textContent = text;
    wrap.appendChild(row);
  }
}

/* ------------------------------------------------------------------ */
/* Summary: tiles for one run, a comparison table for several          */
/* ------------------------------------------------------------------ */

function lambdaErrorPct(pred, truth) {
  return Math.abs(pred - truth) / Math.abs(truth) * 100;
}

function renderSummary() {
  const done = runs.filter((r) => r.job && r.job.status === "completed");
  const wrap = $("summary");
  if (!done.length) { wrap.hidden = true; return; }
  wrap.hidden = false;

  const p = problems[runs[0].request.problem];
  if (runs.length === 1) {
    const r = done[0].job.result;
    const tiles = [
      ["Relative L2 error", r.l2_error_percent.toFixed(2) + " %", ""],
      ["Training time", r.elapsed_seconds.toFixed(1) + " s", ""],
    ];
    if (r.lambda_pred) {
      tiles.push(["Identified λ", r.lambda_pred.map((v) => v.toFixed(3)).join(", "),
        "true: " + r.lambda_names.map((n, i) => `${n} = ${r.lambda_star[i]}`).join(", ")]);
    }
    wrap.className = "tiles";
    wrap.innerHTML = tiles.map(([label, value, sub]) => `
      <div class="tile">
        <div class="tile-label">${label}</div>
        <div class="tile-value">${value}</div>
        ${sub ? `<div class="tile-sub">${sub}</div>` : ""}
      </div>`).join("");
    return;
  }

  const inverse = runs[0].request.mode === "inverse";
  const lamCols = inverse ? p.lambda_names : [];
  const best = (vals, i) => vals.every((v) => v == null) ? false : vals[i] === Math.min(...vals.filter((v) => v != null));
  const rows = runs.map((run) => {
    const r = run.job && run.job.status === "completed" ? run.job.result : null;
    return {
      run, r,
      l2: r ? r.l2_error_percent : null,
      lamErr: lamCols.map((_, k) => r && r.lambda_pred ? lambdaErrorPct(r.lambda_pred[k], p.lambda_star[k]) : null),
    };
  });
  const l2s = rows.map((x) => x.l2);
  const cell = (v, isBest, fmt) => v == null ? `<td class="num muted">—</td>`
    : `<td class="num${isBest ? " best" : ""}">${fmt(v)}${isBest ? ' <span class="best-mark" title="best">★</span>' : ""}</td>`;
  wrap.className = "panel table-card";
  wrap.innerHTML = `
    <h3>Comparison</h3>
    <div class="table-wrap"><table class="compare-table">
      <thead><tr>
        <th>Run</th><th class="num">Relative L2 error</th>
        ${lamCols.map((n, k) => `<th class="num">${n} (true ${p.lambda_star[k]})</th><th class="num">${n} error</th>`).join("")}
        <th class="num">Time</th>
      </tr></thead>
      <tbody>${rows.map((x, i) => `
        <tr>
          <td><span class="swatch" style="background:${cssVar(x.run.color)}"></span>${x.run.label}</td>
          ${cell(x.l2, best(l2s, i), (v) => v.toFixed(2) + " %")}
          ${lamCols.map((_, k) => {
            const errs = rows.map((y) => y.lamErr[k]);
            return cell(x.r && x.r.lambda_pred ? x.r.lambda_pred[k] : null, false, (v) => v.toFixed(4)) +
                   cell(x.lamErr[k], best(errs, i), (v) => v.toFixed(2) + " %");
          }).join("")}
          ${cell(x.r ? x.r.elapsed_seconds : null, false, (v) => v.toFixed(1) + " s")}
        </tr>`).join("")}
      </tbody>
    </table></div>
    <p class="hint">★ marks the lowest error in each column.${inverse ? " λ error is relative to the true value." : ""}</p>`;
}

/* ------------------------------------------------------------------ */
/* SVG line charts (x axis = training iteration)                      */
/* ------------------------------------------------------------------ */

const MARGIN = { top: 12, right: 64, bottom: 26, left: 56 };

function el(name, attrs, parent) {
  const n = document.createElementNS("http://www.w3.org/2000/svg", name);
  for (const [k, v] of Object.entries(attrs)) n.setAttribute(k, v);
  if (parent) parent.appendChild(n);
  return n;
}

function niceTicks(min, max, n = 5) {
  if (min === max) { min -= 1; max += 1; }
  const span = max - min;
  const step = Math.pow(10, Math.floor(Math.log10(span / n)));
  const err = span / n / step;
  const mult = err >= 7.5 ? 10 : err >= 3.5 ? 5 : err >= 1.5 ? 2 : 1;
  const s = step * mult;
  const ticks = [];
  for (let v = Math.ceil(min / s) * s; v <= max + 1e-12; v += s) ticks.push(v);
  return ticks;
}

function fmtNum(v) {
  const a = Math.abs(v);
  if (a < 1e-9) return "0";
  if (a >= 10000 || a < 0.01) return v.toExponential(0);
  return +v.toFixed(3) + "";
}

function fmtIter(v) {
  return v >= 10000 ? (v / 1000) + "k" : v.toLocaleString();
}

/**
 * Draw a line chart.
 * opts: {
 *   series: [{xs, ys, color, label, breakAt?}],  // xs in iterations
 *   xMax, log (bool), refs: [{y, label}], vlines: [{x, label}],
 *   endLabels (bool), margin, height
 * }
 */
function lineChart(svg, opts) {
  const m = opts.margin || MARGIN;
  const W = svg.clientWidth || 640, H = svg.clientHeight || opts.height || 220;
  svg.setAttribute("viewBox", `0 0 ${W} ${H}`);
  svg.replaceChildren();
  const iw = W - m.left - m.right, ih = H - m.top - m.bottom;
  const series = opts.series.filter((s) => s.xs.length);
  if (!series.length) return;

  const tf = opts.log ? (v) => Math.log10(Math.max(v, 1e-12)) : (v) => v;
  const all = series.flatMap((s) => s.ys.map(tf)).concat((opts.refs || []).map((r) => tf(r.y)));
  let lo = Math.min(...all), hi = Math.max(...all);
  let yTicks;
  if (opts.log) {
    lo = Math.floor(lo); hi = Math.ceil(hi);
    if (lo === hi) hi = lo + 1;
    const step = Math.max(1, Math.ceil((hi - lo) / 5));
    yTicks = [];
    for (let d = lo; d <= hi; d += step) yTicks.push(d);
  } else {
    const pad = (hi - lo || 1) * 0.12;
    lo -= pad; hi += pad;
    yTicks = niceTicks(lo, hi, 5);
  }
  const xMax = Math.max(1, opts.xMax);
  const X = (x) => (x / xMax) * iw;
  const Y = (v) => ih - ((tf(v) - lo) / (hi - lo)) * ih;
  const Yt = (t) => ih - ((t - lo) / (hi - lo)) * ih;

  const g = el("g", { transform: `translate(${m.left},${m.top})` }, svg);
  for (const t of yTicks) {
    el("line", { x1: 0, x2: iw, y1: Yt(t), y2: Yt(t), stroke: cssVar("--grid"), "stroke-width": 1 }, g);
    el("text", { x: -8, y: Yt(t) + 3.5, "text-anchor": "end", "font-size": 11, fill: cssVar("--muted") }, g)
      .textContent = opts.log ? `1e${t}` : fmtNum(t);
  }
  el("line", { x1: 0, x2: iw, y1: ih, y2: ih, stroke: cssVar("--baseline"), "stroke-width": 1 }, g);
  for (const v of niceTicks(0, xMax, opts.xTicks || 6)) {
    el("text", { x: X(v), y: ih + 16, "text-anchor": "middle", "font-size": 11, fill: cssVar("--muted") }, g)
      .textContent = fmtIter(v);
  }
  for (const r of opts.refs || []) {
    el("line", { x1: 0, x2: iw, y1: Y(r.y), y2: Y(r.y), stroke: cssVar("--muted"), "stroke-width": 1, "stroke-dasharray": "5,4" }, g);
    el("text", { x: iw + 6, y: Y(r.y) + 3.5, "font-size": 11, fill: cssVar("--muted") }, g).textContent = r.label;
  }
  for (const v of opts.vlines || []) {
    el("line", { x1: X(v.x), x2: X(v.x), y1: 0, y2: ih, stroke: cssVar("--baseline"), "stroke-width": 1, "stroke-dasharray": "4,4" }, g);
    el("text", { x: X(v.x) + 4, y: ih - 6, "font-size": 11, fill: cssVar("--muted") }, g).textContent = v.label;
  }
  for (const s of series) {
    // ``breakAt``: start a new segment there instead of joining across it
    const d = s.xs.map((x, i) => {
      const cmd = i && !(s.breakAt != null && s.xs[i - 1] < s.breakAt && x >= s.breakAt) ? "L" : "M";
      return `${cmd}${X(x).toFixed(1)},${Y(s.ys[i]).toFixed(1)}`;
    }).join("");
    el("path", { d, fill: "none", stroke: s.color, "stroke-width": 2, "stroke-linejoin": "round" }, g);
  }
  if (opts.endLabels) {
    // Direct labels at line ends, nudged apart so they never overlap.
    const labels = series.map((s) => ({ s, y: Y(s.ys.at(-1)), x: X(s.xs.at(-1)) }))
      .sort((a, b) => a.y - b.y);
    for (let i = 1; i < labels.length; i++) {
      if (labels[i].y - labels[i - 1].y < 13) labels[i].y = labels[i - 1].y + 13;
    }
    for (const l of labels) {
      el("text", { x: Math.min(l.x, iw) - 4, y: l.y - 6, "text-anchor": "end", "font-size": 11, "font-weight": 600, fill: cssVar("--text-secondary") }, g)
        .textContent = `${l.s.label} ${fmtNum(l.s.ys.at(-1))}`;
    }
  }
  attachHover(g, iw, ih, series, X, Y, xMax, opts.tooltipValue || ((v) => fmtNum(v)));
}

function nearestIndex(xs, x) {
  let best = 0;
  for (let i = 1; i < xs.length; i++) if (Math.abs(xs[i] - x) < Math.abs(xs[best] - x)) best = i;
  return best;
}

function attachHover(g, iw, ih, series, X, Y, xMax, fmtVal) {
  const hover = el("rect", { x: 0, y: 0, width: iw, height: ih, fill: "transparent" }, g);
  const cross = el("line", { y1: 0, y2: ih, stroke: cssVar("--muted"), "stroke-width": 1, "stroke-dasharray": "3,3", visibility: "hidden" }, g);
  const dots = series.map((s) =>
    el("circle", { r: 4, fill: s.color, stroke: cssVar("--surface-1"), "stroke-width": 2, visibility: "hidden" }, g));
  hover.addEventListener("mousemove", (ev) => {
    const rect = hover.getBoundingClientRect();
    const iter = ((ev.clientX - rect.left) / rect.width) * xMax;
    cross.setAttribute("x1", X(iter)); cross.setAttribute("x2", X(iter));
    cross.setAttribute("visibility", "visible");
    const lines = [`<span class="tt-muted">iter</span> ${Math.round(iter).toLocaleString()}`];
    series.forEach((s, i) => {
      const inRange = iter >= s.xs[0] - xMax * 0.01 && iter <= s.xs.at(-1) + xMax * 0.01;
      if (!inRange) { dots[i].setAttribute("visibility", "hidden"); return; }
      const j = nearestIndex(s.xs, iter);
      dots[i].setAttribute("cx", X(s.xs[j]));
      dots[i].setAttribute("cy", Y(s.ys[j]));
      dots[i].setAttribute("visibility", "visible");
      lines.push(`<span class="swatch" style="background:${s.color}"></span><span class="tt-muted">${s.label}</span> ${fmtVal(s.ys[j])}`);
    });
    tooltip.innerHTML = lines.join("<br>");
    tooltip.hidden = false;
    tooltip.style.left = Math.min(ev.clientX + 14, window.innerWidth - tooltip.offsetWidth - 8) + "px";
    tooltip.style.top = ev.clientY + 14 + "px";
  });
  hover.addEventListener("mouseleave", () => {
    tooltip.hidden = true;
    cross.setAttribute("visibility", "hidden");
    dots.forEach((d) => d.setAttribute("visibility", "hidden"));
  });
}

function legendHtml(items, extra = "") {
  return items.map((it) =>
    `<span><span class="swatch" style="background:${it.color}"></span>${it.label}</span>`).join("") + extra;
}

function chartXMax(withJob) {
  return Math.max(...withJob.map((r) => r.job.progress.total));
}

function phaseLines(withJob) {
  const hp = withJob.find((r) => isHpinn(r.job.request));
  return hp ? [{ x: hp.job.request.epochs, label: "λ search →" }] : [];
}

function renderLossChart(withJob) {
  const series = withJob
    .filter((r) => r.job.progress.loss_history.length > 1)
    .map((r) => {
      const h = r.job.progress.loss_history;
      return {
        xs: r.job.progress.loss_iterations, ys: h, color: cssVar(r.color), label: r.label,
        breakAt: isHpinn(r.job.request) ? r.job.request.epochs : null,  // loss changes meaning
      };
    });
  if (!series.length) return;
  $("loss-card").hidden = false;
  $("loss-legend").innerHTML = runs.length > 1 ? legendHtml(series) : "";
  const vlines = phaseLines(withJob);
  $("loss-hint").hidden = !vlines.length;
  lineChart($("loss-chart"), {
    series, xMax: chartXMax(withJob), log: true, vlines,
    tooltipValue: (v) => v.toExponential(3),
  });
}

function renderLambdaCharts(withJob) {
  const inv = withJob.filter((r) => r.job.request.mode === "inverse");
  const withHist = inv.filter((r) => r.job.progress.lambda_history.length > 1);
  if (!withHist.length) return;
  $("lambda-card").hidden = false;
  const p = problems[inv[0].job.request.problem];
  const xMax = chartXMax(withJob);
  const vlines = phaseLines(withJob);

  const wrap = $("lambda-charts");
  if (wrap.children.length !== p.lambda_names.length) {
    wrap.replaceChildren(...p.lambda_names.map((n) => {
      const fig = document.createElement("figure");
      fig.innerHTML = `<figcaption>${n}</figcaption><svg></svg>`;
      return fig;
    }));
  }
  p.lambda_names.forEach((name, k) => {
    const series = withHist.map((r) => {
      const h = r.job.progress.lambda_history;
      return {
        xs: r.job.progress.lambda_iterations,
        ys: h.map((row) => row[k]), color: cssVar(r.color), label: r.label,
      };
    });
    lineChart(wrap.children[k].querySelector("svg"), {
      series, xMax, refs: [{ y: p.lambda_star[k], label: `${name}* = ${p.lambda_star[k]}` }],
      vlines, endLabels: true,
    });
  });
  $("lambda-legend").innerHTML = legendHtml(
    withHist.map((r) => ({ color: cssVar(r.color), label: `${r.label} estimate` })),
    `<span><span class="swatch dashed"></span>true value</span>`);
}

function renderLrPreview() {
  const { planned, error } = currentPlan();
  const svg = $("lr-chart");
  if (error) {
    svg.replaceChildren();
    $("lr-legend").innerHTML = "";
    showFormError(error);
    $("train-btn").disabled = true;
    return;
  }
  showFormError(null);
  if (!pollTimer) $("train-btn").disabled = false;
  const series = planned.map((r) => ({ ...lrSeries(r.request), color: cssVar(r.color), label: r.label }));
  $("lr-legend").innerHTML = planned.length > 1 ? legendHtml(series) : "";
  const hp = planned.find((r) => isHpinn(r.request));
  lineChart(svg, {
    series, log: true, height: 130, xTicks: 4,
    xMax: Math.max(...series.map((s) => s.xs.at(-1) + 1)),
    vlines: hp ? [{ x: hp.request.epochs, label: "restart" }] : [],
    margin: { top: 10, right: 12, bottom: 24, left: 40 },
    tooltipValue: (v) => v.toExponential(2),
  });
}

/* ------------------------------------------------------------------ */
/* Heatmaps                                                            */
/* ------------------------------------------------------------------ */

function drawHeatmap(canvas, grid, vmin, vmax, ramp, sol, label) {
  const nx = grid.length, nt = grid[0].length;
  canvas.width = nt;
  canvas.height = nx;
  const ctx = canvas.getContext("2d");
  const img = ctx.createImageData(nt, nx);
  const span = vmax - vmin || 1;
  for (let ix = 0; ix < nx; ix++) {
    const row = nx - 1 - ix; // x increases upward
    for (let it = 0; it < nt; it++) {
      const [r, gc, b] = rampColor(ramp, (grid[ix][it] - vmin) / span);
      const o = (row * nt + it) * 4;
      img.data[o] = r; img.data[o + 1] = gc; img.data[o + 2] = b; img.data[o + 3] = 255;
    }
  }
  ctx.putImageData(img, 0, 0);

  canvas.onmousemove = (ev) => {
    const rect = canvas.getBoundingClientRect();
    const it = Math.min(nt - 1, Math.max(0, Math.floor(((ev.clientX - rect.left) / rect.width) * nt)));
    const ix = Math.min(nx - 1, Math.max(0, nx - 1 - Math.floor(((ev.clientY - rect.top) / rect.height) * nx)));
    tooltip.innerHTML =
      `<span class="tt-muted">t</span> ${sol.t[it].toFixed(2)} · ` +
      `<span class="tt-muted">x</span> ${sol.x[ix].toFixed(2)}<br>` +
      `<span class="tt-muted">${label}</span> ${grid[ix][it].toFixed(4)}`;
    tooltip.hidden = false;
    tooltip.style.left = Math.min(ev.clientX + 14, window.innerWidth - tooltip.offsetWidth - 8) + "px";
    tooltip.style.top = ev.clientY + 14 + "px";
  };
  canvas.onmouseleave = () => { tooltip.hidden = true; };
}

function drawColorbar(canvas, ramp) {
  const ctx = canvas.getContext("2d");
  for (let px = 0; px < canvas.width; px++) {
    const [r, g, b] = rampColor(ramp, px / (canvas.width - 1));
    ctx.fillStyle = `rgb(${r},${g},${b})`;
    ctx.fillRect(px, 0, 1, canvas.height);
  }
}

function renderSolution() {
  const done = runs.filter((r) => r.job && r.job.status === "completed");
  if (!done.length) return;
  if (!done.some((r) => r.key === shownSolution)) shownSolution = done[0].key;
  $("solution-card").hidden = false;

  const tabs = $("solution-runs");
  tabs.hidden = runs.length < 2;
  tabs.replaceChildren(...runs.map((r) => {
    const b = document.createElement("button");
    b.type = "button";
    b.setAttribute("role", "tab");
    b.innerHTML = `<span class="swatch" style="background:${cssVar(r.color)}"></span>${r.label}`;
    b.disabled = !done.includes(r);
    b.setAttribute("aria-selected", r.key === shownSolution);
    b.onclick = () => { shownSolution = r.key; renderSolution(); };
    return b;
  }));

  // Shared scales so runs are compared on the same colors.
  const sols = done.map((r) => r.job.result.solution);
  const uAll = sols.flatMap((s) => s.u_pred.flat()).concat(sols[0].u_exact.flat());
  const uMin = Math.min(...uAll), uMax = Math.max(...uAll);
  const eMax = Math.max(...sols.flatMap((s) => s.abs_error.flat())) || 1;

  const run = done.find((r) => r.key === shownSolution);
  const sol = run.job.result.solution;
  $("hm-pred-caption").textContent = `${run.label} prediction`;
  drawHeatmap($("hm-pred"), sol.u_pred, uMin, uMax, BLUE_RAMP, sol, "u_pred");
  drawHeatmap($("hm-exact"), sol.u_exact, uMin, uMax, BLUE_RAMP, sol, "u_exact");
  drawHeatmap($("hm-error"), sol.abs_error, 0, eMax, AQUA_RAMP, sol, "|error|");

  drawColorbar($("cb-u"), BLUE_RAMP);
  drawColorbar($("cb-e"), AQUA_RAMP);
  $("cb-u-min").textContent = fmtNum(uMin);
  $("cb-u-max").textContent = fmtNum(uMax);
  $("cb-e-min").textContent = "0";
  $("cb-e-max").textContent = fmtNum(eMax);
}

/* ------------------------------------------------------------------ */

$("train-form").addEventListener("submit", startTraining);
$("train-form").addEventListener("input", updateFormUI);
$("train-form").addEventListener("change", updateFormUI);
$("cancel-btn").addEventListener("click", cancelTraining);
$("problem").addEventListener("change", updateProblemUI);
$("preset-btn").addEventListener("click", applyNotebookPreset);
window.addEventListener("resize", () => {
  renderLrPreview();
  const withJob = runs.filter((r) => r.job);
  if (withJob.length) { renderLossChart(withJob); renderLambdaCharts(withJob); }
});
updateFormUI();
loadProblems().catch(() => {
  $("problem-desc").textContent = "Could not reach the API — is the backend running?";
});
