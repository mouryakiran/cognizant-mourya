"use strict";

const $ = (id) => document.getElementById(id);
const fmt = (v, d = 0) =>
  Number(v).toLocaleString("en-US", { maximumFractionDigits: d, minimumFractionDigits: 0 });
const money = (v) => "$" + fmt(v, 0);

let charts = {};

function toast(msg) {
  const t = $("toast");
  t.textContent = msg;
  t.classList.remove("hidden");
  setTimeout(() => t.classList.add("hidden"), 4000);
}

function makeChart(id, config) {
  if (charts[id]) charts[id].destroy();
  charts[id] = new Chart($(id), config);
}

function initCharts() {
  makeChart("forecastChart", {
    type: "line",
    data: { labels: [], datasets: [] },
    options: {
      responsive: true, maintainAspectRatio: false,
      interaction: { mode: "index", intersect: false },
      plugins: { legend: { labels: { color: "#dbe7f1" } } },
      scales: {
        x: { ticks: { color: "#8fa3b5", maxTicksLimit: 12 }, grid: { color: "#22303d" } },
        y: { ticks: { color: "#8fa3b5" }, grid: { color: "#22303d" } },
      },
    },
  });
  makeChart("optimChart", {
    type: "line",
    data: { labels: [], datasets: [] },
    options: {
      responsive: true, maintainAspectRatio: false,
      interaction: { mode: "index", intersect: false },
      plugins: { legend: { display: false } },
      scales: {
        x: { ticks: { color: "#8fa3b5", maxTicksLimit: 12 }, grid: { color: "#22303d" } },
        y: { ticks: { color: "#8fa3b5" }, grid: { color: "#22303d" } },
      },
    },
  });
}

async function loadRegions() {
  const r = await fetch("/api/regions");
  const data = await r.json();
  const sel = $("region");
  sel.innerHTML = "";
  (data.regions || []).forEach((reg) => {
    const o = document.createElement("option");
    o.value = reg;
    o.textContent = reg;
    sel.appendChild(o);
  });
  if (!data.regions.length) toast("No regions available");
  const model = data.model || {};
  renderModelCard(model);
}

function renderModelCard(m) {
  const el = $("model-card");
  if (!m || !m.mae_mw) {
    el.innerHTML = `<div class="center">Model metrics unavailable - run <span class="mono">python scripts/train.py</span></div>`;
    return;
  }
  el.innerHTML = `
    <div class="metric-row"><span class="m">MAE</span><span class="mono">${fmt(m.mae_mw, 2)} MW</span></div>
    <div class="metric-row"><span class="m">RMSE</span><span class="mono">${fmt(m.rmse_mw, 2)} MW</span></div>
    <div class="metric-row"><span class="m">MAPE</span><span class="mono">${fmt(m.mape_pct, 2)} %</span></div>
    <div class="metric-row"><span class="m">Test rows</span><span class="mono">${fmt(m.test_rows, 0)}</span></div>
    <div class="metric-row"><span class="m">Train rows</span><span class="mono">${fmt(m.train_rows, 0)}</span></div>
    <div class="metric-row"><span class="m">Trained at</span><span class="mono">${String(m.trained_at || "").slice(0, 16)}</span></div>
  `;
}

async function loadHistoryAndForecast() {
  const region = $("region").value;
  const horizon = +$("horizon").value;
  const site = $("site").value;
  const ssm = site ? +site : null;

  const [histRes, fcRes] = await Promise.all([
    fetch(`/api/history?region=${region}&days=7`),
    fetch(`/api/forecast?region=${region}&horizon=${horizon}&site_scale_mw=${ssm || ""}`),
  ]);
  const hist = await histRes.json();
  const fc = await fcRes.json();
  if (!fc.forecast || !fc.forecast.length) throw new Error("empty forecast");

  const labels = [...hist.points.map((p) => p.datetime.slice(5, 16)),
                  ...fc.forecast.map((p) => p.datetime.slice(5, 16))];
  const historyVals = hist.points.map((p) => p.consumption_mw);
  const histPad = labels.slice(0, labels.length - fc.forecast.length);
  const fcVals = fc.forecast.map((p) => p.forecast_mw);
  const upper = fc.forecast.map((p) => p.upper_mw);
  const lower = fc.forecast.map((p) => p.lower_mw);

  const pad = (arr, n) => Array(n).fill(null).concat(arr);

  makeChart("forecastChart", {
    type: "line",
    data: {
      labels,
      datasets: [
        { label: "History (MW)", data: [...historyVals, ...Array(fcVals.length).fill(null)],
          borderColor: "#38bdf8", backgroundColor: "#38bdf8", borderWidth: 2, pointRadius: 0, spanGaps: false },
        { label: "Upper (95%)", data: [...Array(histVals.length).fill(null), ...upper],
          borderColor: "rgba(45,212,191,0.25)", borderDash: [4, 4], pointRadius: 0, fill: false },
        { label: "Forecast (MW)", data: [...Array(historyVals.length).fill(null), ...fcVals],
          borderColor: "#2dd4bf", backgroundColor: "#2dd4bf", borderWidth: 2, pointRadius: 0,
          fill: { target: "lower", above: "rgba(45,212,191,0.08)" } },
        { label: "Lower (5%)", data: [...Array(historyVals.length).fill(null), ...lower],
          borderColor: "rgba(45,212,191,0.25)", borderDash: [4, 4], pointRadius: 0, fill: false },
      ],
    },
    options: {
      responsive: true, maintainAspectRatio: false,
      interaction: { mode: "index", intersect: false },
      plugins: { legend: { labels: { color: "#dbe7f1" } } },
      scales: {
        x: { ticks: { color: "#8fa3b5", maxTicksLimit: 12 }, grid: { color: "#22303d" } },
        y: { ticks: { color: "#8fa3b5" }, grid: { color: "#22303d" } },
      },
    },
  });
  $("fc-sub").textContent = `${fc.region} &middot; anchor ${fc.anchor.slice(0, 16)} &middot; forecast RMSE ${fc.model_rmse_mw} MW`;

  await loadOptimization(region, horizon, ssm);
  await loadRecommendations(region, horizon, ssm);
  return fc;
}

function renderKpis(fc, opt) {
  const fcVals = fc.forecast.map((p) => p.forecast_mw);
  const peak = Math.max(...fcVals);
  const avg = fcVals.reduce((a, b) => a + b, 0) / fcVals.length;
  const energy = fcVals.reduce((a, b) => a + b, 0); // MWh over horizon
  const s = opt ? opt.summary : null;
  const kpis = [
    { k: "Forecast peak", v: fmt(peak, 0) + " MW",
      d: s ? `baseline ${fmt(s.baseline.peak_mw, 0)} MW` : "" },
    { k: "Avg demand", v: fmt(avg, 0) + " MW", d: `${fc.horizon}h horizon` },
    { k: "Energy", v: fmt(energy, 0) + " MWh", d: "over horizon" },
    { k: "Baseline cost", v: s ? money(s.baseline.total_cost_usd) : "--", d: "energy + demand" },
    { k: "Optimised cost", v: s ? money(s.optimized.total_cost_usd) : "--", d: "" },
    { k: "Savings", v: s ? money(s.savings_usd) : "--",
      d: s ? `${fmt(s.savings_pct, 2)}% &middot; peak -${fmt(s.peak_reduction_pct, 2)}%` : "" },
  ];
  $("kpis").innerHTML = kpis.map(
    (k) => `<div class="kpi"><div class="k">${k.k}</div><div class="v">${k.v}</div><div class="delta">${k.d}</div></div>`
  ).join("");
}

let lastOpt = null;

async function loadOptimization(region, horizon, ssm) {
  const res = await fetch(`/api/optimize?region=${region}&horizon=${horizon}&site_scale_mw=${ssm || ""}`);
  const opt = await res.json();
  lastOpt = opt;
  const sched = opt.schedule || [];
  const labels = sched.map((s) => s.datetime.slice(5, 16));
  makeChart("optimChart", {
    type: "line",
    data: {
      labels,
      datasets: [
        { label: "Baseline", data: sched.map((s) => s.forecast_mw), borderColor: "#38bdf8", borderWidth: 2, pointRadius: 0 },
        { label: "Optimised net", data: sched.map((s) => s.net_mw), borderColor: "#2dd4bf", borderWidth: 2, pointRadius: 0 },
        { label: "Charge", data: sched.map((s) => s.charge_mw), borderColor: "#f59e0b", borderWidth: 1.5, pointRadius: 0,
          borderDash: [5, 4], yAxisID: "y2" },
        { label: "Discharge", data: sched.map((s) => s.discharge_mw), borderColor: "#f43f5e", borderWidth: 1.5,
          pointRadius: 0, borderDash: [5, 4], yAxisID: "y2" },
      ],
    },
    options: {
      responsive: true, maintainAspectRatio: false,
      interaction: { mode: "index", intersect: false },
      plugins: { legend: { labels: { color: "#dbe7f1" } } },
      scales: {
        x: { ticks: { color: "#8fa3b5" }, grid: { color: "#22303d" } },
        y: { ticks: { color: "#8fa3b5" }, grid: { color: "#22303d" } },
        y2: { position: "right", ticks: { color: "#8fa3b5" }, grid: { display: false } },
      },
    },
  });
  const s = opt.summary;
  $("optim-summary").innerHTML = `
    <div class="metric-row"><span class="m">Battery</span>
      <span class="mono">${opt.battery.capacity_mwh} MWh &middot; SOC ${opt.battery.initial_soc * 100}% → ${opt.battery.final_soc * 100}%</span></div>
    <div class="metric-row"><span class="m">Baseline cost</span><span class="mono">${money(s.baseline.total_cost_usd)}</span></div>
    <div class="metric-row"><span class="m">Optimised cost</span><span class="mono">${money(s.optimized.total_cost_usd)}</span></div>
    <div class="metric-row"><span class="m">Savings</span><span class="mono" style="color:var(--good)">${money(s.savings_usd)} (${fmt(s.savings_pct, 2)}%)</span></div>
    <div class="metric-row"><span class="m">Peak reduction</span><span class="mono" style="color:var(--good)">${fmt(s.peak_reduction_pct, 2)}%</span></div>
  `;
  renderKpis({ forecast: opt.schedule.map((x) => Object.assign({}, x)), horizon: opt.horizon }, opt);
}

async function loadRecommendations(region, horizon, ssm) {
  const res = await fetch(`/api/recommendations?region=${region}&horizon=${horizon}&site_scale_mw=${ssm || ""}`);
  const data = await res.json();
  const box = $("recs");
  const recs = data.recommendations || [];
  if (!recs.length) { box.innerHTML = `<div class="center">No recommendations</div>`; return; }
  box.innerHTML = recs.map((r, i) => `
    <div class="rec p-${r.priority}">
      <div class="r-title">${escapeHtml(r.title)} <span class="badge ${r.priority}">${r.priority}</span></div>
      <div class="r-meta">${escapeHtml(r.category)} &middot; confidence ${escapeHtml(r.confidence)}</div>
      <div class="r-desc">${escapeHtml(r.description)}</div>
      ${r.steps && r.steps.length ? `<ul>${r.steps.map((s) => `<li>${escapeHtml(s)}</li>`).join("")}</ul>` : ""}
      ${r.potential_usd ? `<div class="r-meta">Potential savings: <b style="color:var(--good)">${money(r.potential_usd)}</b></div>` : ""}
    </div>`).join("");
}

async function loadAnomalies() {
  const res = await fetch("/api/anomalies?limit=25");
  const data = await res.json();
  const anoms = data.anomalies || [];
  const box = $("anomalies");
  if (!anoms.length) { box.innerHTML = `<div class="center">No anomalies detected in the last 7 days</div>`; return; }
  box.innerHTML = `<table>
    <tr><th>Time</th><th>Region</th><th>Load (MW)</th><th>Dev %</th><th>Reason</th></tr>
    ${anoms.slice(0, 12).map((a) => `
      <tr>
        <td class="mono">${escapeHtml(a.timestamp.slice(0, 16))}</td>
        <td>${escapeHtml(a.region)}</td>
        <td class="mono">${fmt(a.consumption_mw, 0)}</td>
        <td class="mono">${fmt(a.magnitude_pct, 1)}%</td>
        <td>${escapeHtml(a.reason)}</td>
      </tr>`).join("")}
    ${anoms.length > 12 ? `<tr><td colspan="5" class="center">… plus ${anoms.length - 12} more</td></tr>` : ""}
  </table>`;
}

function renderTariff() {
  $("tariff").innerHTML = `
    <div class="metric-row"><span class="m">Peak (15-20h, weekday)</span><span class="mono">$${cfg_peak}/MWh</span></div>
    <div class="metric-row"><span class="m">Shoulder (7-14h, 21-22h)</span><span class="mono">$${cfg_shoulder}/MWh</span></div>
    <div class="metric-row"><span class="m">Off-peak</span><span class="mono">$${cfg_offpeak}/MWh</span></div>
    <div class="metric-row"><span class="m">Demand charge</span><span class="mono">$${cfg_demand}/MWh peak</span></div>
  `;
}

function escapeHtml(s) {
  return String(s).replace(/[&<>"']/g, (c) =>
    ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
}

async function refresh() {
  try {
    $("refresh").disabled = true;
    $("refresh").textContent = "Analysing…";
    const fc = await loadHistoryAndForecast();
    renderKpis(fc, lastOpt);
    await loadAnomalies();
  } catch (err) {
    toast("Error: " + err.message);
    console.error(err);
  } finally {
    $("refresh").disabled = false;
    $("refresh").textContent = "Analyse";
  }
}

let cfg_peak = 95, cfg_shoulder = 70, cfg_offpeak = 45, cfg_demand = 12;

window.addEventListener("DOMContentLoaded", async () => {
  initCharts();
  renderTariff();
  $("refresh").addEventListener("click", refresh);
  try {
    await loadRegions();
    await refresh();
  } catch (e) {
    toast("Init error: " + e.message);
  }
});