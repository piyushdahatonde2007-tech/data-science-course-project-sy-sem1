'use strict';
const $ = (s, r = document) => r.querySelector(s);
const API = window.API_BASE || '';
const st = { tok: 0, R: null, datasets: [], ds: 'mit_all', tab: 'overview', cell: null, charts: {} };
const f0 = x => x == null ? '–' : Math.round(x).toLocaleString();
const f2 = x => x == null ? '–' : (+x).toFixed(2);
const pill = t => `<span class="pill ${t.split(' ')[0]}">${t}</span>`;
const COL = ['#4cc3ff', '#3ddc97', '#ffb84d', '#ff6b6b', '#b48cff', '#ff8fd1', '#9fb3c8', '#6ee7d8'];
Chart.defaults.color = '#8b9ab3'; Chart.defaults.borderColor = '#26334a'; Chart.defaults.maintainAspectRatio = false;

async function api(path, opt) {
  const r = await fetch(API + '/api' + path, opt);
  if (!r.ok) { let m = r.statusText; try { m = (await r.json()).detail || m; } catch (e) {} throw new Error(typeof m === 'string' ? m : JSON.stringify(m)); }
  return r.json();
}
function toast(m) { const t = $('#toast'); t.textContent = m; t.style.display = 'block'; setTimeout(() => t.style.display = 'none', 4500); }
function chart(id, cfg) { if (st.charts[id]) st.charts[id].destroy(); const c = $('#' + id); if (c) st.charts[id] = new Chart(c, cfg); }
const cv = id => `<div class="chart"><canvas id="${id}"></canvas></div>`;
const r2cls = v => v >= 0.5 ? 'pos' : v < 0 ? 'neg' : '';
const dsName = id => (st.datasets.find(d => d.id === id) || {}).name || id;
const dsSelect = (cur, oninp) => `<select onchange="${oninp}(this.value)">${st.datasets.map(d => `<option value="${d.id}" ${d.id === cur ? 'selected' : ''}>${d.name} (${d.n_cells})</option>`).join('')}</select>`;

const TABS = { overview: 'Overview', datasets: 'Datasets', models: 'Models', generalization: 'Generalization', protocols: 'Protocols', nasa: 'NASA data', alt: 'Accelerated life', paper: 'Paper replication', features: 'Features', explorer: 'Cell explorer', live: 'Live predict', upload: 'Upload data' };
function nav() { $('#nav').innerHTML = Object.entries(TABS).map(([k, v]) => `<button class="${k === st.tab ? 'on' : ''}" onclick="go('${k}')">${v}</button>`).join(''); }
window.go = async (t) => { st.tab = t; nav(); $('#view').innerHTML = '<p class="mut">Loading…</p>'; try { await VIEWS[t](); } catch (e) { $('#view').innerHTML = `<div class="note">${e.message}</div>`; } };

// ---------- Overview ----------
async function overview() {
  const R = st.R, b = R.meta.best_model, m = R.cv.mit_all[b], I = R.interval;
  const lobo = Object.entries(R.cross_batch).map(([k, v]) => [k, Math.max(...Object.values(v.models).map(x => x.R2))]);
  const w = R.window[b], wk = Object.keys(w), ab = R.ablation[b];
  const top = R.importance['Gradient Boosting'] ? R.importance['Gradient Boosting'][0] : null;
  let N = null; try { N = await api('/nasa/summary'); } catch (e) {}
  const read = [
    `<b>${b}</b> is the best model by repeated ${R.meta.cv_folds}-fold CV on all cells: R² ${f2(m.R2)} ± ${f2(m.R2_std)}, RMSE ${f0(m.RMSE)} cycles. Chosen on CV, not on a test split.`,
    `Tier (GOOD / NEEDS TESTING / POOR) agreement on out-of-fold predictions: <b>${f0(R.tier_accuracy * 100)}%</b>.`,
    `<span class="${lobo.every(x => x[1] < 0) ? 'neg' : ''}">Cross-batch generalization is the weak point:</span> when a whole batch is held out, the best model's R² per batch is ${lobo.map(x => `${x[0].replace('hold_out_', '')}: ${f2(x[1])}`).join(', ')}. See the Generalization tab.`,
    `Holding out whole <b>charging protocols</b> gives R² ${f2(R.protocol_cv[b].R2)} for ${b} (random split: ${f2(m.R2)}). See the Protocols tab.`,
    `Nominal ${f0(I.nominal * 100)}% prediction intervals actually cover <b>${f0(I.empirical_coverage * 100)}%</b> of out-of-fold cells (mean width ${f0(I.mean_width)} cycles) — treat them as optimistic.`,
    `R² at a ${wk[0]}-cycle window is ${f2(w[wk[0]].R2)} vs ${f2(w[wk[wk.length - 1]].R2)} at ${wk[wk.length - 1]} cycles: more early cycles add ${w[wk[wk.length - 1]].R2 - w[wk[0]].R2 > 0.1 ? 'meaningful' : 'little'} accuracy for ${b}.`,
    `Removing the 3 charging-protocol features changes CV R² from ${f2(ab.with_protocol.R2)} to ${f2(ab.without_protocol.R2)}.` + (top ? ` Top GB feature: <b>${top.feature}</b>.` : '')];
  if (N) { const nb = N.soh.best_model, nm = N.soh.cv[nb]; read.push(`Separate <b>NASA PCoE data</b> (${N.meta.cells_ok} of ${N.meta.cells_total} cells pass the quality gate): state-of-health estimation on unseen batteries reaches R² ${f2(nm.R2)} (±${f2(nm.R2_std)}), RMSE ${f2(nm.RMSE)} SOH points. Remaining-life prediction there does not beat a cycle-count baseline. See the NASA data tab.`); }
  let AL = null; try { AL = await api('/alt/summary'); } catch (e) {}
  if (AL) { const ab = AL.soh.best_model, rb = AL.rul.best_model, bs = AL.rul.ablation[rb]['cycle + load only (baseline)']; read.push(`Accelerated-life packs (${AL.meta.packs} packs, ${AL.meta.cycles.toLocaleString()} cycles): charge-side SOH estimation on unseen packs reaches R² ${f2(AL.soh.cv[ab].R2)}, and remaining life R² ${f2(AL.rul.cv[rb].R2)} against ${f2(bs.R2)} for a cycle-count-plus-load baseline. See the Accelerated life tab.`); }
  $('#view').innerHTML = `<h2>Overview</h2><p class="sub">All figures are out-of-sample (cross-validated) unless stated.</p>
  <div class="grid g4">
   <div class="card kpi"><div class="v">${R.meta.n_features}</div><div class="l">features · first ${R.meta.early_cycles} cycles</div></div>
   <div class="card kpi"><div class="v">${st.datasets.find(d => d.id === 'mit_all').n_cells}</div><div class="l">cells (3 batches)</div></div>
   <div class="card kpi"><div class="v">${f2(m.R2)}</div><div class="l">CV R² · ${b}</div></div>
   <div class="card kpi"><div class="v">${f0(m.RMSE)}</div><div class="l">CV RMSE (cycles)</div></div></div>
  <div class="card" style="margin-top:14px"><h3>Key readings</h3><ul class="read">${read.map(x => `<li>${x}</li>`).join('')}</ul></div>
  <div class="card" style="margin-top:14px"><h3>Predicted vs actual cycle life (out-of-fold, ${b})</h3>${cv('ovsc')}</div>`;
  const o = await api('/oof'); scatter('ovsc', o);
}
function scatter(id, o) {
  const mx = Math.max(...o.actual, ...o.pred) * 1.05;
  chart(id, { type: 'scatter', data: { datasets: [
    { label: 'cells', data: o.actual.map((a, i) => ({ x: a, y: o.pred[i], id: o.ids[i] })), backgroundColor: '#4cc3ffaa' },
    { label: 'ideal', type: 'line', data: [{ x: 0, y: 0 }, { x: mx, y: mx }], borderColor: '#8b9ab3', borderDash: [5, 5], pointRadius: 0 }] },
    options: { scales: { x: { title: { display: true, text: 'actual cycle life' } }, y: { title: { display: true, text: 'predicted' } } },
      plugins: { tooltip: { callbacks: { label: c => c.raw.id ? `${c.raw.id}: actual ${f0(c.raw.x)}, pred ${f0(c.raw.y)}` : '' } } },
      onClick: (e, els) => { if (els.length && els[0].datasetIndex === 0) openCell('mit_all', o.ids[els[0].index]); } } });
}

// ---------- Datasets ----------
async function datasets() {
  const want = st.ds, tok = ++st.tok;
  const d = await api('/datasets/' + want), cells = await api(`/datasets/${want}/cells`);
  if (tok !== st.tok || st.tab !== 'datasets') return;
  const cvm = d.cv;
  $('#view').innerHTML = `<h2>Datasets</h2><p class="sub">Built-in MIT-Stanford batches plus anything you upload. Pick one to inspect.</p>
  <div class="grid g4">${st.datasets.map(x => `<div class="card dscard ${x.id === st.ds ? 'on' : ''}" onclick="setDs('${x.id}')"><b>${x.name}</b><div class="mut">${x.n_cells} cells${x.labeled ? ` · life ${f0(x.life_min)}–${f0(x.life_max)} (median ${f0(x.life_median)})` : ' · unlabeled'}</div><div class="mut">${x.builtin ? 'built-in' : 'uploaded'}</div></div>`).join('')}</div>
  <div class="grid g2" style="margin-top:14px"><div class="card"><h3>Cycle-life distribution — ${d.name}</h3>${cv('hist')}</div>
  <div class="card"><h3>${cvm ? 'Cross-validated accuracy within this dataset' : d.external_validation ? 'External validation (zero-shot)' : 'No labels'}</h3>
  ${cvm ? `<table><tr><th>Model</th><th>R²</th><th>RMSE</th><th>MAE</th><th>MAPE %</th></tr>${Object.entries(cvm).map(([n, v]) => `<tr><td>${n}</td><td class="${r2cls(v.R2)}">${f2(v.R2)} ± ${f2(v.R2_std)}</td><td>${f0(v.RMSE)}</td><td>${f0(v.MAE)}</td><td>${f2(v.MAPE)}</td></tr>`).join('')}</table>
  <p class="mut">Small batches (~46 cells) give noisy estimates; compare across datasets with that in mind.</p>`
    : d.external_validation ? (e => `<p>${e.model}: R² <b class="${r2cls(e.R2)}">${f2(e.R2)}</b>, RMSE ${f0(e.RMSE)}, MAE ${f0(e.MAE)}, MAPE ${f2(e.MAPE)}% on ${e.n} cells. Interval coverage ${f0(e.interval_coverage * 100)}%.</p><div class="note info">${e.note}</div>`)(d.external_validation)
    : '<p class="mut">Add a cycle_life column to validate predictions.</p>'}
  <p>Tiers: ${Object.entries(d.tier_counts).map(([k, v]) => `${pill(k)} ${v}`).join(' &nbsp; ')}</p></div></div>
  <div class="card" style="margin-top:14px"><div class="row"><h3 style="margin:0">Cells</h3><input id="q" placeholder="search id…" oninput="drawCells()"><select id="tf" onchange="drawCells()"><option value="">all tiers</option><option>GOOD</option><option>NEEDS TESTING</option><option>POOR</option></select></div>
  <div class="scroll"><table id="ct"></table></div></div>`;
  st.cells = cells; drawCells();
  const L = d.lives; if (L.length) { const lo = Math.floor(Math.min(...L) / 100) * 100, hi = Math.ceil(Math.max(...L) / 100) * 100, n = (hi - lo) / 100, bins = Array(n).fill(0); L.forEach(v => bins[Math.min(n - 1, Math.floor((v - lo) / 100))]++);
    chart('hist', { type: 'bar', data: { labels: bins.map((_, i) => lo + i * 100), datasets: [{ data: bins, backgroundColor: '#4cc3ff' }] }, options: { plugins: { legend: { display: false } }, scales: { x: { title: { display: true, text: 'cycle life (100-cycle bins)' } } } } }); }
}
window.setDs = id => { st.ds = id; datasets(); };
window.drawCells = () => {
  const q = ($('#q') || {}).value || '', t = ($('#tf') || {}).value || '';
  const rows = st.cells.filter(c => c.battery_id.toLowerCase().includes(q.toLowerCase()) && (!t || c.tier === t));
  $('#ct').innerHTML = `<tr><th>Cell</th><th>Batch</th><th>Actual life</th><th>Predicted</th><th>80% interval</th><th>Pred. RUL</th><th>Tier</th><th>Actual tier</th></tr>` +
    rows.map(c => `<tr class="click" onclick="openCell('${st.ds}','${c.battery_id}')"><td>${c.battery_id}</td><td>${c.batch}</td><td>${f0(c.actual_cycle_life)}</td><td>${f0(c.predicted_cycle_life)}</td><td>${f0(c.lo)}–${f0(c.hi)}</td><td>${f0(c.predicted_rul)}</td><td>${pill(c.tier)}</td><td>${c.actual_tier ? pill(c.actual_tier) : '–'}</td></tr>`).join('');
};
window.openCell = (ds, id) => { st.ds = ds; st.cell = id; go('explorer'); };

// ---------- Models ----------
async function models() {
  const R = st.R, ds = R.cv[st.ds] ? st.ds : 'mit_all', names = R.models;
  const builtin = st.datasets.filter(d => d.builtin);
  $('#view').innerHTML = `<h2>Models</h2><p class="sub">Repeated stratified ${R.meta.cv_folds}-fold CV × ${R.meta.cv_repeats} repeats; target is log(cycle life); preprocessing is fit inside each fold (no leakage).</p>
  <div class="row">Dataset: <select onchange="st.ds=this.value;models()">${builtin.map(d => `<option value="${d.id}" ${d.id === ds ? 'selected' : ''}>${d.name}</option>`).join('')}</select></div>
  <div class="grid g2"><div class="card"><h3>CV R² (mean over repeats)</h3>${cv('mr2')}</div><div class="card"><h3>CV RMSE (cycles)</h3>${cv('mrm')}</div></div>
  <div class="card" style="margin-top:14px"><table><tr><th>Model</th><th>R²</th><th>RMSE</th><th>MAE</th><th>MAPE %</th></tr>${names.map(n => { const v = R.cv[ds][n]; return `<tr><td>${n}${n === R.meta.best_model && ds === 'mit_all' ? ' ★' : ''}</td><td class="${r2cls(v.R2)}">${f2(v.R2)} ± ${f2(v.R2_std)}</td><td>${f0(v.RMSE)} ± ${f0(v.RMSE_std)}</td><td>${f0(v.MAE)}</td><td>${f2(v.MAPE)}</td></tr>`; }).join('')}</table></div>
  <div class="card" style="margin-top:14px"><div class="row"><h3 style="margin:0">Out-of-fold predicted vs actual (all cells)</h3><select id="om" onchange="oofDraw()">${names.map(n => `<option ${n === R.meta.best_model ? 'selected' : ''}>${n}</option>`).join('')}</select></div>${cv('osc')}</div>`;
  const bar = (id, key) => chart(id, { type: 'bar', data: { labels: names, datasets: [{ data: names.map(n => R.cv[ds][n][key]), backgroundColor: names.map((n, i) => COL[i]) }] }, options: { plugins: { legend: { display: false } } } });
  bar('mr2', 'R2'); bar('mrm', 'RMSE'); oofDraw();
}
window.oofDraw = async () => scatter('osc', await api('/oof?model=' + encodeURIComponent($('#om').value)));

// ---------- Generalization ----------
async function generalization() {
  const R = st.R, b = R.meta.best_model, names = R.models, cb = R.cross_batch;
  $('#view').innerHTML = `<h2>Generalization</h2><p class="sub">The honest tests: unseen batches, shorter windows, and dropping protocol features.</p>
  <div class="card"><h3>Leave-one-batch-out (train on two batches, test on the third) — R²</h3>
  <table><tr><th>Held-out</th><th>n train / test</th>${names.map(n => `<th>${n}</th>`).join('')}</tr>${Object.entries(cb).map(([k, v]) => `<tr><td>${k.replace('hold_out_', 'batch ')}</td><td>${v.n_train} / ${v.n_test}</td>${names.map(n => `<td class="${r2cls(v.models[n].R2)}">${f2(v.models[n].R2)}</td>`).join('')}</tr>`).join('')}</table>
  <div class="note">Negative R² means worse than always predicting the training mean. Batches differ in cell conditions and cycle-life ranges (e.g. batch 2 is short-lived), so a model fit on random cells does not transfer to a new batch. Use predictions on new batches only for coarse screening and re-calibrate with local labels.</div></div>
  <div class="grid g2" style="margin-top:14px"><div class="card"><h3>CV R² vs early-cycle window</h3>${cv('win')}</div>
  <div class="card"><h3>Protocol-feature ablation — ${b}</h3>${cv('abl')}<p class="mut">"Protocol" = C1, Q1, C2 (fast-charge settings). They are unavailable for second-life cells with unknown history.</p></div></div>`;
  const ws = Object.keys(R.window[b]);
  chart('win', { type: 'line', data: { labels: ws.map(w => w + ' cycles'), datasets: Object.entries(R.window).map(([n, v], i) => ({ label: n, data: ws.map(w => v[w].R2), borderColor: COL[i], tension: .25 })) }, options: { scales: { y: { title: { display: true, text: 'CV R²' } } } } });
  const a = R.ablation[b];
  chart('abl', { type: 'bar', data: { labels: ['with protocol (40)', 'without protocol (37)'], datasets: [{ label: 'CV R²', data: [a.with_protocol.R2, a.without_protocol.R2], backgroundColor: ['#4cc3ff', '#ffb84d'] }] }, options: { plugins: { legend: { display: false } } } });
}


// ---------- Protocols ----------
async function protocols() {
  const R = st.R, b = R.meta.best_model, rnd = R.cv.mit_all, pc = R.protocol_cv, names = R.models, pa = R.protocol_ablation[b];
  const cells = await api('/datasets/mit_all/cells');
  const gap = rnd[b].R2 - pc[b].R2, nP = R.protocols.filter(p => p.label !== 'unknown').length, nU = (R.protocols.find(p => p.label === 'unknown') || { n: 0 }).n;
  $('#view').innerHTML = `<h2>Charging protocols</h2><p class="sub">${nP} distinct fast-charge protocols (first-step C-rate C1, switch point Q1 in % SOC, second-step rate C2). Replicate cells share a protocol, so a random split can put the same protocol in both train and test. Leave-protocol-out CV holds whole protocols out together.</p>
  <div class="grid g4">
   <div class="card kpi"><div class="v">${nP}</div><div class="l">distinct protocols${nU ? ` (+${nU} cells with none recorded)` : ''}</div></div>
   <div class="card kpi"><div class="v">${f2(rnd[b].R2)}</div><div class="l">random-split CV R² · ${b}</div></div>
   <div class="card kpi"><div class="v ${r2cls(pc[b].R2)}">${f2(pc[b].R2)}</div><div class="l">leave-protocol-out CV R²</div></div>
   <div class="card kpi"><div class="v">${f2(pa.without_protocol.R2)}</div><div class="l">…and without C1/Q1/C2 features</div></div></div>
  <div class="note ${Math.abs(gap) < 0.1 ? 'info' : ''}">Random vs protocol-held-out R² differ by ${f2(gap)} for ${b}. ${gap > 0.1 ? 'Part of the random-split accuracy comes from seeing the same protocol (replicate cells) during training, so expect lower accuracy on a new charging protocol.' : 'The model transfers to unseen protocols about as well as to unseen cells.'} Small groups and one repeat set make this an estimate, not a guarantee.</div>
  <div class="grid g2" style="margin-top:14px"><div class="card"><h3>Cycle life vs first-step C-rate (C1)</h3>${cv('pc1')}</div>
  <div class="card"><h3>CV R²: random split vs leave-protocol-out</h3>${cv('pcm')}</div></div>
  <div class="card" style="margin-top:14px"><h3>Protocol table</h3><div class="scroll"><table><tr><th>Protocol</th><th>Cells</th><th>Mean life</th><th>Range</th><th>Batches</th></tr>${R.protocols.map(p => `<tr><td>${p.label}</td><td>${p.n}</td><td>${f0(p.life_mean)}</td><td>${f0(p.life_min)}–${f0(p.life_max)}</td><td>${p.batches.join(', ')}</td></tr>`).join('')}</table></div></div>`;
  const bs = ['1', '2', '3', 'custom'];
  chart('pc1', { type: 'scatter', data: { datasets: bs.map((k, i) => ({ label: k === 'custom' ? 'unknown' : 'batch ' + k, backgroundColor: COL[i] + 'bb',
      data: cells.filter(c => (c.batch || 'custom') === k && c.c1 != null).map(c => ({ x: c.c1, y: c.actual_cycle_life, id: c.battery_id, p: c.protocol })) })).filter(d => d.data.length) },
    options: { scales: { x: { title: { display: true, text: 'C1 (C-rate)' } }, y: { title: { display: true, text: 'actual cycle life' } } },
      plugins: { tooltip: { callbacks: { label: c => `${c.raw.id} ${c.raw.p}: ${f0(c.raw.y)} cycles` } } } } });
  chart('pcm', { type: 'bar', data: { labels: names, datasets: [{ label: 'random split', data: names.map(n => rnd[n].R2), backgroundColor: '#4cc3ff' }, { label: 'leave-protocol-out', data: names.map(n => pc[n].R2), backgroundColor: '#ffb84d' }] } });
}


// ---------- NASA ----------
const GCOL = { 'ICA': '#4cc3ff', 'ICA Δ': '#b48cff', 'charge/CV': '#3ddc97', 'charge/CV Δ': '#ffb84d', 'temperature': '#ff8fd1', 'temperature Δ': '#ff6b6b' };
async function nasaView() {
  let N; try { N = await api('/nasa/summary'); } catch (e) { $('#view').innerHTML = `<div class="note">${e.message}</div>`; return; }
  const b = N.soh.best_model, cvs = N.soh.cv, mt = N.meta, ab = N.soh.ablation[b], rl = N.rul, names = N.models;
  const hd = ab['absolute + Δ (headline)'], rb = rl.best_model, base = rl.baseline_cycle_only;
  const beats = rl.cv[rb].R2 > base.R2;
  const imp = (N.importance[b] || N.importance['XGBoost']).slice(0, 12);
  st.nc = st.nc || 'B0005';
  const opt = c => `<option value="${c.battery_id}" ${c.battery_id === st.nc ? 'selected' : ''}>${c.battery_id} · ${c.ambient_c.join('/')}°C${c.ok ? '' : ' (excluded)'}</option>`;
  $('#view').innerHTML = `<h2>NASA PCoE battery data</h2>
  <p class="sub">A different data family from MIT-Stanford: ${mt.cells_total} cells, ${mt.data_files.toLocaleString()} cycle files, five ambient temperatures, varied discharge loads, but one identical charge protocol (1.5 A to 4.2 V, then constant voltage). Trained and validated separately. All results are leave-battery-out: whole batteries are held out together.</p>
  <div class="grid g4">
   <div class="card kpi"><div class="v">${mt.cells_ok}/${mt.cells_total}</div><div class="l">cells pass the quality gate</div></div>
   <div class="card kpi"><div class="v">${mt.rows.toLocaleString()}</div><div class="l">usable charge cycles with SOH labels</div></div>
   <div class="card kpi"><div class="v ${r2cls(cvs[b].R2)}">${f2(cvs[b].R2)}</div><div class="l">SOH CV R² · ${b}</div></div>
   <div class="card kpi"><div class="v">${f2(cvs[b].RMSE)}</div><div class="l">SOH error, RMSE (points of %)</div></div></div>
  <div class="note info">Task: estimate each cell's <b>state of health</b> (capacity relative to its own start) from one charge curve plus its start-of-life reference. Capacity in Ah is not comparable between cells because discharge load and cut-off voltage differ, so SOH is cell-relative. The paper's task, predicting total life from the first 50 cycles, is <b>not trainable</b> here: only ${rl.n_cells} usable cells reach 80% SOH, and ${rl.cells_ending_before_50} of them end before cycle 50.</div>
  <div class="grid g2" style="margin-top:14px"><div class="card"><h3>SOH estimation: leave-battery-out CV</h3>
   <table><tr><th>Model</th><th>R²</th><th>RMSE</th><th>MAE</th></tr>${names.map(n => { const v = cvs[n]; return `<tr><td>${n}${n === b ? ' ★' : ''}</td><td class="${r2cls(v.R2)}">${f2(v.R2)} ± ${f2(v.R2_std)}</td><td>${f2(v.RMSE)}</td><td>${f2(v.MAE)}</td></tr>`; }).join('')}</table>
   <p class="mut">Error is in SOH points (100 = the cell's own early capacity).</p></div>
  <div class="card"><h3>Predicted vs measured SOH (out-of-fold, ${b})</h3>${cv('nsc')}<p class="mut" id="nscnote"></p></div></div>
  <div class="card" style="margin-top:14px"><h3>What carries the signal? (${b}, leave-battery-out R²)</h3><div class="chart" style="height:340px"><canvas id="nab"></canvas></div>
   <p class="mut">Absolute ICA features alone are weak (R² ${f2(ab['absolute ICA only (+ambient)'].R2)}). Changes from the cell's own start do the work: Δ charge/constant-voltage statistics alone give ${f2(ab['Δ charge/CV only (+ambient)'].R2)}, and adding ICA and absolute features lifts that to ${f2(hd.R2)}. A cycle-count-only model scores ${f2(ab['cycle number only (+ambient)'].R2)}. The Δ features need the cell's first ${mt.ref_cycles} charge cycles as a start-of-life reference. ICA is computed over ${mt.ica_window_v[0]}–${mt.ica_window_v[1]} V because that is the only range the constant-current phase reaches in most cycles (it typically starts near 3.8 V and varies by cycle), so lower-voltage peaks, like the reference cell's near 3.76 V below, are unavailable for most cycles.</p></div>
  <div class="card" style="margin-top:14px"><div class="row"><h3 style="margin:0">Cell explorer</h3><select onchange="st.nc=this.value;nasaCell()">${N.cells.map(opt).join('')}</select></div><div id="ncinfo"></div>
   <div class="grid g2"><div><h3>SOH over cycles</h3>${cv('ncell')}</div><div><h3>Real ICA: dQ/dV vs voltage during charge</h3>${cv('nica')}</div></div></div>
  <div class="grid g2" style="margin-top:14px"><div class="card"><h3>Remaining useful life (to 80% SOH)</h3>
   <table><tr><th>Model</th><th>R²</th><th>RMSE (cycles)</th><th>MAE</th></tr>${names.map(n => { const v = rl.cv[n]; return `<tr><td>${n}</td><td class="${r2cls(v.R2)}">${f2(v.R2)}</td><td>${f2(v.RMSE)}</td><td>${f2(v.MAE)}</td></tr>`; }).join('')}
   <tr><td>baseline: cycle number + ambient only</td><td class="${r2cls(base.R2)}">${f2(base.R2)}</td><td>${f2(base.RMSE)}</td><td>${f2(base.MAE)}</td></tr></table>
   <div class="note ${beats ? 'info' : ''}">${beats ? 'The best model beats the cycle-count baseline, but with only ' + rl.n_cells + ' cells treat it as indicative.' : 'No model beats the cycle-count baseline. Remaining life of an unseen battery is <b>not reliably predictable</b> from this data (' + rl.n_cells + ' cells reach 80% SOH, at cycles ' + Object.values(rl.eol_cycles).sort((x, y) => x - y).join(', ') + '). Not recommended for use.'}</div></div>
  <div class="card"><h3>Top features (${imp.length}, ${N.importance[b] ? b : 'XGBoost'})</h3><div class="chart"><canvas id="nfi"></canvas></div><p class="mut">Δ = change from the cell's own start-of-life value.</p></div></div>
  ${N.sp1 ? `<div class="card" style="margin-top:14px"><h3>Reference cell test: ${N.sp1.file}</h3><p>Measured initial capacity: <b>${f2(N.sp1.initial_capacity_ah)} Ah</b> (1 A discharge to 2.5 V). ${N.sp1.note}</p>
   <div class="grid g2"><div><h3>Voltage and current profile</h3>${cv('nsp1a')}</div><div><h3>Real ICA of the 1 A constant-current charge</h3>${cv('nsp1b')}</div></div>
   <div class="scroll" style="max-height:220px"><table><tr><th>Step</th><th>Mode</th><th>Hours</th><th>Start V</th><th>End V</th><th>Current (A)</th></tr>${N.sp1.steps.map(x => `<tr><td>${x.step}</td><td>${x.mode}</td><td>${x.hours}</td><td>${x.v_start}</td><td>${x.v_end}</td><td>${x.current_a}</td></tr>`).join('')}</table></div></div>` : ''}
  <div class="card" style="margin-top:14px"><h3>Quality gate: all ${mt.cells_total} cells</h3><p class="mut">Cells are used for SOH labels only if the capacity trajectory is usable. Rejected cells still appear in the explorer.</p>
   <div class="scroll"><table><tr><th>Cell</th><th>Ambient °C</th><th>Cycles</th><th>Reference Ah</th><th>Status</th><th>80% SOH at</th><th>70% SOH at</th></tr>${N.cells.map(c => `<tr class="click" onclick="st.nc='${c.battery_id}';nasaView()"><td>${c.battery_id}</td><td>${c.ambient_c.join('/')}</td><td>${c.n_cycles}</td><td>${c.ref_capacity ? f2(c.ref_capacity) : '–'}</td><td class="${c.ok ? 'pos' : 'neg'}">${c.ok ? 'used' : c.reason}</td><td>${c.eol80_cycle || '–'}</td><td>${c.eol70_cycle || '–'}</td></tr>`).join('')}</table></div></div>`;
  chart('nab', { type: 'bar', data: { labels: Object.keys(ab), datasets: [{ data: Object.values(ab).map(v => v.R2), backgroundColor: Object.keys(ab).map(k => k.includes('headline') ? '#3ddc97' : '#4cc3ff') }] }, options: { indexAxis: 'y', plugins: { legend: { display: false } }, scales: { x: { title: { display: true, text: 'R²' } } } } });
  chart('nfi', { type: 'bar', data: { labels: imp.map(x => x.feature), datasets: [{ data: imp.map(x => x.importance), backgroundColor: imp.map(x => GCOL[x.group] || '#9fb3c8') }] }, options: { indexAxis: 'y', plugins: { legend: { display: false }, tooltip: { callbacks: { afterLabel: c => imp[c.dataIndex].group } } } } });
  if (N.sp1) {
    const P = N.sp1.profile;
    chart('nsp1a', { type: 'line', data: { labels: P.hours, datasets: [{ label: 'voltage (V)', data: P.v, borderColor: '#4cc3ff', pointRadius: 0, yAxisID: 'y' }, { label: 'current (A)', data: P.i, borderColor: '#ffb84d', pointRadius: 0, yAxisID: 'y2' }] },
      options: { scales: { x: { title: { display: true, text: 'test time (h)' }, ticks: { maxTicksLimit: 8 } }, y: { title: { display: true, text: 'V' } }, y2: { position: 'right', grid: { drawOnChartArea: false }, title: { display: true, text: 'A' } } } } });
    chart('nsp1b', { type: 'line', data: { labels: N.sp1.ica.v, datasets: [{ label: 'dQ/dV (Ah/V)', data: N.sp1.ica.ic, borderColor: '#3ddc97', pointRadius: 0, tension: .2 }] }, options: { plugins: { legend: { display: false } }, scales: { x: { title: { display: true, text: 'voltage (V)' }, ticks: { maxTicksLimit: 8 } } } } });
  }
  const o = await api('/nasa/oof');
  chart('nsc', { type: 'scatter', data: { datasets: [{ label: 'cycles', data: o.y.map((y, i) => ({ x: y, y: o.p[i], id: o.b[i], k: o.k[i] })), backgroundColor: '#4cc3ff66', pointRadius: 2.5 },
    { label: 'ideal', type: 'line', data: [{ x: 30, y: 30 }, { x: 115, y: 115 }], borderColor: '#8b9ab3', borderDash: [5, 5], pointRadius: 0 }] },
    options: { scales: { x: { title: { display: true, text: 'measured SOH %' } }, y: { title: { display: true, text: 'predicted SOH %' } } }, plugins: { tooltip: { callbacks: { label: c => c.raw.id ? `${c.raw.id} cycle ${c.raw.k}: measured ${f2(c.raw.x)}, predicted ${f2(c.raw.y)}` : '' } } } } });
  { const lo = Math.min(...o.p), ym = Math.min(...o.y); $('#nscnote').textContent = lo - ym > 5 ? `Lowest prediction is ${f2(lo)}% while measured SOH falls to ${f2(ym)}%: this model cannot predict below the degradation it saw in other cells, so it underestimates the most-aged cycles.` : ''; }
  nasaCell();
}
window.nasaCell = async () => {
  const id = st.nc, [d, ic] = await Promise.all([api('/nasa/cells/' + id), api(`/nasa/cells/${id}/ica?n=5`)]);
  const i = d.info;
  $('#ncinfo').innerHTML = `<p>${id}: ambient ${i.ambient_c.join('/')} °C, ${i.n_cycles} discharge cycles, reference capacity ${i.ref_capacity ? f2(i.ref_capacity) + ' Ah' : '–'}. ${i.ok ? '<span class="pos">Used for SOH labels.</span>' : '<span class="neg">Excluded: ' + i.reason + '.</span>'} ${i.eol80_cycle ? 'Reaches 80% SOH at cycle ' + i.eol80_cycle + '.' : ''}</p><p class="mut">${d.note}</p>`;
  if (d.soh.length) chart('ncell', { type: 'line', data: { labels: d.soh.map(x => x.k), datasets: [{ label: 'measured SOH %', data: d.soh.map(x => x.y), borderColor: '#4cc3ff', pointRadius: 0, tension: .2 }, { label: 'predicted (out-of-fold)', data: d.soh.map(x => x.p), borderColor: '#ffb84d', pointRadius: 0, tension: .2 }] }, options: { scales: { x: { title: { display: true, text: 'discharge cycle' } }, y: { title: { display: true, text: 'SOH %' } } } } });
  else { const ok = d.raw_capacity.filter(x => x.ok); chart('ncell', { type: 'line', data: { labels: ok.map(x => x.k), datasets: [{ label: 'usable capacity (Ah), no SOH label', data: ok.map(x => x.cap), borderColor: '#ff6b6b', pointRadius: 0 }] }, options: { scales: { x: { title: { display: true, text: 'discharge cycle' } } } } }); }
  chart('nica', { type: 'line', data: { labels: (ic.curves[0] || { v: [] }).v, datasets: ic.curves.map((c, j) => ({ label: 'cycle ' + c.k, data: c.ic, borderColor: COL[j % COL.length], pointRadius: 0, tension: .2 })) }, options: { scales: { x: { title: { display: true, text: 'voltage (V)' }, ticks: { maxTicksLimit: 8 } }, y: { title: { display: true, text: 'dQ/dV (Ah/V)' } } } } });
};


// ---------- Accelerated life (NASA/UCF packs) ----------
async function altView() {
  let A; try { A = await api('/alt/summary'); } catch (e) { $('#view').innerHTML = `<div class="note">${e.message}</div>`; return; }
  const mt = A.meta, b = A.soh.best_model, sc = A.soh.cv, ab = A.soh.ablation[b], rb = A.rul.best_model, rc = A.rul.cv, ra = A.rul.ablation[rb];
  const lv = A.life_vs_load, tr = A.soh.second_life_transfer, names = A.models;
  const rbase = ra['cycle + load only (baseline)'], rhead = ra['charge-side Δ (+load) (headline)'], sbase = ab['cycle number + load only'];
  const beats = rc[rb].R2 > rbase.R2 + 0.05;
  const imp = (A.importance.soh || { top: [] }).top.slice(0, 10), impr = (A.importance.rul || { top: [] }).top.slice(0, 10);
  st.ap = st.ap || '01';
  const opt = p => `<option value="${p.pack}" ${p.pack === st.ap ? 'selected' : ''}>${p.pack} · ${p.group.replace('_', ' ')} · ${p.load_label} A</option>`;
  const GL = { regular: 'regular', recommissioned: 'recommissioned', second_life: 'second life' };
  $('#view').innerHTML = `<h2>Accelerated life testing: battery packs cycled to failure</h2>
  <p class="sub">${mt.packs} packs, ${mt.cycles.toLocaleString()} discharge cycles under known load levels (5 to 19 A), from first life to failure; ${A.packs.filter(p => p.group === 'second_life').length} are genuine second-life packs. Source: ${mt.source}. All results are leave-pack-out.</p>
  <div class="grid g4">
   <div class="card kpi"><div class="v">${mt.packs}</div><div class="l">packs · ${mt.reference_cycles} reference discharges</div></div>
   <div class="card kpi"><div class="v ${r2cls(sc[b].R2)}">${f2(sc[b].R2)}</div><div class="l">SOH CV R² · ${b}</div></div>
   <div class="card kpi"><div class="v">${f2(sc[b].RMSE)}</div><div class="l">SOH error, RMSE (points of %)</div></div>
   <div class="card kpi"><div class="v ${r2cls(rc[rb].R2)}">${f2(rc[rb].R2)}</div><div class="l">remaining-life CV R² (baseline ${f2(rbase.R2)})</div></div></div>
  <div class="note info">Charging logs only a voltage trace (no charge current), so charge curves cannot give ICA here. The labels come from periodic <b>reference discharges</b> at a constant 2.5 A. Real ICA (dQ/dV) is computed from those discharges, but a full discharge contains the capacity itself, so ICA-from-discharge is shown as an upper bound. The practical estimator uses <b>charge-side signals only</b> (charge time, charge-voltage shape, rest voltage, voltage sag under load, temperature), each as a change from the pack's own first cycles.</div>
  <div class="grid g2" style="margin-top:14px"><div class="card"><h3>Life vs load (packs that reached 80% capacity)</h3>${cv('alv')}<p class="mut">Life falls with load roughly as load<sup>${f2(lv.slope)}</sup>. Leave-one-pack-out error of this fit is ${f0(lv.loo_mape)}% on only ${lv.n} packs: higher load means shorter life, but the sample is too small for a precise law.</p></div>
  <div class="card"><h3>SOH estimation: leave-pack-out CV</h3><table><tr><th>Model</th><th>R²</th><th>RMSE</th><th>MAE</th></tr>${names.map(n => { const v = sc[n]; return `<tr><td>${n}${n === b ? ' ★' : ''}</td><td class="${r2cls(v.R2)}">${f2(v.R2)} ± ${f2(v.R2_std)}</td><td>${f2(v.RMSE)}</td><td>${f2(v.MAE)}</td></tr>`; }).join('')}</table>
   <p class="mut">${mt.soh_rows} reference discharges from ${mt.soh_packs} packs. SOH = reference capacity / the pack's own initial reference capacity; values below 50% (failure collapse) are excluded. A cycle-count-plus-load baseline scores R² ${f2(sbase.R2)}.</p></div></div>
  <div class="grid g2" style="margin-top:14px"><div class="card"><h3>What carries the signal? (${b}, R²)</h3><div class="chart" style="height:340px"><canvas id="aab"></canvas></div>
   <p class="mut">Absolute features are weak; changes from the pack's own start do the work. The headline set was chosen by CV from the sets shown, which is mildly optimistic. ICA from a reference discharge adds almost nothing beyond charge-side signals (${f2(ab['ICA + charge-side (needs ref. discharge)'].R2)} vs ${f2(ab['charge-side Δ (+load) (headline)'].R2)}).</p></div>
  <div class="card"><h3>Predicted vs measured SOH (out-of-fold, ${b})</h3>${cv('asc')}</div></div>
  <div class="grid g2" style="margin-top:14px"><div class="card"><h3>Remaining life to 80% capacity (${A.rul.n_packs} packs)</h3>
   <table><tr><th>Model</th><th>R²</th><th>RMSE (cycles)</th><th>MAE</th></tr>${names.map(n => { const v = rc[n]; return `<tr><td>${n}${n === rb ? ' ★' : ''}</td><td class="${r2cls(v.R2)}">${f2(v.R2)}</td><td>${f0(v.RMSE)}</td><td>${f0(v.MAE)}</td></tr>`; }).join('')}
   <tr><td>baseline: cycle number + load only</td><td class="${r2cls(rbase.R2)}">${f2(rbase.R2)}</td><td>${f0(rbase.RMSE)}</td><td>${f0(rbase.MAE)}</td></tr></table>
   <div class="note ${beats ? 'info' : ''}">${beats ? `The best model clearly beats the cycle-count-plus-load baseline (R² ${f2(rc[rb].R2)} vs ${f2(rbase.R2)}), but typical error is still about ${f0(rc[rb].RMSE)} cycles, large for packs that last under 100 cycles. Only ${A.rul.n_packs} packs reach 80%, so treat it as indicative.` : `No clear gain over the cycle-count-plus-load baseline (R² ${f2(rc[rb].R2)} vs ${f2(rbase.R2)}).`}</div></div>
  <div class="card"><h3>Remaining life: predicted vs actual (out-of-fold, ${rb})</h3>${cv('arsc')}<p class="mut" id="arnote"></p></div></div>
  ${tr ? `<div class="card" style="margin-top:14px"><h3>Second-life transfer test</h3><p>Trained on first-life packs only, scored on the ${tr.n_packs} second-life packs (${tr.n_rows} reference discharges): R² <b class="${r2cls(tr.R2)}">${f2(tr.R2)}</b>, RMSE ${f2(tr.RMSE)} SOH points, MAE ${f2(tr.MAE)}.</p><div class="note ${tr.R2 > 0.7 ? 'info' : ''}">${tr.R2 > 0.7 ? 'The SOH model transfers to second-life packs.' : 'The SOH model transfers only weakly to second-life packs (error is larger than in the leave-pack-out result above). With 3 second-life packs this is a warning sign, not a verdict: screening reused packs needs data from reused packs.'}</div></div>` : ''}
  <div class="card" style="margin-top:14px"><div class="row"><h3 style="margin:0">Pack explorer</h3><select onchange="st.ap=this.value;altPack()">${A.packs.map(opt).join('')}</select></div><div id="apinfo"></div>
   <div class="grid g2"><div><h3>Discharge capacity per cycle</h3>${cv('apcap')}</div><div><h3>Reference-discharge SOH: measured vs predicted</h3>${cv('apsoh')}</div>
   <div><h3>Remaining life: predicted vs actual</h3>${cv('aprul')}</div><div><h3>Real ICA of reference discharges (2.5 A)</h3>${cv('apica')}</div></div></div>
  <div class="grid g2" style="margin-top:14px"><div class="card"><h3>Top features: SOH (${(A.importance.soh || {}).model || ''})</h3><div class="chart"><canvas id="afi"></canvas></div></div>
  <div class="card"><h3>Top features: remaining life (${(A.importance.rul || {}).model || ''})</h3><div class="chart"><canvas id="afr"></canvas></div><p class="mut">_d = change from the pack's own start; load_a = current load level.</p></div></div>
  <div class="card" style="margin-top:14px"><h3>All ${mt.packs} packs</h3><div class="scroll"><table><tr><th>Pack</th><th>Group</th><th>Load (A)</th><th>Cycles</th><th>Reference runs</th><th>Initial Ah</th><th>80% capacity at cycle</th></tr>${A.packs.map(p => `<tr class="click" onclick="st.ap='${p.pack}';altView()"><td>${p.pack}</td><td>${GL[p.group]}</td><td>${p.load_label}${p.kind === 'variable' ? ' (variable)' : ''}</td><td>${p.cycles}</td><td>${p.ref_cycles}</td><td>${f2(p.c0_ah)}</td><td>${p.eol80_cycle || `not reached (min ${f0(p.cap_min_ratio * 100)}%)`}</td></tr>`).join('')}</table></div>
   <p class="mut">Data licence: MIT. Cite Fricke, Nascimento and Viana (2023), an accelerated life testing dataset for lithium-ion batteries with constant and variable loading conditions.</p></div>`;
  const xs = lv.points.map(p => p.load), lo = Math.min(...xs) * 0.9, hi = Math.max(...xs) * 1.1, line = Array.from({ length: 24 }, (_, i) => { const x = lo * Math.pow(hi / lo, i / 23); return { x, y: Math.exp(lv.intercept + lv.slope * Math.log(x)) }; });
  chart('alv', { type: 'scatter', data: { datasets: [{ label: 'packs', data: lv.points.map(p => ({ x: p.load, y: p.life, id: p.pack })), backgroundColor: '#4cc3ff', pointRadius: 5 }, { label: 'fit', type: 'line', data: line, borderColor: '#ffb84d', pointRadius: 0 }] },
    options: { scales: { x: { type: 'logarithmic', title: { display: true, text: 'load (A)' }, ticks: { callback: v => [10, 12, 14, 16, 19].includes(v) ? v : '' } }, y: { type: 'logarithmic', title: { display: true, text: 'cycles to 80% capacity' }, ticks: { callback: v => [100, 200, 500, 1000].includes(v) ? v : '' } } },
      plugins: { tooltip: { callbacks: { label: c => c.raw.id ? `pack ${c.raw.id}: ${c.raw.x} A, ${c.raw.y} cycles` : '' } } } } });
  chart('aab', { type: 'bar', data: { labels: Object.keys(ab), datasets: [{ data: Object.values(ab).map(v => v.R2), backgroundColor: Object.keys(ab).map(k => k.includes('headline') ? '#3ddc97' : k.includes('ICA') ? '#b48cff' : '#4cc3ff') }] }, options: { indexAxis: 'y', plugins: { legend: { display: false } }, scales: { x: { title: { display: true, text: 'R²' } } } } });
  const bars = (id, arr, col) => chart(id, { type: 'bar', data: { labels: arr.map(x => x.feature), datasets: [{ data: arr.map(x => x.importance), backgroundColor: col }] }, options: { indexAxis: 'y', plugins: { legend: { display: false } } } });
  bars('afi', imp, '#4cc3ff'); bars('afr', impr, '#ffb84d');
  const sc2 = (id, o, ax, lo2, hi2) => chart(id, { type: 'scatter', data: { datasets: [{ label: 'points', data: o.y.map((y, i) => ({ x: y, y: o.p[i], id: o.b[i], k: o.k[i] })), backgroundColor: '#4cc3ff66', pointRadius: 2.5 }, { label: 'ideal', type: 'line', data: [{ x: lo2, y: lo2 }, { x: hi2, y: hi2 }], borderColor: '#8b9ab3', borderDash: [5, 5], pointRadius: 0 }] },
    options: { scales: { x: { title: { display: true, text: 'measured ' + ax } }, y: { title: { display: true, text: 'predicted ' + ax } } }, plugins: { tooltip: { callbacks: { label: c => c.raw.id ? `pack ${c.raw.id} cycle ${c.raw.k}: measured ${f2(c.raw.x)}, predicted ${f2(c.raw.y)}` : '' } } } } });
  const [so, ro] = await Promise.all([api('/alt/oof?task=soh'), api('/alt/oof?task=rul')]);
  sc2('asc', so, 'SOH %', 50, 105); sc2('arsc', ro, 'RUL (cycles)', 0, Math.max(...ro.y));
  { const ym = Math.max(...ro.y), pm = Math.max(...ro.p); $('#arnote').textContent = ym - pm > 100 ? `Highest prediction is ${f0(pm)} cycles while actual remaining life reaches ${f0(ym)}: the model underestimates the longest-lived packs, because it has seen few of them.` : ''; }
  altPack();
}
window.altPack = async () => {
  const d = await api(`/alt/packs/${st.ap}`), i = d.info;
  $('#apinfo').innerHTML = `<p>Pack ${i.pack}: ${i.group.replace('_', ' ')}, load ${i.load_label} A${i.kind === 'variable' ? ' (variable within a mission)' : ''}, ${i.cycles} cycles, initial capacity ${f2(i.c0_ah)} Ah. ${i.eol80_cycle ? 'Reaches 80% capacity at cycle ' + i.eol80_cycle + '.' : 'Does not reach 80% capacity before the test ends (minimum ' + f0(i.cap_min_ratio * 100) + '%).'}</p><p class="mut">${d.note}</p>`;
  const reg = d.capacity.filter(x => x.m === 1), ref = d.capacity.filter(x => x.m === 0);
  chart('apcap', { type: 'scatter', data: { datasets: [{ label: 'regular missions', data: reg.map(x => ({ x: x.k, y: x.c })), backgroundColor: '#4cc3ff55', pointRadius: 1.5 }, { label: 'reference (2.5 A)', data: ref.map(x => ({ x: x.k, y: x.c })), backgroundColor: '#ffb84d', pointRadius: 3 }] }, options: { scales: { x: { title: { display: true, text: 'discharge cycle' } }, y: { title: { display: true, text: 'Ah' } } } } });
  if (d.soh.length) chart('apsoh', { type: 'line', data: { labels: d.soh.map(x => x.k), datasets: [{ label: 'measured SOH %', data: d.soh.map(x => x.y), borderColor: '#4cc3ff', tension: .2 }, { label: 'predicted (out-of-fold)', data: d.soh.map(x => x.p), borderColor: '#ffb84d', tension: .2 }] }, options: { scales: { x: { title: { display: true, text: 'discharge cycle' } } } } });
  else chart('apsoh', { type: 'line', data: { labels: [], datasets: [] } });
  if (d.rul.length) chart('aprul', { type: 'line', data: { labels: d.rul.map(x => x.k), datasets: [{ label: 'actual RUL', data: d.rul.map(x => x.y), borderColor: '#4cc3ff', pointRadius: 0 }, { label: 'predicted (out-of-fold)', data: d.rul.map(x => x.p), borderColor: '#ffb84d', pointRadius: 0 }] }, options: { scales: { x: { title: { display: true, text: 'discharge cycle' } }, y: { title: { display: true, text: 'cycles remaining' } } } } });
  else chart('aprul', { type: 'line', data: { labels: [], datasets: [] } });
  chart('apica', { type: 'line', data: { labels: d.ica.v, datasets: d.ica.curves.map((c, j) => ({ label: 'cycle ' + c.k, data: c.ic, borderColor: COL[j % COL.length], pointRadius: 0, tension: .2 })) }, options: { scales: { x: { title: { display: true, text: 'pack voltage (V)' }, ticks: { maxTicksLimit: 8 } }, y: { title: { display: true, text: '|dQ/dV| (Ah/V)' } } } } });
};


// ---------- Paper replication ----------
async function paperView() {
  const P = await api('/paper'), M = P.models, R = P.results, n = P.seeds.length;
  const fm = (o, k, d = 0) => d ? (+o[k]).toFixed(d) : f0(o[k]);
  const cell = (o, sd) => `${f2(o.R2)}${sd ? ' ± ' + f2(sd.R2) : ''} · ${f0(o.RMSE)} · ${(+o.MAPE).toFixed(1)}%`;
  const rank = k => [...M].sort((a, b) => R[b].mean.R2 - R[a].mean.R2);
  const ours = rank(), paperRank = [...M].sort((a, b) => P.paper[b].R2 - P.paper[a].R2);
  $('#view').innerHTML = `<h2>Replicating the IEEE paper</h2>
  <p class="sub">The paper's evaluation protocol re-run on the MIT-Stanford data we have: stratified split across cycle-life terciles (${P.n_train} train / ${P.n_test} test, the paper's 58%/42%), features standardised on training data, 5-fold grid search on the training set, raw cycle life as the target, four models, four metrics. Results over ${n} random splits (a single split is a noisy draw).</p>
  <div class="note">What cannot be matched: the paper used 124 cells and 108 ICA features from raw voltage-capacity curves. We have ${P.n_cells} cells and ${P.n_features} per-cycle summary features, so these numbers are <b>protocol-matched, not feature-matched</b>. Exactly reproducing the paper's features needs the raw MIT-Stanford curves, which were not available.</div>
  <div class="card" style="margin-top:14px"><h3>Paper vs this project (each cell: R² · RMSE in cycles · MAPE)</h3><div class="scroll"><table><tr><th>Model</th><th>Paper (Table 2)</th><th>Ours, paper protocol, 1st split</th><th>Ours, paper protocol, mean of ${n}</th><th>Ours, improved pipeline, mean of ${n}</th></tr>
  ${M.map(k => `<tr><td>${k}</td><td>${cell(P.paper[k])}</td><td>${cell(R[k].single)}</td><td>${cell(R[k].mean, R[k].std)}</td><td>${cell(R['improved:' + k].mean, R['improved:' + k].std)}</td></tr>`).join('')}</table></div>
  <p class="mut">"Improved pipeline" = log-cycle-life target and fixed regularised hyperparameters, same splits. ${(() => { const g = k => R['improved:' + k].mean.R2 - R[k].mean.R2, up = M.filter(k => g(k) > 0), top = [...M].sort((a, b) => g(b) - g(a))[0], mp = M.filter(k => R['improved:' + k].mean.MAPE < R[k].mean.MAPE).length; return `It lowers MAPE for ${mp} of 4 models and raises mean R² for ${up.length} of 4 (largest gain: ${top}, ${f2(R[top].mean.R2)} to ${f2(R['improved:' + top].mean.R2)}). Where it does not help (${M.filter(k => g(k) <= 0).join(', ') || 'none'}) the difference is within split-to-split noise.`; })()} Our RMSE is larger than the paper's partly because our cells span a wider life range (up to 1,935 cycles in batch 3).</p></div>
  <div class="grid g2" style="margin-top:14px"><div class="card"><h3>R² by model</h3>${cv('pp1')}</div><div class="card"><h3>R² on each of the ${n} random splits (paper protocol)</h3>${cv('pp2')}<p class="mut">Spread across splits is large (XGBoost ${f2(Math.min(...R.XGBoost.per_seed_r2))} to ${f2(Math.max(...R.XGBoost.per_seed_r2))}), so any single-split number, ours or the paper's, carries real uncertainty.</p></div></div>
  <div class="grid g2" style="margin-top:14px"><div class="card"><h3>XGBoost predicted vs actual (paper protocol, first split)</h3>${cv('pp3')}</div>
  <div class="card"><h3>What reproduces and what does not</h3><ul class="read"><li>Best model: <b>${ours[0]}</b> here, ${paperRank[0]} in the paper. ${ours[0] === paperRank[0] ? 'Reproduced.' : 'Differs.'}</li><li>Worst model: <b>${ours[ours.length - 1]}</b> here, ${paperRank[paperRank.length - 1]} in the paper. ${ours[ours.length - 1] === paperRank[paperRank.length - 1] ? 'Reproduced.' : 'Differs.'}</li><li>Middle order: ours ${ours.join(' > ')}; paper ${paperRank.join(' > ')}. Differences between Random Forest and Gradient Boosting are within split-to-split noise (sd about 0.1).</li><li>R² is higher than the paper's for all four models, with the caveat above about different cells and features.</li>${(n => '')(0)}</ul>
  <p class="mut">Not testable here: the paper's claim that mid-charge-voltage ICA features dominate importance (needs the raw curves).</p></div></div>
  <div class="card" style="margin-top:14px"><h3>The paper's method (ICA features + four ML models) on datasets where real ICA is possible</h3><p class="mut">Leave-battery-out cross-validated R² / RMSE. Tasks differ from the paper's, so these show whether the same model comparison holds, not the paper's numbers.</p>
  <div class="scroll"><table><tr><th>Task</th>${M.map(k => `<th>${k}</th>`).join('')}</tr>${Object.entries(P.four_model_ica).map(([t, v]) => `<tr><td>${t}</td>${M.map(k => `<td class="${r2cls(v[k].R2)}">${f2(v[k].R2)} · ${f2(v[k].RMSE)}</td>`).join('')}</tr>`).join('')}</table></div></div>
  <div class="card" style="margin-top:14px"><h3>Settings</h3><ul class="read">${P.caveats.map(c => `<li>${c}</li>`).join('')}<li>Best hyperparameters on the first split: ${Object.entries(P.best_params_first_seed).map(([k, v]) => `${k}: ${Object.entries(v).map(([a, b]) => a + '=' + b).join(', ')}`).join('; ')}.</li></ul></div>`;
  chart('pp1', { type: 'bar', data: { labels: M, datasets: [{ label: 'paper', data: M.map(k => P.paper[k].R2), backgroundColor: '#8b9ab3' }, { label: 'ours, paper protocol (mean)', data: M.map(k => R[k].mean.R2), backgroundColor: '#4cc3ff' }, { label: 'ours, improved (mean)', data: M.map(k => R['improved:' + k].mean.R2), backgroundColor: '#3ddc97' }] }, options: { scales: { y: { title: { display: true, text: 'R²' } } } } });
  chart('pp2', { type: 'line', data: { labels: P.seeds.map(s => 'split ' + (s - P.seeds[0] + 1)), datasets: M.map((k, i) => ({ label: k, data: R[k].per_seed_r2, borderColor: COL[i], pointRadius: 3, tension: 0 })).concat([{ label: 'paper XGBoost 0.507', data: P.seeds.map(() => P.paper.XGBoost.R2), borderColor: '#8b9ab3', borderDash: [5, 5], pointRadius: 0 }]) }, options: { scales: { y: { title: { display: true, text: 'test R²' } } } } });
  const d = P.first_seed_predictions.XGBoost, mx = Math.max(...d.actual, ...d.pred) * 1.05;
  chart('pp3', { type: 'scatter', data: { datasets: [{ label: 'test cells', data: d.actual.map((a, i) => ({ x: a, y: d.pred[i], id: d.ids[i] })), backgroundColor: '#4cc3ffaa' }, { label: 'ideal', type: 'line', data: [{ x: 0, y: 0 }, { x: mx, y: mx }], borderColor: '#8b9ab3', borderDash: [5, 5], pointRadius: 0 }] }, options: { scales: { x: { title: { display: true, text: 'actual cycle life' } }, y: { title: { display: true, text: 'predicted' } } } } });
}

// ---------- Features ----------
async function features() {
  const R = st.R, ms = Object.keys(R.importance), cur = st.fm && R.importance[st.fm] ? st.fm : ms.includes('Gradient Boosting') ? 'Gradient Boosting' : ms[0];
  st.fm = cur;
  $('#view').innerHTML = `<h2>Features</h2><p class="sub">Impurity-based importance from the final model (relative to the top feature). Correlated features share credit, so read as a guide, not as causal.</p>
  <div class="row">Model: <select onchange="st.fm=this.value;features()">${ms.map(m => `<option ${m === cur ? 'selected' : ''}>${m}</option>`).join('')}</select></div><div class="card">${`<div class="chart" style="height:480px"><canvas id="fi"></canvas></div>`}</div>`;
  const top = R.importance[cur].slice(0, 15), cats = [...new Set(R.importance[cur].map(x => x.category))];
  chart('fi', { type: 'bar', data: { labels: top.map(x => x.feature), datasets: [{ data: top.map(x => x.importance), backgroundColor: top.map(x => COL[cats.indexOf(x.category) % COL.length]) }] },
    options: { indexAxis: 'y', plugins: { legend: { display: false }, tooltip: { callbacks: { afterLabel: c => top[c.dataIndex].category } } } } });
}

// ---------- Explorer ----------
async function explorer() {
  const tok = ++st.tok;
  const cells = await api(`/datasets/${st.ds}/cells`);
  if (tok !== st.tok) return;
  if (!st.cell || !cells.find(c => c.battery_id === st.cell)) st.cell = cells[0].battery_id;
  const d = await api(`/datasets/${st.ds}/cells/${encodeURIComponent(st.cell)}`), rec = d.record, cu = d.curves;
  if (tok !== st.tok || st.tab !== 'explorer') return;
  st.lastCurves = cu;
  $('#view').innerHTML = `<h2>Cell explorer</h2><div class="row">${dsSelect(st.ds, 'chDs')}<select onchange="st.cell=this.value;explorer()">${cells.map(c => `<option ${c.battery_id === st.cell ? 'selected' : ''}>${c.battery_id}</option>`).join('')}</select>
  <button class="btn sec" onclick="sendLive()">Send to Live predict →</button></div>
  <div class="grid g2"><div class="card"><h3>Prediction — ${st.cell}</h3>${rec ? `<p>${pill(rec.tier)} predicted life <b>${f0(rec.predicted_cycle_life)}</b> cycles (80% interval ${f0(rec.lo)}–${f0(rec.hi)}), predicted RUL ${f0(rec.predicted_rul)}${rec.actual_cycle_life != null ? `<br>Actual life <b>${f0(rec.actual_cycle_life)}</b> · actual tier ${pill(rec.actual_tier)}` : ''}</p><p class="mut">Source: ${rec.source}</p>` : ''}
   <div class="note ${d.in_training ? '' : 'info'}">${d.note}</div>
   <table><tr><th>Final model</th><th>Pred. life</th><th>Interval</th><th>RUL</th></tr>${Object.entries(d.live).map(([n, v]) => `<tr><td>${n}</td><td>${f0(v.predicted_cycle_life)}</td><td>${f0(v.lo)}–${f0(v.hi)}</td><td>${f0(v.predicted_rul)}</td></tr>`).join('')}</table></div>
  <div class="card"><h3>Capacity (QD, Ah)</h3>${cv('c1')}</div><div class="card"><h3>Internal resistance (IR, Ω)</h3>${cv('c2')}</div>
  <div class="card"><h3>Avg temperature (°C)</h3>${cv('c3')}</div><div class="card"><h3>Charge time (min)</h3>${cv('c4')}</div></div>`;
  [['c1', 'qd'], ['c2', 'ir'], ['c3', 'tavg'], ['c4', 'chargetime']].forEach(([id, k], i) => chart(id, { type: 'line', data: { labels: cu.cycle, datasets: [{ data: cu[k], borderColor: COL[i], pointRadius: 0, tension: .2 }] }, options: { plugins: { legend: { display: false } }, scales: { x: { title: { display: true, text: 'cycle' } } } } }));
}
window.chDs = async id => { st.ds = id; st.cell = null; explorer(); };
window.sendLive = () => { const c = st.lastCurves; st.liveBody = { cycle: c.cycle, IR: c.ir, QC: c.qc, QD: c.qd, Tavg: c.tavg, Tmin: c.tavg, Tmax: c.tavg, chargetime: c.chargetime }; go('live'); };

// ---------- Live predict ----------
async function live() {
  const ex = st.liveBody || { cycle: [], IR: [], QC: [], QD: [], Tavg: [], Tmin: [], Tmax: [], chargetime: [], C1: null, Q1: null, C2: null };
  $('#view').innerHTML = `<h2>Live predict</h2><p class="sub">POST per-cycle readings (≥10 cycles, ideally the first 50) to <code>/api/predict</code>. Tmin/Tmax default to Tavg when sent from the explorer.</p>
  <div class="row"><select id="lm">${st.R.models.map(n => `<option ${n === st.R.meta.best_model ? 'selected' : ''}>${n}</option>`).join('')}</select><button class="btn" onclick="runLive()">Predict</button></div>
  <div class="grid g2"><div class="card"><textarea id="lb">${JSON.stringify(ex)}</textarea></div><div class="card" id="lo"><p class="mut">Pick a cell in the explorer and press “Send to Live predict”, or paste JSON with series: cycle, IR, QC, QD, Tavg, Tmin, Tmax, chargetime (optional C1, Q1, C2).</p></div></div>`;
}
window.runLive = async () => {
  try {
    const body = JSON.parse($('#lb').value); body.model = $('#lm').value;
    const r = await api('/predict', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body) });
    $('#lo').innerHTML = `<h3>${r.model}</h3><p>${pill(r.tier)}</p><p>Predicted life <b>${f0(r.predicted_cycle_life)}</b> cycles<br>80% interval ${f0(r.lo)}–${f0(r.hi)}<br>Predicted RUL ${f0(r.predicted_rul)}<br><span class="mut">${r.n_cycles_used} cycles used</span></p>${r.warnings.map(w => `<div class="note">${w}</div>`).join('')}`;
  } catch (e) { $('#lo').innerHTML = `<div class="note">${e.message}</div>`; }
};

// ---------- Upload ----------
async function upload() {
  const ups = st.datasets.filter(d => !d.builtin);
  $('#view').innerHTML = `<h2>Upload data</h2><p class="sub">Add your own cycling data and see how the MIT-Stanford-trained model reads it (zero-shot). Include <code>cycle_life</code> to get validation metrics.</p>
  <div class="card"><p>Required columns: <code>battery_id, cycle, IR, QC, QD, Tavg, Tmin, Tmax, chargetime</code> · optional: <code>C1, Q1, C2, cycle_life</code>. One row per cell per cycle; ≥10 valid cycles per cell, ideally the first 50. <a href="${API}/api/template.csv" style="color:var(--ac)">Download template</a></p>
  <div class="row"><input type="file" id="fu" accept=".csv"><button class="btn" onclick="doUpload()">Upload & score</button></div></div>
  <div class="card" style="margin-top:14px"><h3>Uploaded datasets</h3>${ups.length ? `<table>${ups.map(u => `<tr><td>${u.name}</td><td>${u.n_cells} cells</td><td>${u.labeled ? 'labeled' : 'unlabeled'}</td><td><button class="btn sec" onclick="setDs('${u.id}');go('datasets')">Open</button> <button class="btn sec" onclick="delDs('${u.id}')">Delete</button></td></tr>`).join('')}</table>` : '<p class="mut">None yet.</p>'}</div>`;
}
window.doUpload = async () => {
  const f = $('#fu').files[0]; if (!f) return toast('Choose a CSV first');
  const fd = new FormData(); fd.append('file', f);
  try { const d = await api('/datasets/upload', { method: 'POST', body: fd }); await refreshDs(); st.ds = d.id; toast(`Loaded ${d.n_cells} cells`); go('datasets'); } catch (e) { toast(e.message); }
};
window.delDs = async id => { await api('/datasets/' + id, { method: 'DELETE' }); await refreshDs(); if (st.ds === id) st.ds = 'mit_all'; upload(); };
async function refreshDs() { st.datasets = await api('/datasets'); }

const VIEWS = { overview, datasets, models, generalization, protocols, nasa: nasaView, alt: altView, paper: paperView, features, explorer, live, upload };
(async () => { try { st.R = await api('/results'); await refreshDs(); nav(); go('overview'); } catch (e) { $('#view').innerHTML = `<div class="note">Cannot reach the API: ${e.message}. Start the backend (see README).</div>`; } })();
