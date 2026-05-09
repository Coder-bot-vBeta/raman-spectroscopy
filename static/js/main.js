/* ── Main application logic ── */

let currentSpectrumId = null;
let lastSpectrumObj   = null;
let toastEl           = null;

// ── Init ──────────────────────────────────────────────────────────────────────
document.addEventListener('DOMContentLoaded', () => {
  toastEl = new bootstrap.Toast(document.getElementById('app-toast'), { delay: 4000 });

  initDropZone();
  initFileInput();
  initModeButtons();
  loadAppStatus();

  document.getElementById('btn-reanalyze')?.addEventListener('click', () => {
    if (currentSpectrumId) runIdentification(currentSpectrumId);
  });
});

// ── Status ────────────────────────────────────────────────────────────────────
async function loadAppStatus() {
  try {
    const res  = await fetch('/api/status');
    const data = await res.json();

    const dbBadge  = document.getElementById('db-badge');
    const dbText   = document.getElementById('db-text');
    if (dbText) dbText.textContent = `${data.n_references} minerals`;
    if (dbBadge) dbBadge.className = 'status-pill online';

    const cnnBadge = document.getElementById('cnn-badge');
    const cnnText  = document.getElementById('cnn-text');
    // CNN badge
    if (data.cnn_available && data.cnn_reliable) {
      if (cnnText)  cnnText.textContent = 'CNN ready';
      if (cnnBadge) cnnBadge.className  = 'status-pill active';
      const cnnRadio = document.getElementById('mode-cnn');
      const cnnLabel = document.querySelector('label[for="mode-cnn"]');
      if (cnnRadio) cnnRadio.disabled = false;
      if (cnnLabel) cnnLabel.classList.remove('disabled');
      const hint = document.getElementById('cnn-hint');
      if (hint) hint.textContent = `CNN: trained on ${data.rruff_count} real spectra`;
    } else if (data.cnn_available && !data.cnn_reliable) {
      if (cnnText)  cnnText.textContent = 'CNN: synthetic only';
      if (cnnBadge) cnnBadge.className  = 'status-pill offline';
    } else {
      if (cnnText)  cnnText.textContent = 'CNN: not trained';
      if (cnnBadge) cnnBadge.className  = 'status-pill offline';
    }

    // RF badge
    const rfBadge = document.getElementById('rf-badge');
    const rfText  = document.getElementById('rf-text');
    if (data.rf_available && data.rf_reliable) {
      if (rfText)  rfText.textContent = 'RF ready';
      if (rfBadge) rfBadge.className  = 'status-pill active';
      const rfRadio = document.getElementById('mode-rf');
      const rfLabel = document.querySelector('label[for="mode-rf"]');
      if (rfRadio) rfRadio.disabled = false;
      if (rfLabel) rfLabel.classList.remove('disabled');
      const hint = document.getElementById('rf-hint');
      if (hint) hint.textContent = `RF: trained on ${data.rruff_count} real spectra`;
    } else {
      if (rfText)  rfText.textContent = 'RF: not trained';
      if (rfBadge) rfBadge.className  = 'status-pill offline';
    }
  } catch (e) {
    console.warn('Status fetch failed:', e);
  }
}

// ── Drop Zone ─────────────────────────────────────────────────────────────────
function initDropZone() {
  const zone = document.getElementById('drop-zone');
  if (!zone) return;
  zone.addEventListener('dragover',  e => { e.preventDefault(); zone.classList.add('drag-over'); });
  zone.addEventListener('dragleave', () => zone.classList.remove('drag-over'));
  zone.addEventListener('drop', e => {
    e.preventDefault();
    zone.classList.remove('drag-over');
    const file = e.dataTransfer.files[0];
    if (file) handleFile(file);
  });
}

function initFileInput() {
  document.getElementById('file-input')?.addEventListener('change', e => {
    if (e.target.files[0]) handleFile(e.target.files[0]);
  });
}

// ── File Handling ─────────────────────────────────────────────────────────────
async function handleFile(file) {
  clearError();
  showPlotSpinner(true);

  const formData = new FormData();
  formData.append('file', file);

  try {
    const res  = await fetch('/api/upload', { method: 'POST', body: formData });
    const data = await res.json();

    if (!res.ok) {
      showError(data.error || 'Upload failed.');
      showPlotSpinner(false);
      return;
    }

    currentSpectrumId    = data.spectrum_id;
    lastSpectrumObj      = data.spectrum;
    window._lastOverlays = [];

    document.getElementById('file-info')?.classList.remove('d-none');
    const fnEl = document.getElementById('file-name');
    if (fnEl) fnEl.textContent =
      `${file.name}  ·  ${data.n_points} pts  ·  ${data.wavenumber_range[0].toFixed(0)}–${data.wavenumber_range[1].toFixed(0)} cm⁻¹`;

    const sfEl = document.getElementById('spectrum-filename');
    if (sfEl) sfEl.textContent = file.name;

    renderSpectrum(data.spectrum, []);
    showPlotSpinner(false);
    await runIdentification(currentSpectrumId);
  } catch (err) {
    showError('Network error: ' + err.message);
    showPlotSpinner(false);
  }
}

// ── Identification ────────────────────────────────────────────────────────────
function skeletonRows(n = 3) {
  return Array.from({ length: n }, (_, i) =>
    `<div class="skeleton skeleton-line ${['w-80','w-60','w-40'][i % 3]}"></div>`
  ).join('');
}

async function runIdentification(sid) {
  const mode = document.querySelector('input[name="mode"]:checked')?.value || 'auto';

  // Show results section + loading skeletons
  showResultsSection(true);
  setScanLine(true);
  document.getElementById('matches-container').innerHTML  = skeletonRows(4);
  document.getElementById('mixture-container').innerHTML  = skeletonRows(3);

  try {
    const [idRes, mixRes] = await Promise.all([
      fetch('/api/identify', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ spectrum_id: sid, mode }),
      }),
      fetch('/api/mixture', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ spectrum_id: sid }),
      }),
    ]);

    setScanLine(false);
    const idData  = await idRes.json();
    const mixData = await mixRes.json();

    if (!idRes.ok) {
      if (idData.fallback) {
        showToast('CNN unavailable — falling back to spectral matching.', 'warning');
        return runIdentificationWithMode(sid, 'matching');
      }
      document.getElementById('matches-container').innerHTML =
        `<div style="color:var(--danger);font-size:0.78rem">${idData.error}</div>`;
      return;
    }

    // Mode badge
    const modeBadge = document.getElementById('mode-used-badge');
    if (modeBadge) {
      if (idData.mode_used === 'cnn') {
        modeBadge.textContent = 'CNN';
        modeBadge.className   = 'header-badge cnn';
      } else if (idData.mode_used === 'rf') {
        modeBadge.textContent = 'Random Forest';
        modeBadge.className   = 'header-badge rf';
      } else {
        modeBadge.textContent = 'Spectral Match';
        modeBadge.className   = 'header-badge matching';
      }
    }

    // Render
    renderMatches(idData.matches);

    if (idData.reference_overlays?.length) {
      window._lastOverlays = idData.reference_overlays;
      if (lastSpectrumObj) renderSpectrum(lastSpectrumObj, idData.reference_overlays);
    }

    renderMixture(mixData);

    if (idData.is_likely_mixture) {
      showToast('Spectrum may be a mixture — see Composition panel.', 'info');
    }

  } catch (err) {
    setScanLine(false);
    showError('Analysis error: ' + err.message);
  }
}

async function runIdentificationWithMode(sid, mode) {
  const radio = document.querySelector(`input[name="mode"][value="${mode}"]`);
  if (radio) radio.checked = true;
  await runIdentification(sid);
}

// ── Sample Spectra ────────────────────────────────────────────────────────────
async function loadSample(name) {
  const csv = generateSampleCSV(name);
  if (!csv) return;
  const file = new File([new Blob([csv], { type: 'text/csv' })], `${name}.csv`, { type: 'text/csv' });
  await handleFile(file);
}

function generateSampleCSV(type) {
  // Cover the full backend grid so high-wavenumber bands (Gypsum 3406,
  // Talc 3677, Kaolinite 3620/3695) aren't lost when those samples are added.
  const grid = [];
  for (let wn = 100; wn <= 4000; wn += 4) grid.push(wn);

  function L(x, c, a, w) {
    const g = w / 2;
    return a * g * g / ((x - c) ** 2 + g * g);
  }

  const lib = {
    quartz:  [[128,1.0,20],[206,0.35,18],[394,0.1,22],[464,10.0,12],[1082,0.07,28]],
    calcite: [[155,0.07,18],[282,0.10,18],[712,0.04,22],[1086,1.0,16]],
    pyrite:  [[343,0.5,10],[380,1.0,9],[430,0.3,14]],
  };

  const peaks = type === 'mixture'
    ? [...lib.quartz, ...lib.calcite.map(p => [p[0], p[1] * 0.4, p[2]])]
    : (lib[type] || lib.quartz);

  const rows = ['wavenumber,intensity'];
  for (const wn of grid) {
    let v = peaks.reduce((s, [c,a,w]) => s + L(wn, c, a, w), 0);
    v = Math.max(0, v + (Math.random() - 0.5) * 0.06);
    rows.push(`${wn},${v.toFixed(4)}`);
  }
  return rows.join('\n');
}

// ── Utilities ─────────────────────────────────────────────────────────────────
function initModeButtons() {
  document.querySelectorAll('input[name="mode"]').forEach(radio => {
    radio.addEventListener('change', () => {
      if (currentSpectrumId) runIdentification(currentSpectrumId);
    });
  });
}

function showResultsSection(show) {
  document.getElementById('results-section')?.classList.toggle('d-none', !show);
}

function showPlotSpinner(show) {
  document.getElementById('plot-spinner')?.classList.toggle('d-none', !show);
}

function setScanLine(show) {
  const el = document.getElementById('scan-line');
  if (el) el.style.display = show ? 'block' : 'none';
}

function showError(msg) {
  const el = document.getElementById('upload-error');
  if (el) { el.textContent = msg; el.classList.remove('d-none'); }
}

function clearError() {
  document.getElementById('upload-error')?.classList.add('d-none');
}

function showToast(msg, type = 'info') {
  const msgEl  = document.getElementById('toast-msg');
  const iconEl = document.getElementById('toast-icon');
  if (msgEl)  msgEl.textContent = msg;
  if (iconEl) {
    iconEl.className = type === 'warning' ? 'bi bi-exclamation-triangle'
                     : type === 'error'   ? 'bi bi-x-circle'
                     : 'bi bi-info-circle';
    iconEl.style.color = type === 'warning' ? 'var(--warning)'
                       : type === 'error'   ? 'var(--danger)'
                       : 'var(--accent)';
  }
  if (toastEl) toastEl.show();
}
