/* ── Plotly spectrum renderer ── */

const PLOTLY_LAYOUT = {
  paper_bgcolor: 'rgba(0,0,0,0)',
  plot_bgcolor:  '#f9f5ef',
  margin: { t: 16, b: 52, l: 58, r: 16 },
  font: { color: '#a09488', size: 11, family: 'Inter, system-ui, sans-serif' },
  xaxis: {
    title: { text: 'Raman Shift (cm⁻¹)', font: { size: 11, color: '#a09488' }, standoff: 8 },
    gridcolor: 'rgba(100,80,60,0.07)',
    linecolor: '#d6cfc2',
    zerolinecolor: '#d6cfc2',
    tickfont: { color: '#b8b0a4', size: 10 },
    tickcolor: '#d6cfc2',
  },
  yaxis: {
    title: { text: 'Intensity (norm.)', font: { size: 11, color: '#a09488' }, standoff: 8 },
    gridcolor: 'rgba(100,80,60,0.07)',
    linecolor: '#d6cfc2',
    zerolinecolor: '#d6cfc2',
    tickfont: { color: '#b8b0a4', size: 10 },
    tickcolor: '#d6cfc2',
  },
  legend: {
    bgcolor: 'rgba(249,245,239,0.9)',
    bordercolor: '#d6cfc2',
    borderwidth: 1,
    font: { size: 10, color: '#695e54' },
    x: 0.01, xanchor: 'left',
    y: 0.99, yanchor: 'top',
  },
  hovermode: 'x unified',
  hoverlabel: {
    bgcolor: '#f9f5ef',
    bordercolor: '#b8b0a4',
    font: { color: '#2a231c', size: 11, family: 'Inter, system-ui, sans-serif' },
  },
};

const PLOTLY_CONFIG = {
  responsive: true,
  displayModeBar: true,
  displaylogo: false,
  modeBarButtonsToRemove: ['select2d', 'lasso2d', 'autoScale2d'],
  toImageButtonOptions: { format: 'png', scale: 2 },
};

let _spectrumData = null;
let _overlayData  = null;

function renderSpectrum(spectrumObj, overlays = []) {
  _spectrumData = spectrumObj;
  _overlayData  = overlays;
  _rebuildPlot();
}

function updateOverlays(overlays) {
  _overlayData = overlays;
  _rebuildPlot();
}

function _rebuildPlot() {
  if (!_spectrumData) return;

  const showRaw       = document.getElementById('show-raw')?.checked ?? true;
  const showProcessed = document.getElementById('show-processed')?.checked ?? true;
  const showBaseline  = document.getElementById('show-baseline')?.checked ?? false;
  const showRefs      = document.getElementById('show-refs')?.checked ?? true;

  const traces = [];

  if (showRaw) {
    traces.push({
      x: _spectrumData.wavenumber,
      y: _spectrumData.raw,
      mode: 'lines',
      name: 'Raw',
      line: { color: 'rgba(124,92,62,0.35)', width: 1 },
      hovertemplate: 'Raw: %{y:.4f}<extra></extra>',
    });
  }

  if (showBaseline && _spectrumData.steps?.baseline) {
    const rawMax = Math.max(..._spectrumData.raw.filter(v => isFinite(v)));
    const blMax  = Math.max(..._spectrumData.steps.baseline.filter(v => isFinite(v)));
    const scale  = blMax > 1e-12 ? rawMax / blMax : 1;
    traces.push({
      x: _spectrumData.wavenumber,
      y: _spectrumData.steps.baseline.map(v => v * scale),
      mode: 'lines',
      name: 'Baseline',
      line: { color: 'rgba(138,64,64,0.55)', width: 1, dash: 'dot' },
      hovertemplate: 'Baseline: %{y:.4f}<extra></extra>',
    });
  }

  if (showProcessed) {
    traces.push({
      x: _spectrumData.wavenumber,
      y: _spectrumData.processed,
      mode: 'lines',
      name: 'Processed',
      line: { color: '#4d7a58', width: 2 },
      hovertemplate: 'Processed: %{y:.4f}<extra></extra>',
    });
  }

  const refColors = ['#a06030', '#6b4e72', '#4d6880', '#8a6520'];
  if (showRefs && _overlayData) {
    _overlayData.forEach((ref, i) => {
      traces.push({
        x: _spectrumData.wavenumber,
        y: ref.spectrum,
        mode: 'lines',
        name: ref.mineral,
        line: { color: refColors[i % refColors.length], width: 1.5, dash: 'dash' },
        opacity: 0.7,
        hovertemplate: `${ref.mineral}: %{y:.4f}<extra></extra>`,
      });
    });
  }

  const plotEl = document.getElementById('spectrum-plot');
  const placeholder = document.getElementById('plot-placeholder');
  if (placeholder) placeholder.style.display = 'none';
  if (plotEl) plotEl.style.display = 'block';

  if (plotEl._hasPlot) {
    Plotly.react('spectrum-plot', traces, PLOTLY_LAYOUT, PLOTLY_CONFIG);
  } else {
    Plotly.newPlot('spectrum-plot', traces, PLOTLY_LAYOUT, PLOTLY_CONFIG);
    plotEl._hasPlot = true;
  }
}

document.addEventListener('DOMContentLoaded', () => {
  ['show-raw', 'show-processed', 'show-baseline', 'show-refs'].forEach(id => {
    document.getElementById(id)?.addEventListener('change', () => _rebuildPlot());
  });
});
