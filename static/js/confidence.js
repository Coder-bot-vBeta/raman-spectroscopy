/* ── Match card & mixture bar rendering ── */

function getConfidenceColor(pct) {
  if (pct >= 75) return 'var(--success)';
  if (pct >= 50) return 'var(--warning)';
  return 'var(--danger)';
}

function getConfidenceClass(pct) {
  if (pct >= 75) return 'high';
  if (pct >= 50) return 'medium';
  return 'low';
}

function renderMatchCard(match, index) {
  const rankClass = index === 0 ? 'rank-1' : index === 1 ? 'rank-2' : index === 2 ? 'rank-3' : '';
  const color     = getConfidenceColor(match.confidence_pct);
  const confClass = getConfidenceClass(match.confidence_pct);
  const sourceTag = match.source === 'cnn'     ? `<span class="match-source-tag cnn">CNN</span>`
                 : match.source === 'rf'      ? `<span class="match-source-tag rf">RF</span>`
                 : `<span class="match-source-tag spectral">Spectral</span>`;

  const peaks = (match.peaks || []).slice(0, 6)
    .map(p => `<span class="peak-tag">${p} cm⁻¹</span>`)
    .join('');

  return `
  <div class="match-card ${rankClass}" data-rank="${index + 1}">
    <div class="match-rank-col">
      <span class="match-rank-num">#${index + 1}</span>
      <span class="match-rank-dot"></span>
    </div>
    <div class="match-body">
      <div class="match-top-row">
        <div style="min-width:0">
          <span class="match-name">${match.mineral}</span>
          <span class="match-formula-badge">${match.formula || ''}</span>
          ${sourceTag}
          ${match.description ? `<div class="match-description">${match.description}</div>` : ''}
          ${peaks ? `<div class="match-peaks">${peaks}</div>` : ''}
        </div>
        <div class="match-confidence-col">
          <div class="match-pct ${confClass}">${match.confidence_pct.toFixed(1)}%</div>
          <div class="match-pct-label">confidence</div>
        </div>
      </div>
      <div class="conf-bar-bg">
        <div class="conf-bar-fill"
             data-target="${match.confidence_pct}"
             style="background:${color}"></div>
      </div>
    </div>
  </div>`;
}

function renderMatches(matches) {
  const container = document.getElementById('matches-container');
  if (!matches || matches.length === 0) {
    container.innerHTML = `<div style="color:var(--text-muted);font-size:0.78rem;padding:0.25rem 0">No matches found.</div>`;
    return;
  }

  container.innerHTML = `<div class="matches-list">${matches.map((m, i) => renderMatchCard(m, i)).join('')}</div>`;

  // Animate bars after one frame
  requestAnimationFrame(() => {
    requestAnimationFrame(() => {
      container.querySelectorAll('.conf-bar-fill').forEach(el => {
        el.style.width = el.dataset.target + '%';
      });
    });
  });
}

function renderMixture(mixtureResult) {
  const container = document.getElementById('mixture-container');
  const badge     = document.getElementById('mixture-badge');

  if (!mixtureResult || mixtureResult.components.length === 0) {
    container.innerHTML = `<div style="color:var(--text-muted);font-size:0.78rem">Could not decompose mixture.</div>`;
    badge.textContent = '—';
    badge.className   = 'header-badge';
    return;
  }

  if (mixtureResult.is_mixture) {
    badge.textContent = `${mixtureResult.components.length} components`;
    badge.className   = 'header-badge mixture';
  } else {
    badge.textContent = 'Pure mineral';
    badge.className   = 'header-badge pure';
  }

  const rows = mixtureResult.components.map((c, i) => `
    <div class="mix-row">
      <div class="mix-label">
        <span class="mix-dot" style="background:${c.color}"></span>
        <span class="mix-name" title="${c.mineral}">${c.mineral}</span>
      </div>
      <div class="mix-bar-bg">
        <div class="mix-bar-fill" data-target="${c.fraction_pct}"
             style="background:${c.color}; opacity:0.85"></div>
      </div>
      <div class="mix-pct">${c.fraction_pct.toFixed(1)}%</div>
    </div>`).join('');

  const quality = mixtureResult.residual_norm < 0.3 ? '✓ Good fit'
                : mixtureResult.residual_norm < 0.6 ? '~ Moderate fit'
                : '⚠ Poor fit — possible unknown mineral';

  const residual = `
    <div class="residual-note">
      ${quality} &nbsp;·&nbsp; residual ${mixtureResult.residual_norm.toFixed(4)}
    </div>`;

  container.innerHTML = `<div class="mixture-bars">${rows}</div>${residual}`;

  requestAnimationFrame(() => {
    requestAnimationFrame(() => {
      container.querySelectorAll('.mix-bar-fill').forEach(el => {
        el.style.width = el.dataset.target + '%';
      });
    });
  });
}
