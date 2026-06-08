/* ═══════════════════════════════════════════════════════════════
   Amazon Intelligence Platform — Frontend Application JS
   ═══════════════════════════════════════════════════════════════ */

const API = 'http://localhost:8000';

// ─── State ─────────────────────────────────────────────────────
let state = {
  currentPanel: 'dashboard',
  selectedRating: null,
  selectedMode: 'simulate',
  activeRunId: null,
  pollingInterval: null,
  sentimentChart: null,
  starChart: null,
  lossChart: null,
  accChart: null,
  allProducts: [],
};

// ─── Chart.js Global Defaults ───────────────────────────────────
Chart.defaults.color = '#94A3B8';
Chart.defaults.borderColor = 'rgba(255,255,255,0.06)';
Chart.defaults.font.family = 'Inter, sans-serif';
Chart.defaults.font.size = 11;

const CHART_COLORS = {
  amber:  '#F59E0B',
  cyan:   '#06B6D4',
  green:  '#10B981',
  red:    '#EF4444',
  purple: '#8B5CF6',
  blue:   '#3B82F6',
};

// ─── Navigation ─────────────────────────────────────────────────
function switchPanel(panelId) {
  document.querySelectorAll('.panel').forEach(p => p.classList.remove('active'));
  document.querySelectorAll('.nav-item').forEach(n => n.classList.remove('active'));

  document.getElementById(`panel-${panelId}`)?.classList.add('active');
  document.getElementById(`nav-${panelId}`)?.classList.add('active');

  const titles = {
    dashboard: ['Analytics Dashboard', 'Real-time NLP insights'],
    analyzer:  ['Review Analyzer', 'NLP-powered sentiment & fake detection'],
    products:  ['Product Intelligence', 'AI summaries per product'],
    training:  ['Model Training', 'BERT fine-tuning with MLflow'],
  };

  const [title, sub] = titles[panelId] || ['Platform', ''];
  document.getElementById('page-title').textContent = title;
  document.getElementById('page-subtitle').textContent = sub;

  state.currentPanel = panelId;

  if (panelId === 'dashboard') loadDashboard();
  if (panelId === 'products') loadProductsPanel();
  if (panelId === 'training') loadRunsList();
}

document.querySelectorAll('.nav-item').forEach(nav => {
  nav.addEventListener('click', e => {
    e.preventDefault();
    switchPanel(nav.dataset.panel);
  });
});

// ─── API Helper ─────────────────────────────────────────────────
async function apiFetch(path, opts = {}) {
  const res = await fetch(`${API}${path}`, {
    headers: { 'Content-Type': 'application/json', ...opts.headers },
    ...opts,
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: 'Unknown error' }));
    throw new Error(err.detail || `HTTP ${res.status}`);
  }
  return res.json();
}

// ─── Dashboard ──────────────────────────────────────────────────
async function loadDashboard() {
  setRefreshing(true);
  try {
    const [overview, sentDist, products] = await Promise.all([
      apiFetch('/api/metrics/overview'),
      apiFetch('/api/metrics/sentiment-distribution'),
      apiFetch('/api/products'),
    ]);

    // KPIs
    animateCount('kpi-total-val', overview.total_reviews);
    document.getElementById('kpi-sentiment-val').textContent =
      overview.sentiment_positive_pct + '%';
    document.getElementById('kpi-sentiment-delta').textContent =
      `${overview.sentiment.positive} of ${overview.total_reviews} reviews`;
    animateCount('kpi-fake-val', overview.fake_reviews);
    document.getElementById('kpi-fake-delta').textContent =
      `${overview.fake_percentage}% of total`;
    animateCount('kpi-products-val', overview.total_products);

    // Sentiment donut
    buildSentimentChart(overview.sentiment);

    // Star bar chart
    buildStarChart(sentDist);

    // Products table
    state.allProducts = products;
    renderProductsTable(products);
  } catch (e) {
    console.error('Dashboard load error:', e);
    showToast('Could not connect to API. Is the backend running?', 'error');
  } finally {
    setRefreshing(false);
  }
}

function animateCount(id, target) {
  const el = document.getElementById(id);
  if (!el) return;
  const start = 0;
  const duration = 600;
  const startTime = performance.now();
  function update(now) {
    const progress = Math.min((now - startTime) / duration, 1);
    const eased = 1 - Math.pow(1 - progress, 3);
    el.textContent = Math.floor(eased * target).toLocaleString();
    if (progress < 1) requestAnimationFrame(update);
  }
  requestAnimationFrame(update);
}

function buildSentimentChart(sentiment) {
  const ctx = document.getElementById('sentimentChart');
  if (!ctx) return;
  if (state.sentimentChart) state.sentimentChart.destroy();

  state.sentimentChart = new Chart(ctx, {
    type: 'doughnut',
    data: {
      labels: ['Positive', 'Negative', 'Neutral'],
      datasets: [{
        data: [sentiment.positive, sentiment.negative, sentiment.neutral],
        backgroundColor: ['rgba(16,185,129,0.8)', 'rgba(239,68,68,0.8)', 'rgba(59,130,246,0.8)'],
        borderColor: ['#10B981', '#EF4444', '#3B82F6'],
        borderWidth: 1.5,
        hoverOffset: 8,
      }]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      cutout: '68%',
      plugins: {
        legend: { position: 'bottom', labels: { padding: 16, boxWidth: 12, boxHeight: 12 } },
        tooltip: { callbacks: { label: ctx => ` ${ctx.parsed} reviews` } }
      },
    }
  });
}

function buildStarChart(dist) {
  const ctx = document.getElementById('starChart');
  if (!ctx) return;
  if (state.starChart) state.starChart.destroy();

  const labels = ['1★', '2★', '3★', '4★', '5★'];
  const values = [dist[1], dist[2], dist[3], dist[4], dist[5]];
  const colors = ['#EF4444', '#F97316', '#F59E0B', '#84CC16', '#10B981'];

  state.starChart = new Chart(ctx, {
    type: 'bar',
    data: {
      labels,
      datasets: [{
        data: values,
        backgroundColor: colors.map(c => c + 'BB'),
        borderColor: colors,
        borderWidth: 1.5,
        borderRadius: 4,
      }]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: { legend: { display: false } },
      scales: {
        x: { grid: { display: false } },
        y: { beginAtZero: true, ticks: { precision: 0 } }
      }
    }
  });
}

function renderProductsTable(products) {
  const tbody = document.getElementById('products-tbody');
  if (!products.length) {
    tbody.innerHTML = '<tr><td colspan="7" class="loading-cell">No products found</td></tr>';
    return;
  }

  tbody.innerHTML = products.map(p => {
    const fakeClass = p.fake_review_pct > 30 ? 'high' : p.fake_review_pct > 10 ? 'mid' : 'low';
    const sentClass = p.sentiment_score > 0.6 ? 'positive' : p.sentiment_score < 0.3 ? 'negative' : 'neutral';
    const sentLabel = p.sentiment_score > 0.6 ? '😊 Positive' : p.sentiment_score < 0.3 ? '😞 Negative' : '😐 Neutral';

    return `<tr>
      <td><div class="product-name-cell" title="${p.name}">${p.name}</div></td>
      <td><span class="asin-code">${p.asin}</span></td>
      <td>${p.review_count.toLocaleString()}</td>
      <td><span class="rating-display">★ ${(p.avg_rating || 0).toFixed(1)}</span></td>
      <td><span class="sentiment-chip ${sentClass}">${sentLabel}</span></td>
      <td><span class="fake-pct-cell ${fakeClass}">${p.fake_review_pct}%</span></td>
      <td><button class="view-btn" onclick="openProductModal('${p.asin}')">View</button></td>
    </tr>`;
  }).join('');
}

// ─── Refresh Button ─────────────────────────────────────────────
function setRefreshing(on) {
  const btn = document.getElementById('refresh-btn');
  btn.classList.toggle('spinning', on);
}

document.getElementById('refresh-btn').addEventListener('click', () => {
  if (state.currentPanel === 'dashboard') loadDashboard();
  if (state.currentPanel === 'products') loadProductsPanel();
  if (state.currentPanel === 'training') loadRunsList();
});

// ─── Review Analyzer ────────────────────────────────────────────
const reviewTextEl = document.getElementById('review-text');
const charCountEl  = document.getElementById('char-count');

reviewTextEl.addEventListener('input', () => {
  const len = reviewTextEl.value.length;
  charCountEl.textContent = `${len} / 5000`;
  charCountEl.style.color = len > 4500 ? '#EF4444' : '#4B5563';
});

// Star selector
document.querySelectorAll('.star-btn').forEach(btn => {
  btn.addEventListener('click', () => {
    const val = parseInt(btn.dataset.val);
    state.selectedRating = state.selectedRating === val ? null : val;
    updateStarDisplay();
  });
});

function updateStarDisplay() {
  document.querySelectorAll('.star-btn').forEach(btn => {
    btn.classList.toggle('active', parseInt(btn.dataset.val) <= (state.selectedRating || 0));
  });
}

// Sample review buttons
document.querySelectorAll('.sample-btn').forEach(btn => {
  btn.addEventListener('click', () => {
    reviewTextEl.value = btn.dataset.review;
    state.selectedRating = parseInt(btn.dataset.rating) || null;
    updateStarDisplay();
    charCountEl.textContent = `${reviewTextEl.value.length} / 5000`;
  });
});

// Analyze button
document.getElementById('analyze-btn').addEventListener('click', analyzeReview);

async function analyzeReview() {
  const text = reviewTextEl.value.trim();
  if (text.length < 5) {
    showToast('Please enter at least 5 characters.', 'warning');
    return;
  }

  const btn = document.getElementById('analyze-btn');
  btn.disabled = true;

  // Show loading
  document.getElementById('results-placeholder').classList.add('hidden');
  document.getElementById('results-content').classList.add('hidden');
  document.getElementById('results-loading').classList.remove('hidden');

  try {
    const result = await apiFetch('/api/analyze', {
      method: 'POST',
      body: JSON.stringify({
        text,
        rating: state.selectedRating,
        verified_purchase: document.getElementById('verified-purchase').checked,
        helpful_votes: 0,
        total_votes: 0,
      }),
    });

    renderAnalysisResult(result);
  } catch (e) {
    showToast('Analysis failed: ' + e.message, 'error');
    document.getElementById('results-placeholder').classList.remove('hidden');
  } finally {
    document.getElementById('results-loading').classList.add('hidden');
    btn.disabled = false;
  }
}

function renderAnalysisResult(r) {
  // Show results
  document.getElementById('results-content').classList.remove('hidden');

  // Sentiment badge
  const badge = document.getElementById('sentiment-badge');
  const icons = { positive: '😊 Positive', negative: '😞 Negative', neutral: '😐 Neutral' };
  badge.textContent = icons[r.sentiment_label] || r.sentiment_label;
  badge.className = `sentiment-badge-large ${r.sentiment_label}`;

  document.getElementById('sentiment-score').textContent =
    (r.sentiment_score * 100).toFixed(1) + '%';
  document.getElementById('confidence-bar').style.width =
    (r.sentiment_score * 100) + '%';

  // Stars
  const starsEl = document.getElementById('stars-display');
  starsEl.innerHTML = Array.from({ length: 5 }, (_, i) =>
    `<span class="${i < r.star_class ? '' : 'empty'}">★</span>`
  ).join('');

  document.getElementById('model-badge').textContent = r.model_used;

  // Fake verdict
  const verdict = document.getElementById('fake-verdict');
  verdict.textContent = r.is_fake ? '⚠️ Suspicious Review' : '✅ Appears Genuine';
  verdict.className = `fake-verdict ${r.is_fake ? 'is-fake' : 'is-genuine'}`;

  const fakePct = (r.fake_probability * 100).toFixed(1) + '%';
  document.getElementById('fake-prob-pct').textContent = fakePct;

  const bar = document.getElementById('fake-prob-bar');
  bar.style.width = (r.fake_probability * 100) + '%';
  bar.className = `prob-bar-inner ${r.fake_probability > 0.5 ? 'high' : 'low'}`;

  document.getElementById('ml-score').textContent = (r.ml_score * 100).toFixed(0) + '%';
  document.getElementById('heuristic-score').textContent = (r.heuristic_score * 100).toFixed(0) + '%';

  // Reasons
  const reasonsList = document.getElementById('reasons-list');
  if (r.fake_reasons && r.fake_reasons.length) {
    reasonsList.innerHTML = r.fake_reasons.map(reason =>
      `<div class="reason-tag">⚠ ${reason}</div>`
    ).join('');
  } else {
    reasonsList.innerHTML = '<div class="reason-tag" style="background:rgba(16,185,129,0.08);color:#10B981;border-color:rgba(16,185,129,0.2)">✓ No suspicious patterns detected</div>';
  }
}

// ─── Products Panel ─────────────────────────────────────────────
async function loadProductsPanel() {
  const grid = document.getElementById('products-grid');
  grid.innerHTML = '<div class="loading-cell"><div class="spinner"></div> Loading products...</div>';

  try {
    const products = await apiFetch('/api/products');
    state.allProducts = products;
    renderProductsGrid(products);
  } catch (e) {
    grid.innerHTML = `<div class="loading-cell" style="color:#EF4444">Failed to load: ${e.message}</div>`;
  }
}

function renderProductsGrid(products) {
  const grid = document.getElementById('products-grid');

  if (!products.length) {
    grid.innerHTML = '<div class="loading-cell">No products found</div>';
    return;
  }

  grid.innerHTML = products.map(p => {
    const fakePct = p.fake_review_pct || 0;
    const fakeColor = fakePct > 30 ? '#EF4444' : fakePct > 10 ? '#F59E0B' : '#10B981';
    const sentPct = ((p.sentiment_score || 0) * 100).toFixed(0);

    return `
    <div class="product-card" onclick="openProductModal('${p.asin}')">
      <div class="product-card-name">${p.name}</div>
      <div class="product-card-asin">${p.asin}</div>
      <div class="product-stats">
        <div class="product-stat">
          <div class="product-stat-val" style="color:#F59E0B">★ ${(p.avg_rating || 0).toFixed(1)}</div>
          <div class="product-stat-label">Avg Rating</div>
        </div>
        <div class="product-stat">
          <div class="product-stat-val">${p.review_count}</div>
          <div class="product-stat-label">Reviews</div>
        </div>
        <div class="product-stat">
          <div class="product-stat-val" style="color:#10B981">${sentPct}%</div>
          <div class="product-stat-label">Positive</div>
        </div>
        <div class="product-stat">
          <div class="product-stat-val" style="color:${fakeColor}">${fakePct}%</div>
          <div class="product-stat-label">Fake</div>
        </div>
      </div>
      <div class="product-card-footer">
        <div class="fake-bar-mini">
          <div class="fake-bar-mini-fill" style="width:${Math.min(fakePct, 100)}%;background:${fakeColor}"></div>
        </div>
        <span class="fake-label-mini" style="color:${fakeColor}">${fakePct}% fake</span>
      </div>
    </div>`;
  }).join('');
}

// Product search filter
document.getElementById('product-search').addEventListener('input', e => {
  const q = e.target.value.toLowerCase();
  const filtered = state.allProducts.filter(p =>
    p.name.toLowerCase().includes(q) || p.asin.toLowerCase().includes(q)
  );
  renderProductsGrid(filtered);
});

// Product Modal
async function openProductModal(asin) {
  const modal = document.getElementById('product-modal');
  const loading = document.getElementById('modal-loading');
  const content = document.getElementById('modal-content');

  modal.classList.remove('hidden');
  loading.classList.remove('hidden');
  content.innerHTML = '';

  try {
    const [reviews, summary] = await Promise.all([
      apiFetch(`/api/products/${asin}/reviews?limit=10`),
      apiFetch(`/api/products/${asin}/summary`),
    ]);

    const product = state.allProducts.find(p => p.asin === asin);

    content.innerHTML = `
      <h2>${product?.name || asin}</h2>
      <div class="modal-asin">${asin} · ${product?.category || 'N/A'}</div>

      <div class="modal-sentiment-bar">
        <div class="sentiment-stat pos">
          <div class="sentiment-stat-count">${summary.sentiment_breakdown.positive}</div>
          <div class="sentiment-stat-label">Positive</div>
        </div>
        <div class="sentiment-stat neg">
          <div class="sentiment-stat-count">${summary.sentiment_breakdown.negative}</div>
          <div class="sentiment-stat-label">Negative</div>
        </div>
        <div class="sentiment-stat neu">
          <div class="sentiment-stat-count">${summary.sentiment_breakdown.neutral}</div>
          <div class="sentiment-stat-label">Neutral</div>
        </div>
      </div>

      <div class="summary-block">
        <h4>🤖 AI Summary</h4>
        <div class="summary-text">${summary.overall_summary}</div>
      </div>

      <div class="pros-cons-grid">
        <div class="pros-card">
          <h4>Pros</h4>
          <ul>${(summary.pros || []).map(p => `<li>${p}</li>`).join('')}</ul>
        </div>
        <div class="cons-card">
          <h4>Cons</h4>
          <ul>${(summary.cons || []).map(c => `<li>${c}</li>`).join('')}</ul>
        </div>
      </div>

      <div class="modal-reviews-section">
        <h4>Sample Reviews</h4>
        ${reviews.map(r => `
          <div class="review-item">
            <div class="review-item-meta">
              <span class="sentiment-chip ${r.sentiment_label}">${r.sentiment_label}</span>
              ${r.is_fake ? '<span class="review-fake-tag">⚠ Flagged</span>' : ''}
              ${r.rating ? `<span style="color:#F59E0B;font-size:0.75rem">★ ${r.rating}</span>` : ''}
            </div>
            ${r.text}
          </div>
        `).join('')}
      </div>
    `;
  } catch (e) {
    content.innerHTML = `<div style="color:#EF4444;padding:20px">Failed to load: ${e.message}</div>`;
  } finally {
    loading.classList.add('hidden');
  }
}

document.getElementById('modal-close').addEventListener('click', () => {
  document.getElementById('product-modal').classList.add('hidden');
});

document.getElementById('product-modal').addEventListener('click', e => {
  if (e.target === document.getElementById('product-modal')) {
    document.getElementById('product-modal').classList.add('hidden');
  }
});

// ─── Training Panel ─────────────────────────────────────────────
const epochsSlider = document.getElementById('epochs-slider');
epochsSlider.addEventListener('input', () => {
  document.getElementById('epochs-display').textContent = epochsSlider.value;
});

// Mode cards
document.querySelectorAll('.mode-card').forEach(card => {
  card.addEventListener('click', () => {
    document.querySelectorAll('.mode-card').forEach(c => c.classList.remove('selected'));
    card.classList.add('selected');
    state.selectedMode = card.dataset.mode;
  });
});

// Start training
document.getElementById('start-training-btn').addEventListener('click', startTraining);

async function startTraining() {
  const btn = document.getElementById('start-training-btn');
  btn.disabled = true;

  // Reset UI
  resetTrainingUI();

  try {
    const result = await apiFetch('/api/train', {
      method: 'POST',
      body: JSON.stringify({
        epochs: parseInt(epochsSlider.value),
        model_name: document.getElementById('model-select').value,
        simulate: state.selectedMode === 'simulate',
      }),
    });

    state.activeRunId = result.run_id;
    setRunStatus('running');
    appendLog(`[STARTED] Run ID: ${result.run_id}`);

    // Build empty charts
    initTrainingCharts(parseInt(epochsSlider.value));

    // Poll for updates
    if (state.pollingInterval) clearInterval(state.pollingInterval);
    state.pollingInterval = setInterval(() => pollTrainingStatus(result.run_id), 2500);

    showToast(`Training started! Run ID: ${result.run_id}`, 'success');
    loadRunsList();
  } catch (e) {
    showToast('Failed to start training: ' + e.message, 'error');
    appendLog(`[ERROR] ${e.message}`);
  } finally {
    btn.disabled = false;
  }
}

async function pollTrainingStatus(runId) {
  try {
    const status = await apiFetch(`/api/train/${runId}/status`);
    updateTrainingMetrics(status);
    updateTrainingCharts(status.metrics_history || []);
    setRunStatus(status.status);

    // Sync logs
    const logEl = document.getElementById('log-terminal');
    if (status.logs) logEl.textContent = status.logs;
    logEl.scrollTop = logEl.scrollHeight;

    if (status.status === 'completed' || status.status === 'failed') {
      clearInterval(state.pollingInterval);
      state.pollingInterval = null;
      showToast(`Training ${status.status}!`, status.status === 'completed' ? 'success' : 'error');
      loadRunsList();
    }
  } catch (e) {
    console.error('Poll error:', e);
  }
}

function resetTrainingUI() {
  ['m-epoch','m-train-loss','m-val-loss','m-accuracy','m-f1'].forEach(id => {
    document.getElementById(id).textContent = '—';
  });
  document.getElementById('log-terminal').textContent = 'Initializing training...\n';
  setRunStatus('idle');
}

function setRunStatus(status) {
  const badge = document.getElementById('run-status-badge');
  badge.textContent = status.charAt(0).toUpperCase() + status.slice(1);
  badge.className = `run-status-badge ${status}`;
}

function updateTrainingMetrics(status) {
  if (status.current_epoch > 0) {
    document.getElementById('m-epoch').textContent = `${status.current_epoch}/${status.epochs}`;
    document.getElementById('m-train-loss').textContent = (status.train_loss || 0).toFixed(4);
    document.getElementById('m-val-loss').textContent = (status.val_loss || 0).toFixed(4);
    document.getElementById('m-accuracy').textContent = ((status.accuracy || 0) * 100).toFixed(1) + '%';
    document.getElementById('m-f1').textContent = ((status.f1_score || 0) * 100).toFixed(1) + '%';
  }
}

function appendLog(text) {
  const logEl = document.getElementById('log-terminal');
  logEl.textContent += text + '\n';
  logEl.scrollTop = logEl.scrollHeight;
}

document.getElementById('log-clear').addEventListener('click', () => {
  document.getElementById('log-terminal').textContent = '';
});

// Training Charts
function initTrainingCharts(epochs) {
  const labels = Array.from({ length: epochs }, (_, i) => `Epoch ${i + 1}`);

  if (state.lossChart) state.lossChart.destroy();
  if (state.accChart) state.accChart.destroy();

  state.lossChart = new Chart(document.getElementById('lossChart'), {
    type: 'line',
    data: {
      labels,
      datasets: [
        {
          label: 'Train Loss',
          data: new Array(epochs).fill(null),
          borderColor: CHART_COLORS.amber,
          backgroundColor: 'rgba(245,158,11,0.1)',
          tension: 0.4,
          fill: true,
          pointRadius: 5,
          pointHoverRadius: 7,
        },
        {
          label: 'Val Loss',
          data: new Array(epochs).fill(null),
          borderColor: CHART_COLORS.red,
          backgroundColor: 'rgba(239,68,68,0.1)',
          tension: 0.4,
          fill: true,
          pointRadius: 5,
          pointHoverRadius: 7,
        }
      ]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: { legend: { position: 'bottom' } },
      scales: { y: { beginAtZero: false } },
      animation: { duration: 400 }
    }
  });

  state.accChart = new Chart(document.getElementById('accChart'), {
    type: 'line',
    data: {
      labels,
      datasets: [
        {
          label: 'Accuracy',
          data: new Array(epochs).fill(null),
          borderColor: CHART_COLORS.cyan,
          backgroundColor: 'rgba(6,182,212,0.1)',
          tension: 0.4,
          fill: true,
          pointRadius: 5,
          pointHoverRadius: 7,
        },
        {
          label: 'F1 Score',
          data: new Array(epochs).fill(null),
          borderColor: CHART_COLORS.purple,
          backgroundColor: 'rgba(139,92,246,0.1)',
          tension: 0.4,
          fill: true,
          pointRadius: 5,
          pointHoverRadius: 7,
        }
      ]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: { legend: { position: 'bottom' } },
      scales: { y: { min: 0, max: 1 } },
      animation: { duration: 400 }
    }
  });
}

function updateTrainingCharts(history) {
  if (!state.lossChart || !state.accChart || !history.length) return;

  history.forEach((m, i) => {
    state.lossChart.data.datasets[0].data[i] = m.train_loss;
    state.lossChart.data.datasets[1].data[i] = m.val_loss;
    state.accChart.data.datasets[0].data[i] = m.accuracy;
    state.accChart.data.datasets[1].data[i] = m.f1_score;
  });

  state.lossChart.update('none');
  state.accChart.update('none');
}

// Training Runs List
async function loadRunsList() {
  const list = document.getElementById('runs-list');
  try {
    const runs = await apiFetch('/api/train/runs');
    if (!runs.length) {
      list.innerHTML = '<div class="empty-runs">No training runs yet</div>';
      return;
    }
    list.innerHTML = runs.map(r => `
      <div class="run-item">
        <div>
          <span class="run-item-id">#${r.run_id}</span>
          <span style="font-size:0.68rem;color:#4B5563;margin-left:8px">${r.model_name?.replace('distilbert-base-uncased', 'DistilBERT') || ''}</span>
        </div>
        <span class="run-item-status ${r.status}">${r.status}</span>
      </div>
    `).join('');
  } catch (e) {
    list.innerHTML = '<div class="empty-runs" style="color:#EF4444">Failed to load runs</div>';
  }
}

// ─── Toast Notifications ─────────────────────────────────────────
function showToast(msg, type = 'info') {
  const existing = document.querySelector('.toast');
  if (existing) existing.remove();

  const colors = {
    success: { bg: 'rgba(16,185,129,0.15)', border: 'rgba(16,185,129,0.4)', color: '#10B981' },
    error:   { bg: 'rgba(239,68,68,0.15)',  border: 'rgba(239,68,68,0.4)',  color: '#EF4444' },
    warning: { bg: 'rgba(245,158,11,0.15)', border: 'rgba(245,158,11,0.4)', color: '#F59E0B' },
    info:    { bg: 'rgba(6,182,212,0.15)',  border: 'rgba(6,182,212,0.4)',  color: '#06B6D4' },
  };
  const c = colors[type] || colors.info;

  const toast = document.createElement('div');
  toast.className = 'toast';
  toast.style.cssText = `
    position:fixed;bottom:24px;right:24px;z-index:9999;
    background:${c.bg};border:1px solid ${c.border};color:${c.color};
    padding:12px 20px;border-radius:10px;font-size:0.85rem;font-weight:500;
    box-shadow:0 8px 24px rgba(0,0,0,0.4);backdrop-filter:blur(12px);
    animation:slideUp 0.3s ease;max-width:360px;line-height:1.4;
  `;
  toast.textContent = msg;
  document.body.appendChild(toast);

  // Inject keyframe if not already present
  if (!document.getElementById('toast-styles')) {
    const style = document.createElement('style');
    style.id = 'toast-styles';
    style.textContent = `@keyframes slideUp { from { transform: translateY(20px); opacity: 0; } to { transform: none; opacity: 1; } }`;
    document.head.appendChild(style);
  }

  setTimeout(() => toast.remove(), 4000);
}

// ─── Load More Products ─────────────────────────────────────────
document.getElementById('load-more-products')?.addEventListener('click', () => {
  showToast('All products already loaded', 'info');
});

// ─── Init ────────────────────────────────────────────────────────
document.addEventListener('DOMContentLoaded', () => {
  loadDashboard();
  // Check API health
  apiFetch('/api/health')
    .then(() => showToast('Connected to Amazon Intelligence API ✓', 'success'))
    .catch(() => showToast('API not reachable — start the backend server', 'error'));
});
