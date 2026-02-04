/**
 * Nova Intelligence Agent - Frontend JavaScript
 * Handles voice input, API calls, feature toggles, and result display
 */

const API_BASE = '/api';

// DOM Elements
const micBtn = document.getElementById('micBtn');
const textInput = document.getElementById('textInput');
const sendBtn = document.getElementById('sendBtn');
const status = document.getElementById('status');
const planOutput = document.getElementById('planOutput');
const intelOutput = document.getElementById('intelOutput');
const newsOutput = document.getElementById('newsOutput');

// Feature toggles state
const features = {
    news: true,
    summary: false,
    sentiment: false,
    trends: false
};

// Last result data for download
let lastResultData = null;
let lastExecutionMeta = null;

// Search history (persisted to localStorage)
const MAX_HISTORY = 10;
let searchHistory = JSON.parse(localStorage.getItem('novaSearchHistory') || '[]');

function saveToHistory(query) {
    // Remove if already exists
    searchHistory = searchHistory.filter(h => h.query !== query);
    // Add to front
    searchHistory.unshift({
        query: query,
        timestamp: new Date().toISOString(),
        features: { ...features }
    });
    // Keep max
    if (searchHistory.length > MAX_HISTORY) {
        searchHistory = searchHistory.slice(0, MAX_HISTORY);
    }
    localStorage.setItem('novaSearchHistory', JSON.stringify(searchHistory));
    renderHistory();
}

function renderHistory() {
    const historyContainer = document.getElementById('historyList');
    if (!historyContainer) return;

    if (searchHistory.length === 0) {
        historyContainer.innerHTML = '<p class="history-empty">No recent searches</p>';
        return;
    }

    historyContainer.innerHTML = searchHistory.map((h, i) => `
        <div class="history-item" data-index="${i}">
            <span class="history-query">${h.query}</span>
            <span class="history-time">${formatTime(h.timestamp)}</span>
        </div>
    `).join('');

    // Add click handlers
    historyContainer.querySelectorAll('.history-item').forEach(item => {
        item.addEventListener('click', () => {
            const idx = parseInt(item.dataset.index);
            const h = searchHistory[idx];
            textInput.value = h.query;
            sendCommand(h.query);
        });
    });
}

function formatTime(isoString) {
    const date = new Date(isoString);
    const now = new Date();
    const diffMs = now - date;
    const diffMins = Math.floor(diffMs / 60000);

    if (diffMins < 1) return 'just now';
    if (diffMins < 60) return `${diffMins}m ago`;
    if (diffMins < 1440) return `${Math.floor(diffMins / 60)}h ago`;
    return date.toLocaleDateString();
}

// Speech Recognition Setup
const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
let recognition = null;
let isListening = false;

if (SpeechRecognition) {
    recognition = new SpeechRecognition();
    recognition.continuous = false;
    recognition.interimResults = false;
    recognition.lang = 'en-US';

    recognition.onstart = () => {
        isListening = true;
        micBtn.classList.add('listening');
        setStatus('🎤 Listening...', 'loading');
    };

    recognition.onresult = (event) => {
        const transcript = event.results[0][0].transcript;
        textInput.value = transcript;
        setStatus(`Heard: "${transcript}"`, 'success');
        setTimeout(() => sendCommand(transcript), 500);
    };

    recognition.onerror = (event) => {
        console.error('Speech error:', event.error);
        setStatus(`Voice error: ${event.error}`, 'error');
        stopListening();
    };

    recognition.onend = () => {
        stopListening();
    };
}

// Event Listeners
micBtn.addEventListener('click', toggleListening);
sendBtn.addEventListener('click', () => {
    const text = textInput.value.trim();
    if (text) sendCommand(text);
});

textInput.addEventListener('keypress', (e) => {
    if (e.key === 'Enter') {
        const text = textInput.value.trim();
        if (text) sendCommand(text);
    }
});

// Feature toggle badges
document.querySelectorAll('.toggle-badge').forEach(badge => {
    badge.addEventListener('click', () => {
        const feature = badge.dataset.feature;
        if (feature === 'news') return; // News is always on

        features[feature] = !features[feature];
        badge.classList.toggle('active', features[feature]);

        updateStatusHint();
    });
});

// Example chips
document.querySelectorAll('.chip').forEach(chip => {
    chip.addEventListener('click', () => {
        const cmd = chip.dataset.cmd;
        textInput.value = cmd;
        sendCommand(cmd);
    });
});

// Functions
function updateStatusHint() {
    const active = Object.entries(features)
        .filter(([k, v]) => v)
        .map(([k]) => k);
    setStatus(`Features: ${active.join(', ')}. Enter a topic!`);
}

function toggleListening() {
    if (!recognition) {
        setStatus('Voice not supported in this browser', 'error');
        return;
    }

    if (isListening) {
        recognition.stop();
    } else {
        try {
            recognition.start();
        } catch (e) {
            console.error('Failed to start recognition:', e);
        }
    }
}

function stopListening() {
    isListening = false;
    micBtn.classList.remove('listening');
}

function setStatus(message, type = '') {
    status.textContent = message;
    status.className = 'status';
    if (type) status.classList.add(type);
}

function buildCommand(topic) {
    // Build a natural language command based on toggles
    let cmd = topic;
    const extras = [];

    if (features.summary) extras.push('summarize');
    if (features.sentiment) extras.push('sentiment analysis');
    if (features.trends) extras.push('trends');

    if (extras.length > 0) {
        cmd = `${topic} with ${extras.join(' and ')}`;
    }

    return cmd;
}

async function sendCommand(topic) {
    const fullCommand = buildCommand(topic);
    sendBtn.disabled = true;
    clearResults();

    // Save to history
    saveToHistory(topic);

    // Build expected steps based on toggles
    const steps = buildExpectedSteps();

    // Show execution pipeline overlay
    showExecutionPipeline(steps, topic);

    const startTime = Date.now();

    try {
        // Simulate step progression (news_fetcher always first)
        await simulateStepProgress(steps, 0, 'news_fetcher');

        const response = await fetch(`${API_BASE}/command`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ text: fullCommand })
        });

        if (!response.ok) {
            throw new Error(`HTTP ${response.status}`);
        }

        const data = await response.json();

        // Mark all steps complete based on actual result
        await markStepsFromResult(steps, data.result);

        // Show summary
        const elapsed = ((Date.now() - startTime) / 1000).toFixed(1);
        showExecutionSummary(steps, elapsed, data.result?.errors?.length || 0);

        // Wait a bit then hide overlay and show results
        await delay(1500);
        hideExecutionPipeline();

        displayResults(data);
        setStatus('✅ Intelligence report ready!', 'success');

    } catch (error) {
        console.error('API error:', error);
        markStepFailed(steps, 0);
        await delay(1000);
        hideExecutionPipeline();
        setStatus(`❌ Error: ${error.message}`, 'error');
    } finally {
        sendBtn.disabled = false;
    }
}

// Build expected steps based on current feature toggles
function buildExpectedSteps() {
    const steps = [
        { id: 'news_fetcher', name: 'Fetching News', icon: '📰', status: 'pending' }
    ];

    if (features.summary) {
        steps.push({ id: 'summarizer', name: 'Generating Summary', icon: '🧠', status: 'pending' });
    }
    if (features.sentiment) {
        steps.push({ id: 'sentiment', name: 'Sentiment Analysis', icon: '💭', status: 'pending' });
    }
    if (features.trends) {
        steps.push({ id: 'trends', name: 'Trend Extraction', icon: '📊', status: 'pending' });
    }
    if (features.export) {
        steps.push({ id: 'exporter', name: 'Export Report', icon: '💾', status: 'pending' });
    }

    return steps;
}

// Show the execution pipeline overlay
function showExecutionPipeline(steps, topic) {
    const overlay = document.getElementById('executionOverlay');
    const pipelineSteps = document.getElementById('pipelineSteps');
    const strategy = document.getElementById('executionStrategy');
    const summary = document.getElementById('executionSummary');

    // Set strategy text
    const toolNames = steps.map(s => s.name.toLowerCase()).join(' → ');
    strategy.textContent = `Agent Strategy: ${toolNames}`;

    // Hide summary
    summary.classList.add('hidden');

    // Render steps
    pipelineSteps.innerHTML = steps.map((step, i) => `
        <div class="pipeline-step pending" data-step-id="${step.id}">
            <div class="step-icon pending">⏳</div>
            <div class="step-info">
                <div class="step-name">${step.icon} ${step.name}</div>
                <div class="step-status">Waiting...</div>
            </div>
            <div class="step-time"></div>
        </div>
    `).join('');

    // Show overlay
    overlay.classList.remove('hidden');
    overlay.classList.remove('fade-out');

    // Start first step immediately
    updateStepStatus(steps[0].id, 'active', 'Running...');
}

// Update a step's visual status
function updateStepStatus(stepId, status, statusText, time = null) {
    const stepEl = document.querySelector(`[data-step-id="${stepId}"]`);
    if (!stepEl) return;

    // Update class
    stepEl.className = `pipeline-step ${status}`;

    // Update icon
    const iconEl = stepEl.querySelector('.step-icon');
    iconEl.className = `step-icon ${status}`;

    const icons = {
        pending: '⏳',
        active: '⚡',
        completed: '✓',
        failed: '✗',
        skipped: '⊘'
    };
    iconEl.textContent = icons[status] || '⏳';

    // Update status text
    stepEl.querySelector('.step-status').textContent = statusText;

    // Update time if provided
    if (time) {
        stepEl.querySelector('.step-time').textContent = time;
    }
}

// Simulate step progression with delays
async function simulateStepProgress(steps, startIdx, currentTool) {
    const stepDelay = 800; // ms between visual updates

    for (let i = startIdx; i < steps.length; i++) {
        const step = steps[i];

        // Mark current as active
        updateStepStatus(step.id, 'active', 'Processing...');

        // Wait a bit to show activity
        await delay(stepDelay);
    }
}

// Mark steps based on actual API result
async function markStepsFromResult(steps, result) {
    if (!result) return;

    const toolResults = result.tools_executed || [];
    const skipped = result.skipped || [];

    for (const step of steps) {
        const executed = toolResults.find(t => t.tool === step.id);
        const wasSkipped = skipped.find(s => s.tool === step.id);

        if (executed) {
            if (executed.success) {
                updateStepStatus(step.id, 'completed', 'Done!', executed.retries > 0 ? `+${executed.retries} retries` : '');
            } else {
                updateStepStatus(step.id, 'failed', executed.error || 'Failed');
            }
        } else if (wasSkipped) {
            updateStepStatus(step.id, 'skipped', wasSkipped.reason || 'Skipped');
        } else {
            updateStepStatus(step.id, 'completed', 'Done!');
        }

        await delay(200); // Stagger the visual updates
    }
}

// Mark a specific step as failed
function markStepFailed(steps, idx) {
    if (steps[idx]) {
        updateStepStatus(steps[idx].id, 'failed', 'Error occurred');
    }
}

// Show the execution summary
function showExecutionSummary(steps, elapsed, errorCount) {
    const summary = document.getElementById('executionSummary');
    const summaryText = document.getElementById('summaryText');
    const summaryIcon = summary.querySelector('.summary-icon');

    const completedCount = steps.filter(s =>
        document.querySelector(`[data-step-id="${s.id}"]`)?.classList.contains('completed')
    ).length;

    if (errorCount === 0) {
        summaryIcon.textContent = '✅';
        summaryText.textContent = `${completedCount} tools • ${elapsed}s • No errors`;
        summary.style.borderColor = 'var(--success)';
        summary.style.background = 'rgba(34, 197, 94, 0.1)';
    } else {
        summaryIcon.textContent = '⚠️';
        summaryText.textContent = `${completedCount} tools • ${elapsed}s • ${errorCount} error(s)`;
        summary.style.borderColor = 'var(--warning)';
        summary.style.background = 'rgba(234, 179, 8, 0.1)';
    }

    // Save state for summary chip
    lastExecutionState = { steps, elapsed, errorCount };

    summary.classList.remove('hidden');
}

// Last execution state for chip expansion
let lastExecutionState = { steps: [], elapsed: 0, errorCount: 0 };

// Hide the execution pipeline overlay and show summary chip
function hideExecutionPipeline() {
    const overlay = document.getElementById('executionOverlay');
    overlay.classList.add('fade-out');

    setTimeout(() => {
        overlay.classList.add('hidden');
        overlay.classList.remove('fade-out');
        // Show the collapsed summary chip
        showSummaryChip();
    }, 500);
}

// Show the summary chip at bottom of screen
function showSummaryChip() {
    const chip = document.getElementById('summaryChip');
    const chipText = document.getElementById('chipText');
    const chipIcon = document.getElementById('chipIcon');

    if (!chip || !chipText) return;

    const { steps, elapsed, errorCount } = lastExecutionState;
    const completedCount = steps.filter(s =>
        document.querySelector(`[data-step-id="${s.id}"]`)?.classList.contains('completed')
    ).length;

    // Set chip content
    if (errorCount === 0) {
        chipIcon.textContent = '✅';
        chipText.textContent = `${completedCount} tools • ${elapsed}s • Success`;
        chip.classList.remove('has-errors');
    } else {
        chipIcon.textContent = '⚠️';
        chipText.textContent = `${completedCount} tools • ${elapsed}s • ${errorCount} error(s)`;
        chip.classList.add('has-errors');
    }

    // Show with animation
    chip.classList.remove('hidden');
    chip.classList.add('visible');
}

// Hide the summary chip
function hideSummaryChip() {
    const chip = document.getElementById('summaryChip');
    if (chip) {
        chip.classList.remove('visible');
        chip.classList.add('hidden');
    }
}

// Expand from chip back to full overlay
function expandFromChip() {
    hideSummaryChip();
    const overlay = document.getElementById('executionOverlay');
    overlay.classList.remove('hidden');
    overlay.classList.remove('fade-out');
}

// Summary chip event handlers
document.getElementById('chipExpandBtn')?.addEventListener('click', expandFromChip);
document.getElementById('chipDismissBtn')?.addEventListener('click', hideSummaryChip);

// Utility delay function
function delay(ms) {
    return new Promise(resolve => setTimeout(resolve, ms));
}

// View trace button handler
document.getElementById('viewTraceBtn')?.addEventListener('click', () => {
    // Keep overlay visible longer for inspection
    document.getElementById('executionSummary').classList.add('hidden');
});

// ===== DYNAMIC PANEL VISIBILITY =====

// Panel configuration - maps toggle IDs to panel selectors
const PANEL_CONFIG = {
    'toggleSummary': { panel: '.intel-panel', name: 'Intelligence' },
    'toggleSentiment': { panel: '.intel-panel', name: 'Sentiment' },
    'toggleNews': { panel: '.news-panel', name: 'News' },
    'toggleTrends': { panel: '.intel-panel', name: 'Trends' }
};

// Update panel visibility based on active toggles
function updatePanelVisibility() {
    const resultsGrid = document.querySelector('.results-grid');
    const intelPanel = document.querySelector('.intel-panel');
    const newsPanel = document.querySelector('.news-panel');

    if (!resultsGrid) return;

    // Check which toggles are active
    const summaryActive = document.getElementById('toggleSummary')?.checked ?? true;
    const sentimentActive = document.getElementById('toggleSentiment')?.checked ?? true;
    const newsActive = document.getElementById('toggleNews')?.checked ?? true;
    const trendsActive = document.getElementById('toggleTrends')?.checked ?? true;

    // Intel panel shows if summary, sentiment, or trends are active
    const showIntel = summaryActive || sentimentActive || trendsActive;

    // Animate intel panel
    if (intelPanel) {
        if (showIntel && intelPanel.classList.contains('panel-hidden')) {
            intelPanel.classList.remove('panel-hidden');
            intelPanel.classList.add('panel-entering');
            setTimeout(() => intelPanel.classList.remove('panel-entering'), 500);
        } else if (!showIntel && !intelPanel.classList.contains('panel-hidden')) {
            intelPanel.classList.add('panel-exiting');
            setTimeout(() => {
                intelPanel.classList.add('panel-hidden');
                intelPanel.classList.remove('panel-exiting');
            }, 300);
        }
    }

    // Animate news panel
    if (newsPanel) {
        if (newsActive && newsPanel.classList.contains('panel-hidden')) {
            newsPanel.classList.remove('panel-hidden');
            newsPanel.classList.add('panel-entering');
            setTimeout(() => newsPanel.classList.remove('panel-entering'), 500);
        } else if (!newsActive && !newsPanel.classList.contains('panel-hidden')) {
            newsPanel.classList.add('panel-exiting');
            setTimeout(() => {
                newsPanel.classList.add('panel-hidden');
                newsPanel.classList.remove('panel-exiting');
            }, 300);
        }
    }

    // Check if any panels are visible
    const anyVisible = showIntel || newsActive;

    // Show/hide empty state message
    let emptyState = resultsGrid.querySelector('.empty-state');
    if (!anyVisible) {
        if (!emptyState) {
            emptyState = document.createElement('div');
            emptyState.className = 'empty-state';
            emptyState.innerHTML = `
                <div class="empty-icon">🎛️</div>
                <p>Enable features to see results</p>
            `;
            resultsGrid.appendChild(emptyState);
        }
        emptyState.classList.add('visible');
    } else if (emptyState) {
        emptyState.classList.remove('visible');
        setTimeout(() => emptyState.remove(), 300);
    }
}

// Attach toggle listeners for dynamic panel updates
function initPanelToggles() {
    const toggleIds = ['toggleSummary', 'toggleSentiment', 'toggleNews', 'toggleTrends'];
    toggleIds.forEach(id => {
        const toggle = document.getElementById(id);
        if (toggle) {
            toggle.addEventListener('change', updatePanelVisibility);
        }
    });
}

function clearResults() {
    intelOutput.innerHTML = '';
    newsOutput.innerHTML = '';
}

function displayResults(data) {
    if (data.result && data.result.data) {
        // Save for download
        lastResultData = data.result.data;
        // Save execution metadata for Package Builder
        lastExecutionMeta = {
            tools_executed: data.result.tools_executed || [],
            errors: data.result.errors || [],
            fallbacks_used: data.result.fallbacks_used || [],
            regenerated: data.result.regenerated || [],
            skipped: data.result.skipped || [],
            success: data.result.success ?? true
        };
        displayIntelligence(data.result.data);
    }
}

function displayIntelligence(data) {
    let html = '';

    // Summary
    if (data.summary) {
        const summaryText = data.summary.summary || (typeof data.summary === 'string' ? data.summary : '');
        html += `
            <div class="intel-section">
                <h4>📝 Summary</h4>
                <p>${summaryText}</p>
                ${data.summary.key_points ? `
                    <ul>
                        ${data.summary.key_points.map(p => `<li>${p}</li>`).join('')}
                    </ul>
                ` : ''}
            </div>
        `;
    }

    // Sentiment Intelligence V2
    if (data.sentiment) {
        const s = data.sentiment;
        const scorePercent = Math.round((s.score || 0.5) * 100);
        const sentimentClass = s.overall || 'neutral';
        const directionIcon = s.direction === 'improving' ? '📈' : s.direction === 'deteriorating' ? '📉' : '➡️';
        const momentumIcon = s.momentum_strength === 'strong' ? '🚀' : s.momentum_strength === 'weak' ? '🐌' : '⚡';
        const biasIcon = s.market_bias === 'risk_on' ? '🟢' : s.market_bias === 'risk_off' ? '🔴' : '⚪';

        html += `
            <div class="intel-section sentiment-intel">
                <h4>💭 <span class="term" data-tooltip="AI-powered analysis of market narrative and sentiment direction">Sentiment Intelligence</span></h4>
                
                <div class="sentiment-header">
                    <span class="mood-label ${sentimentClass}" data-tooltip="Overall narrative tone detected in headlines">${s.mood_label || capitalize(s.overall)}</span>
                    <span class="direction-badge" data-tooltip="Whether sentiment is getting better, worse, or staying the same">${directionIcon} ${capitalize(s.direction || 'stable')}</span>
                </div>
                
                <div class="sentiment-bar">
                    <div class="sentiment-fill ${sentimentClass}" style="width: ${scorePercent}%"></div>
                </div>
                
                <div class="market-indicators">
                    <span class="indicator term" data-tooltip="Momentum: Speed and strength of sentiment change. Strong = rapid shift, Weak = slow change">
                        ${momentumIcon} <span class="term" data-tooltip="How fast and strong sentiment is moving">Momentum</span>: <strong>${capitalize(s.momentum_strength || 'moderate')}</strong>
                    </span>
                    <span class="indicator">
                        ${biasIcon} <span class="term" data-tooltip="Market Bias: Risk-On = investors favor growth/risk assets. Risk-Off = investors favor safe havens. Balanced = mixed positioning">Bias</span>: <strong class="term" data-tooltip="${s.market_bias === 'risk_on' ? 'Investors favor risky assets like stocks & crypto' : s.market_bias === 'risk_off' ? 'Investors prefer safe assets like bonds & gold' : 'No clear preference between risk and safety'}">${s.market_bias === 'risk_on' ? 'Risk-On' : s.market_bias === 'risk_off' ? 'Risk-Off' : 'Balanced'}</strong>
                    </span>
                </div>
                
                ${s.reasoning ? `
                    <div class="reasoning">
                        <strong><span class="term" data-tooltip="AI-generated explanation of why sentiment is the way it is">Analyst View</span>:</strong> ${s.reasoning}
                    </div>
                ` : ''}
                
                <div class="signals-grid">
                    ${s.positive_signals && s.positive_signals.length > 0 ? `
                        <div class="signal-col positive">
                            <strong><span class="term" data-tooltip="Factors driving positive market sentiment">✅ Bullish Signals</span></strong>
                            <ul>${s.positive_signals.map(sig => `<li>${sig}</li>`).join('')}</ul>
                        </div>
                    ` : ''}
                    ${s.negative_signals && s.negative_signals.length > 0 ? `
                        <div class="signal-col negative">
                            <strong><span class="term" data-tooltip="Factors creating caution or negative sentiment">⚠️ Risk Signals</span></strong>
                            <ul>${s.negative_signals.map(sig => `<li>${sig}</li>`).join('')}</ul>
                        </div>
                    ` : ''}
                </div>
                
                ${s.emerging_themes && s.emerging_themes.length > 0 ? `
                    <div class="emerging-themes">
                        <strong><span class="term" data-tooltip="Hot topics and entities appearing frequently across sources">🔥 Emerging Themes</span>:</strong>
                        ${s.emerging_themes.map(t => `<span class="theme-tag">${t}</span>`).join('')}
                    </div>
                ` : ''}
                
                <div class="sentiment-meta">
                    <span class="conf-badge ${s.confidence === 'high' ? 'conf-high' : s.confidence === 'low' ? 'conf-low' : 'conf-med'}" data-tooltip="How certain the AI is about this analysis. High = consistent signals, Low = mixed/sparse data">
                        <span class="term" data-tooltip="Certainty level based on data consistency">Confidence</span>: ${capitalize(s.confidence || 'medium')}
                    </span>
                    <span class="risk-badge ${s.risk_level === 'high' ? 'risk-high' : s.risk_level === 'moderate' ? 'risk-mod' : 'risk-low'}" data-tooltip="Overall threat level detected in coverage">
                        <span class="term" data-tooltip="Level of negative/threatening news in coverage">Risk</span>: ${capitalize(s.risk_level || 'low')}
                    </span>
                </div>
            </div>
        `;
    }

    // Trends
    if (data.trends && data.trends.trending_topics) {
        html += `
            <div class="intel-section">
                <h4>📊 Trending Topics</h4>
                <div class="trend-tags">
                    ${data.trends.trending_topics.slice(0, 8).map(t =>
            `<span class="trend-tag">${t.topic} (${t.mentions})</span>`
        ).join('')}
                </div>
            </div>
        `;
    }

    // Exported file
    if (data.exported_file) {
        html += `
            <div class="intel-section">
                <h4>💾 Exported</h4>
                <p><code>${data.exported_file}</code></p>
            </div>
        `;
    }

    intelOutput.innerHTML = html || '<p>No intelligence data available.</p>';

    // Display news articles
    if (data.news && data.news.length > 0) {
        displayNews(data.news);
    }
}

function displayNews(articles) {
    const html = articles.map(article => `
        <div class="news-item">
            <a href="${article.link}" target="_blank" rel="noopener">${article.title}</a>
            <div class="news-source">${article.source} ${article.published ? '· ' + article.published : ''}</div>
        </div>
    `).join('');

    newsOutput.innerHTML = html;
}

function capitalize(str) {
    if (!str) return '';
    return str.charAt(0).toUpperCase() + str.slice(1);
}

// Initialize
async function init() {
    renderHistory(); // Load saved history
    loadSettings(); // Load saved language preferences
    initDictionary(); // Setup dictionary hover
    try {
        const resp = await fetch(`${API_BASE}/capabilities`);
        const caps = await resp.json();
        console.log('Nova Intelligence Agent loaded:', caps);
        setStatus(`Ready! Click badges to toggle features, then enter a topic.`);
        await loadLanguages(); // Load available languages
    } catch (e) {
        console.error('Failed to load capabilities:', e);
        setStatus('⚠️ Backend not connected. Start server with: uvicorn app.main:app --reload');
    }
}

init();

// ============ SETTINGS PANEL ============

const settingsBtn = document.getElementById('settingsBtn');
const settingsModal = document.getElementById('settingsModal');
const closeSettings = document.getElementById('closeSettings');
const languageSelector = document.getElementById('languageSelector');
const translateLang = document.getElementById('translateLang');

// Settings state
let selectedLanguages = JSON.parse(localStorage.getItem('novaLanguages') || '["hi", "es", "fr"]');
let dictionaryEnabled = localStorage.getItem('novaDictEnabled') !== 'false';

function loadSettings() {
    const dictToggle = document.getElementById('dictToggle');
    if (dictToggle) dictToggle.checked = dictionaryEnabled;
}

settingsBtn?.addEventListener('click', () => {
    settingsModal.classList.remove('hidden');
    loadLanguages(); // Load languages when settings opens
});

closeSettings?.addEventListener('click', () => {
    settingsModal.classList.add('hidden');
});

settingsModal?.addEventListener('click', (e) => {
    if (e.target === settingsModal) {
        settingsModal.classList.add('hidden');
    }
});

document.getElementById('dictToggle')?.addEventListener('change', (e) => {
    dictionaryEnabled = e.target.checked;
});

// Save Settings Button
document.getElementById('saveSettings')?.addEventListener('click', () => {
    // Save settings to localStorage
    localStorage.setItem('novaLanguages', JSON.stringify(selectedLanguages));
    localStorage.setItem('novaDictEnabled', dictionaryEnabled);

    // Update translate dropdown
    updateTranslateDropdown();

    // Close modal with confirmation
    settingsModal.classList.add('hidden');
    setStatus('✅ Settings saved!');
});

async function loadLanguages() {
    // Hardcoded languages as fallback
    const defaultLanguages = [
        { code: "en", name: "English" },
        { code: "hi", name: "Hindi" },
        { code: "es", name: "Spanish" },
        { code: "fr", name: "French" },
        { code: "de", name: "German" },
        { code: "zh", name: "Chinese" },
        { code: "ja", name: "Japanese" },
        { code: "ko", name: "Korean" },
        { code: "ar", name: "Arabic" },
        { code: "pt", name: "Portuguese" },
        { code: "ru", name: "Russian" },
        { code: "it", name: "Italian" },
        { code: "ta", name: "Tamil" },
        { code: "te", name: "Telugu" },
        { code: "bn", name: "Bengali" },
        { code: "mr", name: "Marathi" },
        { code: "gu", name: "Gujarati" },
        { code: "pa", name: "Punjabi" },
    ];

    let languages = defaultLanguages;

    try {
        const resp = await fetch(`${API_BASE}/languages`);
        const data = await resp.json();
        if (data.languages && data.languages.length > 0) {
            languages = data.languages;
        }
    } catch (e) {
        console.log('Using fallback languages');
    }

    if (languageSelector) {
        languageSelector.innerHTML = languages.map(lang => `
            <label class="lang-option ${selectedLanguages.includes(lang.code) ? 'selected' : ''}">
                <input type="checkbox" value="${lang.code}" 
                       ${selectedLanguages.includes(lang.code) ? 'checked' : ''}>
                ${lang.name}
            </label>
        `).join('');

        languageSelector.querySelectorAll('input[type="checkbox"]').forEach(cb => {
            cb.addEventListener('change', handleLanguageChange);
        });
    }

    updateTranslateDropdown();
}

function handleLanguageChange(e) {
    const code = e.target.value;
    const label = e.target.closest('.lang-option');

    if (e.target.checked) {
        if (selectedLanguages.length >= 3) {
            e.target.checked = false;
            alert('Maximum 3 languages allowed');
            return;
        }
        selectedLanguages.push(code);
        label.classList.add('selected');
    } else {
        selectedLanguages = selectedLanguages.filter(c => c !== code);
        label.classList.remove('selected');
    }

    localStorage.setItem('novaLanguages', JSON.stringify(selectedLanguages));
    updateTranslateDropdown();
}

function updateTranslateDropdown() {
    if (!translateLang) return;

    const langNames = {
        en: 'English', hi: 'Hindi', es: 'Spanish', fr: 'French', de: 'German',
        zh: 'Chinese', ja: 'Japanese', ko: 'Korean', ar: 'Arabic', pt: 'Portuguese',
        ru: 'Russian', it: 'Italian', ta: 'Tamil', te: 'Telugu', bn: 'Bengali',
        mr: 'Marathi', gu: 'Gujarati', pa: 'Punjabi'
    };

    translateLang.innerHTML = `
        <option value="">🌐 Translate</option>
        ${selectedLanguages.map(code => `<option value="${code}">${langNames[code] || code}</option>`).join('')}
    `;
}

// ============ TRANSLATION ============

translateLang?.addEventListener('change', async (e) => {
    const targetLang = e.target.value;
    if (!targetLang) return;

    const intelContent = document.getElementById('intelOutput');
    if (!intelContent) return;

    const originalText = intelContent.innerText;
    if (!originalText.trim()) {
        alert('No content to translate');
        return;
    }

    setStatus('🌐 Translating...', 'loading');

    try {
        const resp = await fetch(`${API_BASE}/translate`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                text: originalText.slice(0, 2000),
                from: 'en',
                to: targetLang
            })
        });

        const data = await resp.json();

        if (data.success) {
            // Store original for toggle back
            if (!intelContent.dataset.original) {
                intelContent.dataset.original = intelContent.innerHTML;
            }

            // Show translated
            intelContent.innerHTML = `
                <div class="translated-content">
                    <div class="translate-banner">
                        🌐 Translated to ${e.target.options[e.target.selectedIndex].text}
                        <button class="show-original-btn" onclick="showOriginal()">Show Original</button>
                    </div>
                    <div class="translated-text">${data.translated}</div>
                </div>
            `;
            setStatus('✅ Translated successfully');
        } else {
            setStatus('❌ Translation failed: ' + (data.error || 'Unknown error'));
        }
    } catch (err) {
        setStatus('❌ Translation error: ' + err.message);
    }

    translateLang.value = '';
});

function showOriginal() {
    const intelContent = document.getElementById('intelOutput');
    if (intelContent && intelContent.dataset.original) {
        intelContent.innerHTML = intelContent.dataset.original;
        delete intelContent.dataset.original;
        setStatus('Showing original content');
    }
}

// ============ DICTIONARY SEARCH (Toggle Mode) ============

const dictPopup = document.getElementById('dictPopup');
const dictToggleBtn = document.getElementById('dictToggleBtn');
const dictSearchBox = document.getElementById('dictSearchBox');
const dictWordInput = document.getElementById('dictWordInput');
const dictSearchBtn = document.getElementById('dictSearchBtn');

// Toggle dictionary search box
dictToggleBtn?.addEventListener('click', () => {
    dictToggleBtn.classList.toggle('active');
    dictSearchBox.classList.toggle('hidden');
    if (!dictSearchBox.classList.contains('hidden')) {
        dictWordInput.focus();
    }
});

// Search on button click
dictSearchBtn?.addEventListener('click', () => {
    const word = dictWordInput.value.trim();
    if (word) {
        lookupWord(word);
    }
});

// Search on Enter key
dictWordInput?.addEventListener('keypress', (e) => {
    if (e.key === 'Enter') {
        const word = dictWordInput.value.trim();
        if (word) {
            lookupWord(word);
        }
    }
});

async function lookupWord(word) {
    if (!dictPopup) return;

    setStatus(`📖 Looking up: ${word}...`, 'loading');

    try {
        const resp = await fetch(`${API_BASE}/dictionary/${word}`);
        const data = await resp.json();

        if (data.success) {
            dictPopup.querySelector('.dict-word').textContent = data.word;
            dictPopup.querySelector('.dict-pos').textContent = data.partOfSpeech || '';
            dictPopup.querySelector('.dict-defs').innerHTML = data.definitions
                .map(d => `<div class="dict-def">• ${d}</div>`).join('');
            dictPopup.querySelector('.dict-source').textContent = `Source: ${data.source}`;

            // Position popup near the search box
            const rect = dictSearchBox.getBoundingClientRect();
            dictPopup.style.left = rect.left + 'px';
            dictPopup.style.top = (rect.bottom + 10) + 'px';
            dictPopup.classList.remove('hidden');

            setStatus(`✅ Definition found for "${word}"`);

            // Auto-hide after 15 seconds
            setTimeout(() => {
                dictPopup.classList.add('hidden');
            }, 15000);
        } else {
            setStatus(`❌ "${word}" not found. ${data.suggestions?.length ? 'Try: ' + data.suggestions.join(', ') : ''}`);
        }
    } catch (e) {
        setStatus('❌ Dictionary error: ' + e.message);
    }

    // Clear input
    dictWordInput.value = '';
}

// Close popup on click outside
document.addEventListener('click', (e) => {
    if (dictPopup && !e.target.closest('.dict-popup') && !e.target.closest('.dict-controls')) {
        dictPopup.classList.add('hidden');
    }
});

// Make showOriginal global
window.showOriginal = showOriginal;

// ===== INTELLIGENCE PACKAGE BUILDER =====

const packageModal = document.getElementById('packageModal');
const openPackageBtn = document.getElementById('openPackageBuilder');
const closePackageBtn = document.getElementById('closePackageModal');

// Open Package Builder
openPackageBtn?.addEventListener('click', () => {
    if (!lastResultData) {
        setStatus('⚠️ No data to package. Run a search first.', 'warning');
        return;
    }
    updatePackagePreview();
    packageModal?.classList.remove('hidden');
});

// Close Package Builder
closePackageBtn?.addEventListener('click', () => {
    packageModal?.classList.add('hidden');
});

// Close on backdrop click
packageModal?.addEventListener('click', (e) => {
    if (e.target === packageModal) {
        packageModal.classList.add('hidden');
    }
});

// Update Package Preview
function updatePackagePreview() {
    if (!lastResultData) return;

    const contentsGrid = document.getElementById('packageContents');
    const qualityBadge = document.getElementById('qualityBadge');
    const articleCount = document.getElementById('articleCount');
    const sectionCount = document.getElementById('sectionCount');
    const estimatedSize = document.getElementById('estimatedSize');
    const formatRec = document.getElementById('formatRecommendation');

    // Build contents preview
    const contentItems = [
        { key: 'news', icon: '📰', label: 'News Articles', toggle: features.news },
        { key: 'summary', icon: '🧠', label: 'AI Summary', toggle: features.summary },
        { key: 'sentiment', icon: '💭', label: 'Sentiment Analysis', toggle: features.sentiment },
        { key: 'trends', icon: '📊', label: 'Trend Extraction', toggle: features.trends }
    ];

    contentsGrid.innerHTML = contentItems.map(item => {
        const hasData = lastResultData[item.key];
        const included = item.toggle && hasData;
        return `
            <div class="content-item ${included ? 'included' : 'excluded'}">
                <span class="item-check">${included ? '✔' : '✗'}</span>
                <span>${item.icon} ${item.label}</span>
            </div>
        `;
    }).join('');

    // Calculate stats
    const articles = lastResultData.news?.length || 0;
    const sections = contentItems.filter(i => i.toggle && lastResultData[i.key]).length;
    const dataSize = JSON.stringify(getFilteredData()).length;
    const sizeKB = (dataSize / 1024).toFixed(1);

    articleCount.textContent = articles;
    sectionCount.textContent = sections;
    estimatedSize.textContent = `~${sizeKB} KB`;

    // Quality badge logic
    qualityBadge.className = 'quality-badge';
    if (sections >= 3) {
        qualityBadge.classList.add('quality-full');
        qualityBadge.innerHTML = '<span class="badge-icon">🟢</span><span class="badge-text">Full Intelligence Report</span>';
    } else if (sections >= 2) {
        qualityBadge.classList.add('quality-partial');
        qualityBadge.innerHTML = '<span class="badge-icon">🟡</span><span class="badge-text">Partial Report</span>';
    } else {
        qualityBadge.classList.add('quality-raw');
        qualityBadge.innerHTML = '<span class="badge-icon">🔴</span><span class="badge-text">Raw Data Export</span>';
    }

    // Smart format recommendation
    let recFormat = 'JSON';
    let recReason = 'for API integration';

    if (features.summary || features.sentiment || features.trends) {
        recFormat = 'Markdown';
        recReason = 'for rich formatting';
    } else if (features.news && !features.summary && !features.sentiment) {
        recFormat = 'CSV';
        recReason = 'for spreadsheet analysis';
    }

    formatRec.querySelector('.rec-text').innerHTML =
        `Recommended: <strong>${recFormat}</strong> ${recReason}`;

    // Execution Quality stats
    const toolsRan = document.getElementById('toolsRanCount');
    const retriesEl = document.getElementById('retriesCount');
    const fallbacksEl = document.getElementById('fallbacksCount');
    const execConfidence = document.getElementById('execConfidence');

    if (lastExecutionMeta && toolsRan) {
        const toolCount = lastExecutionMeta.tools_executed?.length || 0;
        const totalRetries = lastExecutionMeta.tools_executed?.reduce((sum, t) => sum + (t.retries || 0), 0) || 0;
        const fallbackCount = lastExecutionMeta.fallbacks_used?.length || 0;
        const errorCount = lastExecutionMeta.errors?.length || 0;

        toolsRan.textContent = toolCount;
        retriesEl.textContent = totalRetries;
        fallbacksEl.textContent = fallbackCount;

        // Confidence badge based on execution health
        let confidence = 'high';
        let confidenceText = '🟢 High Confidence';

        if (errorCount > 0) {
            confidence = 'low';
            confidenceText = '🔴 Low Confidence';
        } else if (fallbackCount > 0 || totalRetries > 2) {
            confidence = 'medium';
            confidenceText = '🟡 Recovered';
        }

        execConfidence.innerHTML = `<span class="confidence-badge ${confidence}">${confidenceText}</span>`;
    }
}

// Get filtered data based on active toggles
function getFilteredData() {
    const filtered = {};
    if (features.news && lastResultData?.news) filtered.news = lastResultData.news;
    if (features.summary && lastResultData?.summary) filtered.summary = lastResultData.summary;
    if (features.sentiment && lastResultData?.sentiment) filtered.sentiment = lastResultData.sentiment;
    if (features.trends && lastResultData?.trends) filtered.trends = lastResultData.trends;
    return filtered;
}

// Export function
async function exportToFormat(format) {
    const filteredData = getFilteredData();

    if (Object.keys(filteredData).length === 0) {
        setStatus('⚠️ No features selected. Enable at least one toggle.', 'warning');
        return false;
    }

    try {
        const response = await fetch(`${API_BASE}/export`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                data: filteredData,
                format: format,
                filename: 'nova_intelligence_report'
            })
        });

        if (!response.ok) throw new Error('Export failed');

        const blob = await response.blob();
        const url = window.URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        const ext = format === 'markdown' ? 'md' : format;
        a.download = `nova_intelligence_report.${ext}`;
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
        window.URL.revokeObjectURL(url);

        return true;
    } catch (error) {
        console.error('Export error:', error);
        setStatus(`❌ Export failed: ${error.message}`, 'error');
        return false;
    }
}

// Individual format buttons
document.getElementById('exportJson')?.addEventListener('click', async () => {
    if (await exportToFormat('json')) {
        setStatus('✅ JSON report downloaded!', 'success');
    }
});

document.getElementById('exportMd')?.addEventListener('click', async () => {
    if (await exportToFormat('markdown')) {
        setStatus('✅ Markdown report downloaded!', 'success');
    }
});

document.getElementById('exportCsv')?.addEventListener('click', async () => {
    if (await exportToFormat('csv')) {
        setStatus('✅ CSV report downloaded!', 'success');
    }
});

document.getElementById('exportDocx')?.addEventListener('click', async () => {
    if (await exportToFormat('docx')) {
        setStatus('✅ Word document downloaded!', 'success');
    }
});

document.getElementById('exportPdf')?.addEventListener('click', async () => {
    if (await exportToFormat('pdf')) {
        setStatus('✅ PDF report downloaded!', 'success');
    }
});

// Export All Formats
document.getElementById('exportAll')?.addEventListener('click', async () => {
    setStatus('📦 Downloading all 5 formats...', 'loading');

    const formats = ['json', 'markdown', 'csv', 'docx', 'pdf'];
    let successCount = 0;

    for (const format of formats) {
        if (await exportToFormat(format)) {
            successCount++;
            await delay(300);
        }
    }

    if (successCount === formats.length) {
        setStatus(`✅ All ${successCount} formats downloaded!`, 'success');
    } else {
        setStatus(`⚠️ ${successCount}/${formats.length} formats downloaded`, 'warning');
    }
});

// Copy JSON to Clipboard
document.getElementById('copyJson')?.addEventListener('click', async () => {
    const copyBtn = document.getElementById('copyJson');
    const filteredData = getFilteredData();

    if (Object.keys(filteredData).length === 0) {
        setStatus('⚠️ No data to copy', 'warning');
        return;
    }

    try {
        const jsonStr = JSON.stringify(filteredData, null, 2);
        await navigator.clipboard.writeText(jsonStr);

        copyBtn.classList.add('copied');
        copyBtn.textContent = '✅ Copied!';

        setTimeout(() => {
            copyBtn.classList.remove('copied');
            copyBtn.textContent = '📋 Copy JSON to Clipboard';
        }, 2000);

        setStatus('✅ JSON copied to clipboard!', 'success');
    } catch (error) {
        setStatus('❌ Failed to copy', 'error');
    }
});

// Initialize dynamic panel toggles
initPanelToggles();

