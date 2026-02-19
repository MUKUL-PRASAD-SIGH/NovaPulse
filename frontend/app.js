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
    trends: false,
    // MAS Tools (new)
    scraper: false,
    entities: false,
    images: false,
    social: false,
    research: false
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

        updateSelectAllState();
        updateStatusHint();
    });
});

// Select All button
const selectAllBtn = document.getElementById('selectAllBtn');
if (selectAllBtn) {
    selectAllBtn.addEventListener('click', () => {
        // Check if all non-news features are currently active
        const toggleableFeatures = Object.keys(features).filter(f => f !== 'news');
        const allActive = toggleableFeatures.every(f => features[f]);

        // Toggle: if all active → deselect all; otherwise → select all
        const newState = !allActive;

        toggleableFeatures.forEach(f => {
            features[f] = newState;
        });

        // Update all badge visuals
        document.querySelectorAll('.toggle-badge').forEach(badge => {
            const feature = badge.dataset.feature;
            if (feature === 'news') return;
            badge.classList.toggle('active', newState);
        });

        // Update Select All button state
        updateSelectAllState();
        updateStatusHint();
    });
}

function updateSelectAllState() {
    const btn = document.getElementById('selectAllBtn');
    if (!btn) return;
    const toggleableFeatures = Object.keys(features).filter(f => f !== 'news');
    const allActive = toggleableFeatures.every(f => features[f]);
    btn.classList.toggle('active', allActive);
    btn.textContent = allActive ? '⚡ Deselect All' : '⚡ Select All';
}


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
    if (features.scraper) extras.push('full article content');
    if (features.entities) extras.push('entity extraction');
    if (features.images) extras.push('image analysis');
    if (features.social) extras.push('social media monitoring');
    if (features.research) extras.push('research papers');

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
            // Try to parse error body for details
            try {
                const errData = await response.json();
                throw new Error(errData.detail || `HTTP ${response.status}`);
            } catch (parseErr) {
                if (parseErr.message.includes('HTTP')) throw parseErr;
                throw new Error(`HTTP ${response.status}`);
            }
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

    if (features.scraper) {
        steps.push({ id: 'web_scraper', name: 'Scraping Articles', icon: '🌐', status: 'pending' });
    }
    if (features.summary) {
        steps.push({ id: 'summarizer', name: 'Generating Summary', icon: '🧠', status: 'pending' });
    }
    if (features.sentiment) {
        steps.push({ id: 'sentiment', name: 'Sentiment Analysis', icon: '💭', status: 'pending' });
    }
    if (features.trends) {
        steps.push({ id: 'trends', name: 'Trend Extraction', icon: '📊', status: 'pending' });
    }
    if (features.entities) {
        steps.push({ id: 'entity_extractor', name: 'Extracting Entities', icon: '👤', status: 'pending' });
    }
    if (features.images) {
        steps.push({ id: 'image_analyzer', name: 'Analyzing Images', icon: '🖼️', status: 'pending' });
    }
    if (features.social) {
        steps.push({ id: 'social_monitor', name: 'Monitoring Social Media', icon: '📱', status: 'pending' });
    }
    if (features.research) {
        steps.push({ id: 'research_assistant', name: 'Searching Research', icon: '📚', status: 'pending' });
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
        if (chipIcon) chipIcon.textContent = '✅';
        chipText.textContent = `${completedCount} tools • ${elapsed}s • Success`;
        chip.classList.remove('has-errors');
    } else {
        if (chipIcon) chipIcon.textContent = '⚠️';
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
        displayMASResults(data.result.data); // Display MAS tool results
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

    // Trends V2
    if (data.trends && data.trends.trending_topics) {
        const trends = data.trends;
        html += `
            <div class="intel-section">
                <h4>📊 Trending Topics</h4>
                ${trends.rising_topics && trends.rising_topics.length > 0 ? `
                    <div class="trend-category rising">
                        <span class="category-label">🔥 Rising</span>
                        <div class="trend-tags">
                            ${trends.rising_topics.slice(0, 4).map(t =>
            `<span class="trend-tag rising-tag">${t.velocity_icon || '📈'} ${t.topic} <small title="Weighted Score = Mentions × Time Weight × Source Weight">(${t.score})</small></span>`
        ).join('')}
                        </div>
                    </div>
                ` : ''}
                <div class="trend-tags">
                    ${trends.trending_topics.slice(0, 8).map(t =>
            `<span class="trend-tag ${t.velocity === 'rising' || t.velocity === 'rising_fast' ? 'rising-tag' : t.velocity === 'fading' || t.velocity === 'fading_fast' ? 'fading-tag' : ''}">${t.velocity_icon || '➡️'} ${t.topic} <small title="Weighted Score = Mentions × Time Weight × Source Weight">(${t.score || t.mentions})</small></span>`
        ).join('')}
                </div>
                ${trends.fading_topics && trends.fading_topics.length > 0 ? `
                    <div class="trend-category fading">
                        <span class="category-label">📉 Fading</span>
                        <div class="trend-tags">
                            ${trends.fading_topics.slice(0, 3).map(t =>
            `<span class="trend-tag fading-tag">${t.velocity_icon || '↘️'} ${t.topic}</span>`
        ).join('')}
                        </div>
                    </div>
                ` : ''}
                ${trends.active_narratives && trends.active_narratives.length > 0 ? `
                    <div class="active-narratives">
                        <h5>📰 Active News Narratives</h5>
                        <div class="narrative-list">
                            ${trends.active_narratives.slice(0, 4).map(n => `
                                <div class="narrative-card ${n.story_direction?.includes('Positive') ? 'positive' : n.story_direction?.includes('Critical') ? 'critical' : 'neutral'}">
                                    <div class="narrative-header">
                                        <span class="story-icon">${n.story_icon || '📰'}</span>
                                        <span class="story-topic">${n.topic}</span>
                                        <span class="news-cycle">${n.news_cycle || 'Active'}</span>
                                    </div>
                                    <div class="narrative-details">
                                        <span title="Story Direction">📖 ${n.story_direction || 'Stable Coverage'}</span>
                                        <span title="Coverage Growth">📊 ${n.coverage || 'Steady'}</span>
                                        <span title="Tone of Coverage">🗞️ ${n.tone || 'Neutral'}</span>
                                    </div>
                                    ${n.why_trending && n.why_trending.length > 0 ? `
                                        <div class="why-trending">
                                            <small>Why trending:</small>
                                            <ul>${n.why_trending.slice(0, 2).map(r => `<li>${r}</li>`).join('')}</ul>
                                        </div>
                                    ` : ''}
                                </div>
                            `).join('')}
                        </div>
                        ${trends.news_narrative_summary ? `
                            <p class="news-summary">💡 <strong>News Summary:</strong> ${trends.news_narrative_summary}</p>
                        ` : ''}
                    </div>
                ` : ''}
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

// ===== MAS TOOLS DISPLAY FUNCTIONS =====

function displayMASResults(data) {
    // Display Entity Network
    if (data.entities) {
        displayEntities(data.entities);
    }

    // Display Image Gallery
    if (data.images) {
        displayImages(data.images);
    }

    // Display Social Media
    if (data.social) {
        displaySocial(data.social);
    }

    // Display Research
    if (data.research) {
        displayResearch(data.research);
    }
}

function displayEntities(entities) {
    const panel = document.getElementById('entityPanel');
    const output = document.getElementById('entityOutput');

    if (!panel || !output) return;

    let html = `
        <div class="mas-explanation">
            <p>🔍 <strong>Named Entity Recognition (NER)</strong> — Automatically extracts people, organizations, and locations mentioned across all fetched articles. Relationships show which entities appear together in the same sentence, suggesting a connection.</p>
        </div>
    `;

    // Entity categories
    if (entities.entities) {
        html += '<div class="entity-list">';

        // People
        if (entities.entities.people && entities.entities.people.length > 0) {
            html += `
                <div class="entity-category">
                    <h5>👤 People (${entities.entities.people.length})</h5>
                    ${entities.entities.people.slice(0, 6).map(p => `
                        <div class="entity-item">
                            <span class="entity-name">${p.name}</span>
                            <span class="entity-count">${p.mentions || 1}</span>
                        </div>
                    `).join('')}
                    ${entities.entities.people.length > 6 ? `<span style="font-size:0.65rem;color:var(--text-muted);margin-left:4px;">+${entities.entities.people.length - 6} more</span>` : ''}
                </div>
            `;
        }

        // Organizations
        if (entities.entities.organizations && entities.entities.organizations.length > 0) {
            html += `
                <div class="entity-category">
                    <h5>🏢 Organizations (${entities.entities.organizations.length})</h5>
                    ${entities.entities.organizations.slice(0, 6).map(o => `
                        <div class="entity-item">
                            <span class="entity-name">${o.name}</span>
                            <span class="entity-count">${o.mentions || 1}</span>
                        </div>
                    `).join('')}
                    ${entities.entities.organizations.length > 6 ? `<span style="font-size:0.65rem;color:var(--text-muted);margin-left:4px;">+${entities.entities.organizations.length - 6} more</span>` : ''}
                </div>
            `;
        }

        // Locations
        if (entities.entities.locations && entities.entities.locations.length > 0) {
            html += `
                <div class="entity-category">
                    <h5>📍 Locations (${entities.entities.locations.length})</h5>
                    ${entities.entities.locations.slice(0, 6).map(l => `
                        <div class="entity-item">
                            <span class="entity-name">${l.name}</span>
                            <span class="entity-count">${l.mentions || 1}</span>
                        </div>
                    `).join('')}
                    ${entities.entities.locations.length > 6 ? `<span style="font-size:0.65rem;color:var(--text-muted);margin-left:4px;">+${entities.entities.locations.length - 6} more</span>` : ''}
                </div>
            `;
        }

        html += '</div>';
    }

    // Relationships
    if (entities.relationships && entities.relationships.length > 0) {
        html += `
            <div class="relationship-graph">
                <h5>🔗 Relationships (${entities.relationships.length})</h5>
                ${entities.relationships.slice(0, 8).map(r => `
                    <div class="relationship-item">
                        <span>${r.source}</span>
                        <span class="relationship-arrow">→</span>
                        <span>${r.target}</span>
                    </div>
                `).join('')}
                ${entities.relationships.length > 8 ? `<span style="font-size:0.65rem;color:var(--text-muted);display:block;margin-top:0.3rem;">+${entities.relationships.length - 8} more connections</span>` : ''}
            </div>
        `;
    }

    output.innerHTML = html || '<p>No entities extracted.</p>';
    panel.classList.remove('hidden');
}

function displayImages(images) {
    const panel = document.getElementById('imagePanel');
    const output = document.getElementById('imageOutput');

    if (!panel || !output) return;

    const imageList = images.detailed_results || images.images || images.analyzed_images || [];

    if (imageList.length === 0) {
        const attempted = images.total_images || 0;
        const noImgMsg = attempted > 0
            ? `${attempted} image(s) were attempted but could not be fetched or analyzed.`
            : `No article images were found to analyze.`;
        output.innerHTML = `
            <div class="mas-explanation">
                <p>🔍 <strong>Image Intelligence</strong> — AI-powered image analysis: scene description, object detection, type classification, manipulation forensics, and article relevance scoring. ${noImgMsg}</p>
            </div>`;
        panel.classList.remove('hidden');
        return;
    }

    // Aggregate insights header
    const objectsDetected = images.objects_detected || [];
    const imageTypes = images.image_types || {};
    const descriptions = images.descriptions || [];

    let summaryHtml = `<div class="mas-explanation" style="margin-bottom: 14px;">
        <p>🧠 <strong>Image Intelligence</strong> — Analyzed <strong>${images.successful || imageList.length}</strong> image(s) using AI vision analysis.</p>`;

    if (objectsDetected.length > 0) {
        summaryHtml += `<p style="margin-top:6px;">🏷️ <strong>Objects detected:</strong> ${objectsDetected.slice(0, 10).map(o => `<span style="background:rgba(99,102,241,0.15);padding:2px 8px;border-radius:10px;font-size:0.85em;margin:2px;display:inline-block;">${o}</span>`).join(' ')}</p>`;
    }

    if (Object.keys(imageTypes).length > 0) {
        const typeLabels = Object.entries(imageTypes).map(([t, c]) => `${t}: ${c}`).join(' · ');
        summaryHtml += `<p style="margin-top:4px;">📊 <strong>Types:</strong> ${typeLabels}</p>`;
    }

    summaryHtml += `</div>`;

    const cardsHtml = imageList.map(img => {
        const va = img.vision_analysis || {};
        const it = img.image_type || {};
        const ms = img.manipulation_score || {};
        const meta = img.metadata || {};
        const exif = meta.exif || {};

        // AI description
        const description = va.description || '';
        const newsValue = va.news_value || '';
        const mood = va.mood || '';
        const relevance = va.relevance || '';
        const sceneType = va.scene_type || '';
        const objects = va.objects || [];
        const source = va.source === 'nova_vision' ? '🧠 Nova AI' : '📐 Local Analysis';

        // Image type badge
        const typeBadge = it.type ? `<span class="image-type-badge image-type-${it.type}">${it.type.toUpperCase()}</span>` : '';

        // Relevance badge
        const relevanceColors = { high: '#22c55e', medium: '#f59e0b', low: '#ef4444' };
        const relevanceBadge = relevance ? `<span style="background:${relevanceColors[relevance] || '#666'};color:#fff;padding:2px 8px;border-radius:10px;font-size:0.75em;font-weight:600;">Relevance: ${relevance}</span>` : '';

        // Risk badge
        const riskColors = { low: '#22c55e', medium: '#f59e0b', high: '#ef4444' };
        const riskLevel = ms.risk_level || 'unknown';
        const riskLabel = riskLevel === 'low' ? '✓ Authentic' : riskLevel === 'medium' ? '⚠ Review' : riskLevel === 'high' ? '⛔ Suspicious' : '? Unknown';

        // Manipulation flags
        const flags = ms.flags || [];
        const flagsHtml = flags.length > 0 ? `<div style="margin-top:4px;">${flags.map(f => `<span style="color:#f59e0b;font-size:0.78em;">⚠ ${f}</span>`).join('<br>')}</div>` : '';

        // Objects as tags
        const objectTags = objects.length > 0 ? `
            <div style="margin-top:6px;">
                ${objects.slice(0, 6).map(o => `<span style="background:rgba(99,102,241,0.12);color:#a5b4fc;padding:2px 7px;border-radius:8px;font-size:0.78em;margin:2px;display:inline-block;">${o}</span>`).join('')}
            </div>` : '';

        // Dominant colors
        const colors = va.dominant_colors || [];
        const colorsHtml = colors.length > 0 ? `
            <div class="image-colors" style="margin-top:6px;">
                ${colors.slice(0, 5).map(color => `
                    <div class="color-swatch" style="background-color: ${color}" title="${color}"></div>
                `).join('')}
            </div>` : '';

        // EXIF info
        const exifHtml = Object.keys(exif).length > 0 ? `
            <div style="margin-top:5px;font-size:0.78em;color:#8e95a5;">
                📷 ${exif.camera_make ? exif.camera_make + ' ' : ''}${exif.camera_model || ''}${exif.software ? ' · ' + exif.software : ''}${exif.date_time ? ' · ' + exif.date_time : ''}
            </div>` : '';

        return `
        <div class="image-card" style="display:flex;gap:16px;padding:14px;margin-bottom:12px;background:rgba(30,35,50,0.6);border-radius:12px;border:1px solid rgba(99,102,241,0.15);">
            <div style="flex-shrink:0;position:relative;">
                <img src="${img.url}" alt="Article image" class="image-preview" loading="lazy" style="width:220px;height:160px;object-fit:cover;border-radius:8px;" onerror="this.src='data:image/svg+xml,%3Csvg xmlns=%22http://www.w3.org/2000/svg%22 width=%22220%22 height=%22160%22%3E%3Crect fill=%22%232d3346%22 width=%22220%22 height=%22160%22/%3E%3Ctext x=%2250%25%22 y=%2250%25%22 fill=%22%238e95a5%22 text-anchor=%22middle%22 dy=%22.3em%22%3EImage%3C/text%3E%3C/svg%3E'">
                <div style="position:absolute;top:6px;left:6px;">${typeBadge}</div>
            </div>
            <div style="flex:1;min-width:0;">
                <div style="display:flex;align-items:center;gap:8px;flex-wrap:wrap;margin-bottom:6px;">
                    <div class="image-status ${riskLevel === 'low' ? 'safe' : 'warning'}" style="font-size:0.85em;padding:2px 10px;border-radius:10px;">
                        ${riskLabel}
                    </div>
                    ${relevanceBadge}
                    <span style="color:#64748b;font-size:0.75em;margin-left:auto;">${source}</span>
                </div>
                ${description ? `<p style="margin:4px 0;color:#e2e8f0;font-size:0.9em;line-height:1.4;">${description}</p>` : ''}
                ${newsValue && newsValue !== 'Visual context for the news story' ? `<p style="margin:2px 0;color:#94a3b8;font-size:0.82em;font-style:italic;">📰 ${newsValue}</p>` : ''}
                <div style="margin-top:4px;color:#8e95a5;font-size:0.82em;">
                    ${meta.width || '?'}×${meta.height || '?'} · ${meta.format || 'Unknown'} · ${meta.file_size_kb || '?'} KB${mood ? ` · Mood: ${mood}` : ''}${sceneType ? ` · ${sceneType}` : ''}
                </div>
                ${objectTags}
                ${colorsHtml}
                ${flagsHtml}
                ${exifHtml}
            </div>
        </div>`;
    }).join('');

    output.innerHTML = summaryHtml + cardsHtml;
    panel.classList.remove('hidden');
}

function displaySocial(social) {
    const panel = document.getElementById('socialPanel');
    const output = document.getElementById('socialOutput');

    if (!panel || !output) return;

    let html = `
        <div class="mas-explanation">
            <p>🔍 <strong>Social Media Monitoring</strong> — Scans Reddit (live) and Twitter (API required) for discussions about your topic. Shows post volume, sentiment breakdown, trending subreddits, and engagement velocity. Low buzz = few recent posts found.</p>
        </div>
    `;
    html += '<div class="social-platforms">';

    // Reddit
    if (social.platforms?.reddit) {
        const reddit = social.platforms.reddit;
        html += `
            <div class="platform-card">
                <div class="platform-header">
                    <span class="platform-name">🔴 Reddit</span>
                    <span class="platform-status">Live</span>
                </div>
                <div class="platform-stats">
                    <div class="stat-item">
                        <span class="stat-value">${reddit.post_count || 0}</span>
                        <span class="stat-label">Posts</span>
                    </div>
                    <div class="stat-item">
                        <span class="stat-value">${reddit.velocity?.posts_per_day || 0}</span>
                        <span class="stat-label">Per Day</span>
                    </div>
                </div>
                ${reddit.sentiment ? `
                    <div class="sentiment-breakdown">
                        <div class="sentiment-pill positive">${reddit.sentiment.positive ?? 0}% Positive</div>
                        <div class="sentiment-pill neutral">${reddit.sentiment.neutral ?? 0}% Neutral</div>
                        <div class="sentiment-pill negative">${reddit.sentiment.negative ?? 0}% Negative</div>
                    </div>
                ` : ''}
                ${reddit.top_posts && reddit.top_posts.length > 0 ? `
                    <div class="top-posts">
                        <h6 style="font-size: 0.75rem; margin-bottom: 0.5rem; color: var(--text-muted);">Top Posts</h6>
                        ${reddit.top_posts.slice(0, 3).map(post => `
                            <div class="post-item">
                                <div class="post-title">${(post.title || '').substring(0, 80)}${(post.title || '').length > 80 ? '...' : ''}</div>
                                <div class="post-meta">
                                    <span>r/${post.subreddit || 'unknown'}</span>
                                    <span>↑ ${post.score || 0}</span>
                                    <span>💬 ${post.num_comments || 0}</span>
                                </div>
                            </div>
                        `).join('')}
                    </div>
                ` : ''}
            </div>
        `;
    }

    // Twitter (if available)
    if (social.platforms?.twitter) {
        const twitter = social.platforms.twitter;
        html += `
            <div class="platform-card">
                <div class="platform-header">
                    <span class="platform-name">🐦 Twitter</span>
                    <span class="platform-status">${twitter.simulated ? 'Simulated' : 'Live'}</span>
                </div>
                <div class="platform-stats">
                    <div class="stat-item">
                        <span class="stat-value">${twitter.tweet_count || 0}</span>
                        <span class="stat-label">Tweets</span>
                    </div>
                    <div class="stat-item">
                        <span class="stat-value">${twitter.engagement || 0}</span>
                        <span class="stat-label">Engagement</span>
                    </div>
                </div>
            </div>
        `;
    }

    html += '</div>';

    // Aggregate insights
    if (social.aggregate) {
        html += `
            <div style="padding: 1rem; background: var(--bg-elevated); border-radius: 8px; margin-top: 1rem;">
                <h5 style="font-size: 0.85rem; margin-bottom: 0.5rem;">📊 Aggregate Insights</h5>
                <p style="font-size: 0.8rem; color: var(--text-secondary);">
                    <strong>${social.aggregate.total_mentions}</strong> total mentions • 
                    <strong>${social.aggregate.overall_sentiment}</strong> sentiment • 
                    <strong>${social.aggregate.social_buzz_level}</strong> buzz level
                </p>
            </div>
        `;
    }

    output.innerHTML = html;
    panel.classList.remove('hidden');
}

function displayResearch(research) {
    const panel = document.getElementById('researchPanel');
    const output = document.getElementById('researchOutput');

    if (!panel || !output) return;

    let html = `
        <div class="mas-explanation">
            <p>🔍 <strong>Academic & Technical Research</strong> — Searches arXiv for academic papers, GitHub for related repositories, and StackOverflow for community discussions. Provides links to PDFs, repos, and Q&A threads relevant to your topic.</p>
        </div>
    `;
    html += '<div class="research-sections">';

    // Academic Papers
    if (research.academic_papers && research.academic_papers.papers && research.academic_papers.papers.length > 0) {
        const papers = research.academic_papers.papers;
        html += `
            <div class="research-section">
                <h5>📄 Academic Papers <span class="research-count">${papers.length}</span></h5>
                <div class="research-list">
                    ${papers.slice(0, 5).map(paper => `
                        <div class="research-item">
                            <div class="research-title">${paper.title}</div>
                            <div class="research-meta">
                                <span class="research-authors">${paper.authors?.slice(0, 3).join(', ')}${paper.authors?.length > 3 ? ' et al.' : ''}</span>
                                <span>${paper.published?.substring(0, 10) || 'N/A'}</span>
                            </div>
                            <div class="research-links">
                                ${paper.pdf_url ? `<a href="${paper.pdf_url}" target="_blank" class="research-link">📥 PDF</a>` : ''}
                                ${paper.arxiv_url ? `<a href="${paper.arxiv_url}" target="_blank" class="research-link">🔗 arXiv</a>` : ''}
                            </div>
                        </div>
                    `).join('')}
                </div>
            </div>
        `;
    }

    // GitHub Repositories
    if (research.technical_docs?.github && research.technical_docs.github.repositories && research.technical_docs.github.repositories.length > 0) {
        const repos = research.technical_docs.github.repositories;
        html += `
            <div class="research-section">
                <h5>💻 GitHub Repositories <span class="research-count">${repos.length}</span></h5>
                <div class="research-list">
                    ${repos.slice(0, 5).map(repo => `
                        <div class="research-item">
                            <div class="research-title">${repo.name}</div>
                            <div class="research-meta">
                                <span>${repo.description || 'No description'}</span>
                            </div>
                            <div class="repo-stats">
                                <span class="repo-stat">⭐ <strong>${repo.stars || 0}</strong> stars</span>
                                ${repo.language ? `<span class="repo-stat">💬 <strong>${repo.language}</strong></span>` : ''}
                            </div>
                            <div class="research-links">
                                ${repo.url ? `<a href="${repo.url}" target="_blank" class="research-link">🔗 View Repo</a>` : ''}
                            </div>
                        </div>
                    `).join('')}
                </div>
            </div>
        `;
    }

    // StackOverflow
    if (research.technical_docs?.stackoverflow && research.technical_docs.stackoverflow.questions && research.technical_docs.stackoverflow.questions.length > 0) {
        const questions = research.technical_docs.stackoverflow.questions;
        html += `
            <div class="research-section">
                <h5>💡 StackOverflow Discussions <span class="research-count">${questions.length}</span></h5>
                <div class="research-list">
                    ${questions.slice(0, 5).map(q => `
                        <div class="research-item">
                            <div class="research-title">${q.title}</div>
                            <div class="research-meta">
                                <span>Score: ${q.score || 0}</span>
                                <span>${q.answer_count || 0} answers</span>
                            </div>
                            <div class="research-links">
                                ${q.link ? `<a href="${q.link}" target="_blank" class="research-link">🔗 View Question</a>` : ''}
                            </div>
                        </div>
                    `).join('')}
                </div>
            </div>
        `;
    }

    html += '</div>';

    output.innerHTML = html;
    panel.classList.remove('hidden');
}

function capitalize(str) {
    if (!str) return '';
    return str.charAt(0).toUpperCase() + str.slice(1);
}

// ============ SETTINGS STATE (must be before init) ============

// Settings state - declared early so init() -> loadSettings() can access them
let selectedLanguages = JSON.parse(localStorage.getItem('novaLanguages') || '["hi", "es", "fr"]');
let dictionaryEnabled = localStorage.getItem('novaDictEnabled') !== 'false';

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

// Add collapse button functionality for MAS panels
document.addEventListener('DOMContentLoaded', () => {
    document.querySelectorAll('.collapse-btn').forEach(btn => {
        btn.addEventListener('click', () => {
            const targetId = btn.dataset.target;
            const targetContent = document.getElementById(targetId);
            if (targetContent) {
                targetContent.classList.toggle('collapsed');
                btn.textContent = targetContent.classList.contains('collapsed') ? '▶' : '▼';
            }
        });
    });
});

init();

// ============ SETTINGS PANEL ============

const settingsBtn = document.getElementById('settingsBtn');
const settingsModal = document.getElementById('settingsModal');
const closeSettings = document.getElementById('closeSettings');
const languageSelector = document.getElementById('languageSelector');
const translateLang = document.getElementById('translateLang');

// Settings state
// selectedLanguages and dictionaryEnabled are declared above init() to avoid temporal dead zone

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

// Clear History Button
document.getElementById('clearHistoryBtn')?.addEventListener('click', () => {
    searchHistory = [];
    localStorage.removeItem('novaSearchHistory');
    renderHistory();
    setStatus('🗑️ Search history cleared!');
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

    // Build contents preview — core + MAS tools
    const contentItems = [
        { key: 'news', icon: '📰', label: 'News Articles', toggle: features.news },
        { key: 'summary', icon: '🧠', label: 'AI Summary', toggle: features.summary },
        { key: 'sentiment', icon: '💭', label: 'Sentiment Analysis', toggle: features.sentiment },
        { key: 'trends', icon: '📊', label: 'Trend Extraction', toggle: features.trends },
        { key: 'scraped_articles', icon: '🌐', label: 'Web Scraper', toggle: features.scraper },
        { key: 'entities', icon: '👤', label: 'Entity Network', toggle: features.entities },
        { key: 'images', icon: '🖼️', label: 'Image Intelligence', toggle: features.images },
        { key: 'social', icon: '📱', label: 'Social Media', toggle: features.social },
        { key: 'research', icon: '📚', label: 'Research Library', toggle: features.research }
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

    // Quality badge logic — updated thresholds for 9 tools
    qualityBadge.className = 'quality-badge';
    if (sections >= 5) {
        qualityBadge.classList.add('quality-full');
        qualityBadge.innerHTML = '<span class="badge-icon">🟢</span><span class="badge-text">Full Intelligence Report</span>';
    } else if (sections >= 3) {
        qualityBadge.classList.add('quality-partial');
        qualityBadge.innerHTML = '<span class="badge-icon">🟡</span><span class="badge-text">Partial Report</span>';
    } else {
        qualityBadge.classList.add('quality-raw');
        qualityBadge.innerHTML = '<span class="badge-icon">🔴</span><span class="badge-text">Raw Data Export</span>';
    }

    // Smart format recommendation
    let recFormat = 'JSON';
    let recReason = 'for API integration';

    if (features.entities || features.social || features.research) {
        recFormat = 'JSON';
        recReason = 'for structured intelligence data';
    } else if (features.summary || features.sentiment || features.trends) {
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
    // MAS tools
    if (features.scraper && lastResultData?.scraped_articles) filtered.scraped_articles = lastResultData.scraped_articles;
    if (features.entities && lastResultData?.entities) filtered.entities = lastResultData.entities;
    if (features.images && lastResultData?.images) filtered.images = lastResultData.images;
    if (features.social && lastResultData?.social) filtered.social = lastResultData.social;
    if (features.research && lastResultData?.research) filtered.research = lastResultData.research;
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

