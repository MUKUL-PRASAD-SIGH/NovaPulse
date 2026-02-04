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
    export: true
};

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
    setStatus('⏳ Processing with Nova...', 'loading');
    sendBtn.disabled = true;
    clearResults();

    // Save to history
    saveToHistory(topic);

    try {
        const response = await fetch(`${API_BASE}/command`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ text: fullCommand })
        });

        if (!response.ok) {
            throw new Error(`HTTP ${response.status}`);
        }

        const data = await response.json();
        displayResults(data);
        setStatus('✅ Intelligence report ready!', 'success');

    } catch (error) {
        console.error('API error:', error);
        setStatus(`❌ Error: ${error.message}`, 'error');
    } finally {
        sendBtn.disabled = false;
    }
}

function clearResults() {
    planOutput.textContent = '';
    intelOutput.innerHTML = '';
    newsOutput.innerHTML = '';
}

function displayResults(data) {
    if (data.plan) {
        planOutput.textContent = JSON.stringify(data.plan, null, 2);
    }

    if (data.result && data.result.data) {
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

