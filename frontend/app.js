/**
 * Nova Intelligence Agent - Frontend JavaScript
 * Handles voice input, API calls, and result display
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
        // Auto-send after voice input
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

// Example chips
document.querySelectorAll('.chip').forEach(chip => {
    chip.addEventListener('click', () => {
        const cmd = chip.dataset.cmd;
        textInput.value = cmd;
        sendCommand(cmd);
    });
});

// Functions
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

async function sendCommand(text) {
    setStatus('⏳ Processing with Nova...', 'loading');
    sendBtn.disabled = true;
    clearResults();

    try {
        const response = await fetch(`${API_BASE}/command`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ text })
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
    // Display plan
    if (data.plan) {
        planOutput.textContent = JSON.stringify(data.plan, null, 2);
    }

    // Display intelligence data
    if (data.result && data.result.data) {
        displayIntelligence(data.result.data);
    }
}

function displayIntelligence(data) {
    let html = '';

    // Summary
    if (data.summary) {
        html += `
            <div class="intel-section">
                <h4>📝 Summary</h4>
                <p>${data.summary.summary || data.summary}</p>
            </div>
        `;
    }

    // Sentiment
    if (data.sentiment) {
        const s = data.sentiment;
        const scorePercent = Math.round((s.score || 0.5) * 100);
        const sentimentClass = s.overall || 'neutral';
        
        html += `
            <div class="intel-section">
                <h4>💭 Sentiment: ${capitalize(s.overall)} (${scorePercent}%)</h4>
                <div class="sentiment-bar">
                    <div class="sentiment-fill ${sentimentClass}" style="width: ${scorePercent}%"></div>
                </div>
                ${s.breakdown ? `
                    <small>✅ ${s.breakdown.positive || 0} positive · 
                           ⚪ ${s.breakdown.neutral || 0} neutral · 
                           ❌ ${s.breakdown.negative || 0} negative</small>
                ` : ''}
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

// Fetch capabilities on load
async function init() {
    try {
        const resp = await fetch(`${API_BASE}/capabilities`);
        const caps = await resp.json();
        console.log('Nova Intelligence Agent loaded:', caps);
        setStatus(`Ready! Say "Get AI news with sentiment analysis" to start.`);
    } catch (e) {
        console.error('Failed to load capabilities:', e);
        setStatus('⚠️ Backend not connected. Start server with: uvicorn app.main:app --reload');
    }
}

init();
