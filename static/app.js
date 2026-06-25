/**
 * IIT Jammu Unified AI Assistant — Frontend JavaScript
 *
 * Handles:
 *   - Sending/receiving chat messages via API (no department selection)
 *   - Auto department routing display (dept badge on responses)
 *   - Rendering markdown-like formatting in responses
 *   - Auto-growing textarea
 *   - Typing indicator animation
 *   - Auto-scroll behavior
 */

// ─── State ──────────────────────────────────────────────────────────────────

let isLoading = false;

// Department display names for badge rendering
const DEPT_DISPLAY_NAMES = {
    ee: "Electrical Engineering",
    computer_science_engineering: "Computer Science & Engineering",
    mechanical_engineering: "Mechanical Engineering",
    civil_engineering: "Civil Engineering",
    "chemical-engineering": "Chemical Engineering",
    bsbe: "Biosciences & Bioengineering",
    chemistry: "Chemistry",
    hss: "Humanities & Social Sciences",
    idp: "Interdisciplinary Programmes",
    "materials-engineering": "Materials Engineering",
    mathematics: "Mathematics",
    physics: "Physics",
};

// ─── DOM References ─────────────────────────────────────────────────────────

const chatMessages = document.getElementById('chat-messages');
const chatInput = document.getElementById('chat-input');
const sendButton = document.getElementById('send-button');
const chatContainer = document.getElementById('chat-container');

// ─── Initialization ─────────────────────────────────────────────────────────

document.addEventListener('DOMContentLoaded', () => {
    chatInput.addEventListener('keydown', handleKeyDown);
    chatInput.addEventListener('input', autoGrow);
    chatInput.focus();
    checkLlmStatus();
});

// ─── Event Handlers ─────────────────────────────────────────────────────────

function handleKeyDown(e) {
    if (e.key === 'Enter' && !e.shiftKey) {
        e.preventDefault();
        sendMessage();
    }
}

function autoGrow() {
    chatInput.style.height = 'auto';
    chatInput.style.height = Math.min(chatInput.scrollHeight, 120) + 'px';
}

// ─── Send Message ───────────────────────────────────────────────────────────

function sendMessage() {
    const message = chatInput.value.trim();
    if (!message || isLoading) return;

    addMessage(message, 'user');

    chatInput.value = '';
    chatInput.style.height = 'auto';

    showTypingIndicator();

    setLoading(true);

    fetch('/api/chat', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ message }),
    })
    .then(res => res.json())
    .then(data => {
        hideTypingIndicator();
        if (data.response) {
            addMessage(data.response, 'bot', {
                retrievalTime: data.retrieval_time,
                totalTime: data.total_time,
                routedDepartments: data.routed_departments || [],
                routingReason: data.routing_reason || '',
            });
        } else if (data.error) {
            addMessage('Sorry, I encountered an error. Please try again.', 'bot');
        }
    })
    .catch(err => {
        hideTypingIndicator();
        console.error('Chat API error:', err);
        addMessage('Unable to connect to the server. Please check if the backend is running.', 'bot');
    })
    .finally(() => {
        setLoading(false);
        chatInput.focus();
    });
}

function sendSuggestion(chipElement) {
    const text = chipElement.textContent.trim();
    chatInput.value = text;
    sendMessage();
}

// ─── Department Badge Rendering ─────────────────────────────────────────────

function renderDeptBadges(departments) {
    if (!departments || departments.length === 0) return '';
    const badges = departments.map(code => {
        const name = DEPT_DISPLAY_NAMES[code] || code;
        return `<span class="dept-badge">📍 ${name}</span>`;
    });
    return `<div class="dept-badges">${badges.join('')}</div>`;
}

// ─── Message Rendering ─────────────────────────────────────────────────────

function addMessage(text, sender, meta = {}) {
    const messageDiv = document.createElement('div');
    messageDiv.className = `message ${sender}-message`;

    const avatarSVG = sender === 'bot'
        ? `<svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
            <path d="M12 2L2 7l10 5 10-5-10-5z"/>
            <path d="M2 17l10 5 10-5"/>
            <path d="M2 12l10 5 10-5"/>
           </svg>`
        : `<svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
            <path d="M20 21v-2a4 4 0 0 0-4-4H8a4 4 0 0 0-4 4v2"/>
            <circle cx="12" cy="7" r="4"/>
           </svg>`;

    let formattedText = sender === 'bot' ? renderMarkdown(text) : escapeHtml(text);

    let badgesHtml = '';
    if (sender === 'bot' && meta.routedDepartments && meta.routedDepartments.length > 0) {
        badgesHtml = renderDeptBadges(meta.routedDepartments);
    }

    let metaHtml = '';
    if (meta.totalTime) {
        metaHtml = `<div class="message-meta">
            <span class="message-time">${getCurrentTime()}</span>
            <span class="message-stats">Answered in ${meta.totalTime}s</span>
        </div>`;
    }

    messageDiv.innerHTML = `
        <div class="message-avatar">${avatarSVG}</div>
        <div class="message-content">
            ${badgesHtml}
            <div class="message-text">${formattedText}</div>
            ${metaHtml}
        </div>
    `;

    chatMessages.appendChild(messageDiv);
    scrollToBottom();
}

// ─── Typing Indicator ──────────────────────────────────────────────────────

function showTypingIndicator() {
    const indicator = document.createElement('div');
    indicator.className = 'message bot-message';
    indicator.id = 'typing-indicator';

    indicator.innerHTML = `
        <div class="message-avatar">
            <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                <path d="M12 2L2 7l10 5 10-5-10-5z"/>
                <path d="M2 17l10 5 10-5"/>
                <path d="M2 12l10 5 10-5"/>
            </svg>
        </div>
        <div class="message-content">
            <div class="message-text">
                <div class="typing-indicator">
                    <div class="typing-dot"></div>
                    <div class="typing-dot"></div>
                    <div class="typing-dot"></div>
                </div>
            </div>
        </div>
    `;

    chatMessages.appendChild(indicator);
    scrollToBottom();
}

function hideTypingIndicator() {
    const indicator = document.getElementById('typing-indicator');
    if (indicator) indicator.remove();
}

// ─── Pre-processing: Strip HTML noise from raw text ────────────────────────

function sanitizeText(text) {
    text = decodeEntities(String(text || ''));

    text = text.replace(/<a\b([^>]*)>([\s\S]*?)<\/a>/gi, (match, attrs, label) => {
        const href = attrs.match(/href\s*=\s*["']([^"']+)["']/i);
        const cleanLabel = label.replace(/<\/?[^>\n]+>/g, '').trim();
        return href ? `[${cleanLabel || sanitizeUrl(href[1])}](${sanitizeUrl(href[1])})` : cleanLabel;
    });

    text = text.replace(/\[([^\]]+)\]\(([^)\n]+)\)/g, (match, label, url) => {
        return `[${label.trim()}](${sanitizeUrl(url)})`;
    });

    text = text.replace(/\{:\s*[^}]*\}/g, '');
    text = text.replace(/\{(?:target|rel|class|style)[^}]*\}/gi, '');
    text = text.replace(/\s*(?:target|rel|class|style)\s*=\s*["'][^"']*["']/gi, '');
    text = text.replace(/\s*(?:target|rel|class|style)\s*=\s*[^)\s>]+/gi, '');
    text = text.replace(/(https?:\/\/[^\s<>)"']+)["']?\s*>/g, '$1 ');
    text = text.replace(/<\/?[^>\n]+>/g, '');

    return text.replace(/[ \t]{2,}/g, ' ').trim();
}

// ─── Markdown Renderer ──────────────────────────────────────────────────────

function renderMarkdown(text) {
    text = sanitizeText(text).replace(/\r\n?/g, '\n');

    const blocks = [];
    let paragraph = [];
    let listType = null;
    let listItems = [];

    const flushParagraph = () => {
        if (!paragraph.length) return;
        blocks.push(`<p>${renderInline(paragraph.join(' '))}</p>`);
        paragraph = [];
    };

    const flushList = () => {
        if (!listItems.length) return;
        const tag = listType === 'ol' ? 'ol' : 'ul';
        const cls = listType === 'ol' ? 'md-list md-list-numbered' : 'md-list';
        blocks.push(`<${tag} class="${cls}">${listItems.join('')}</${tag}>`);
        listType = null;
        listItems = [];
    };

    for (const rawLine of text.split('\n')) {
        const line = rawLine.trim();

        if (!line) {
            flushParagraph();
            flushList();
            continue;
        }

        const heading = line.match(/^(#{2,4})\s+(.+)$/);
        if (heading) {
            flushParagraph();
            flushList();
            const level = heading[1].length === 2 ? 'h3' : 'h4';
            blocks.push(`<${level} class="md-heading">${renderInline(heading[2])}</${level}>`);
            continue;
        }

        if (/^---+$/.test(line)) {
            flushParagraph();
            flushList();
            blocks.push('<hr class="md-divider">');
            continue;
        }

        const ordered = line.match(/^\d+\.\s+(.+)$/);
        const bullet = line.match(/^[-*•]\s+(.+)$/);
        if (ordered || bullet) {
            flushParagraph();
            const nextType = ordered ? 'ol' : 'ul';
            if (listType && listType !== nextType) {
                flushList();
            }
            listType = nextType;
            listItems.push(`<li>${renderInline((ordered || bullet)[1])}</li>`);
            continue;
        }

        flushList();
        paragraph.push(line);
    }

    flushParagraph();
    flushList();

    return blocks.join('');
}

// ─── Utilities ──────────────────────────────────────────────────────────────

function renderInline(text) {
    const links = [];
    const tokenPrefix = '@@MD_LINK_';

    text = text.replace(/\[([^\]]+)\]\(([^)\n]+)\)/g, (match, label, url) => {
        const cleanUrl = sanitizeUrl(url);
        const token = `${tokenPrefix}${links.length}@@`;
        links.push(`<a href="${escapeAttribute(cleanUrl)}" target="_blank" rel="noopener">${escapeHtml(label)}</a>`);
        return token;
    });

    let html = escapeHtml(text);

    html = html.replace(/`([^`]+)`/g, '<code>$1</code>');
    html = html.replace(/\*\*(.+?)\*\*/g, '<strong>$1</strong>');
    html = html.replace(/\*(.+?)\*/g, '<em>$1</em>');

    html = html.replace(/https?:\/\/[^\s<]+/g, (match) => {
        const split = splitTrailingPunctuation(match);
        const cleanUrl = sanitizeUrl(split.url);
        return `<a href="${escapeAttribute(cleanUrl)}" target="_blank" rel="noopener">${escapeHtml(cleanUrl)}</a>${split.trailing}`;
    });

    html = html.replace(/([a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,})/g, (match, email) => {
        return `<a href="mailto:${escapeAttribute(email)}" class="email-link">${escapeHtml(email)}</a>`;
    });

    links.forEach((link, index) => {
        html = html.replace(`${tokenPrefix}${index}@@`, link);
    });

    return html;
}

function sanitizeUrl(url) {
    let cleanUrl = decodeEntities(String(url || '')).trim();
    cleanUrl = cleanUrl.split(/\s+(?:target|rel|class|style)\s*=/i)[0];
    return cleanUrl.replace(/^["'<]+|[>"']+$/g, '').trim();
}

function splitTrailingPunctuation(url) {
    const match = url.match(/^(.+?)([.,;:!?]+)?$/);
    return {
        url: match ? match[1] : url,
        trailing: match && match[2] ? match[2] : '',
    };
}

function decodeEntities(text) {
    const div = document.createElement('div');
    div.innerHTML = text;
    return div.textContent || '';
}

function escapeAttribute(text) {
    return escapeHtml(text).replace(/"/g, '&quot;');
}

function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}

function getCurrentTime() {
    const now = new Date();
    return now.toLocaleTimeString('en-US', { hour: '2-digit', minute: '2-digit' });
}

function scrollToBottom() {
    requestAnimationFrame(() => {
        chatContainer.scrollTop = chatContainer.scrollHeight;
    });
}

function setLoading(loading) {
    isLoading = loading;
    sendButton.disabled = loading;
}

// ─── API Key Modal State & Operations ───────────────────────────────────────

const apiKeyModal = document.getElementById('api-key-modal');
const apiKeyInput = document.getElementById('api-key-input');
const settingsButton = document.getElementById('settings-button');
const modalError = document.getElementById('modal-error');
const modalSuccess = document.getElementById('modal-success');
const btnSaveKey = document.getElementById('btn-save-key');
const btnClearKey = document.getElementById('btn-clear-key');
const successToast = document.getElementById('success-toast');

let isGeminiActive = false;

function checkLlmStatus() {
    fetch('/api/llm-status')
        .then(res => res.json())
        .then(data => {
            if (data.provider === 'gemini') {
                isGeminiActive = true;
                settingsButton.style.display = 'flex';
                updateStatusBadge(data.has_api_key);
                
                if (!data.has_api_key) {
                    const savedKey = localStorage.getItem('gemini_api_key');
                    if (savedKey) {
                        autoSubmitApiKey(savedKey);
                    } else {
                        openApiKeyModal(true);
                    }
                }
            } else {
                isGeminiActive = false;
                settingsButton.style.display = 'none';
            }
        })
        .catch(err => console.error('Error fetching LLM status:', err));
}

function updateStatusBadge(hasKey) {
    const statusText = document.querySelector('.status-text');
    const statusBadge = document.getElementById('status-badge');
    const statusDot = document.querySelector('.status-dot');
    if (isGeminiActive) {
        if (hasKey) {
            statusText.textContent = 'Gemini Active';
            statusBadge.style.background = 'rgba(59, 130, 246, 0.1)';
            statusBadge.style.borderColor = 'rgba(59, 130, 246, 0.2)';
            statusText.style.color = 'var(--accent-primary)';
            statusDot.style.background = 'var(--accent-primary)';
        } else {
            statusText.textContent = 'Gemini (No Key)';
            statusBadge.style.background = 'rgba(239, 68, 68, 0.1)';
            statusBadge.style.borderColor = 'rgba(239, 68, 68, 0.2)';
            statusText.style.color = '#ef4444';
            statusDot.style.background = '#ef4444';
        }
    } else {
        statusText.textContent = 'Online';
        statusBadge.style.background = 'rgba(16, 185, 129, 0.1)';
        statusBadge.style.borderColor = 'rgba(16, 185, 129, 0.2)';
        statusText.style.color = '#10b981';
        statusDot.style.background = '#10b981';
    }
}

function openApiKeyModal(blocking = false) {
    modalError.style.display = 'none';
    modalSuccess.style.display = 'none';
    
    const savedKey = localStorage.getItem('gemini_api_key');
    if (savedKey) {
        apiKeyInput.value = savedKey;
        btnClearKey.style.display = 'inline-flex';
    } else {
        apiKeyInput.value = '';
        btnClearKey.style.display = 'none';
    }
    
    const closeBtn = document.querySelector('.modal-close');
    if (blocking) {
        closeBtn.style.visibility = 'hidden';
    } else {
        closeBtn.style.visibility = 'visible';
    }
    
    apiKeyModal.style.display = 'flex';
    apiKeyInput.focus();
}

function closeApiKeyModal() {
    const savedKey = localStorage.getItem('gemini_api_key');
    if (isGeminiActive && !savedKey) {
        showModalError('An API Key is required to chat with the Gemini model.');
        return;
    }
    apiKeyModal.style.display = 'none';
}

function togglePasswordVisibility() {
    const eyeIcon = document.getElementById('eye-icon');
    if (apiKeyInput.type === 'password') {
        apiKeyInput.type = 'text';
        eyeIcon.innerHTML = `
            <path d="M17.94 17.94A10.07 10.07 0 0 1 12 20c-7 0-11-8-11-8a18.45 18.45 0 0 1 5.06-5.94M9.9 4.24A9.12 9.12 0 0 1 12 4c7 0 11 8 11 8a18.5 18.5 0 0 1-2.16 3.19m-6.72-1.07a3 3 0 1 1-4.24-4.24"></path>
            <line x1="1" y1="1" x2="23" y2="23"></line>
        `;
    } else {
        apiKeyInput.type = 'password';
        eyeIcon.innerHTML = `
            <path d="M1 12s4-8 11-8 11 8 11 8-4 8-11 8-11-8-11-8z"></path>
            <circle cx="12" cy="12" r="3"></circle>
        `;
    }
}

function showModalError(msg) {
    modalError.textContent = msg;
    modalError.style.display = 'block';
    modalSuccess.style.display = 'none';
}

// Ensure error is cleared when typing
apiKeyInput.addEventListener('input', () => {
    modalError.style.display = 'none';
});

function showModalSuccess(msg) {
    modalSuccess.textContent = msg;
    modalSuccess.style.display = 'block';
    modalError.style.display = 'none';
}

function saveApiKey() {
    const api_key = apiKeyInput.value.trim();
    if (!api_key) {
        showModalError('Please enter an API Key.');
        return;
    }
    
    setModalLoading(true);
    
    fetch('/api/set-gemini-key', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ api_key })
    })
    .then(res => res.json().then(data => ({ status: res.status, data })))
    .then(({ status, data }) => {
        setModalLoading(false);
        if (status === 200 && data.ok) {
            localStorage.setItem('gemini_api_key', api_key);
            showModalSuccess('API Key is valid and saved!');
            updateStatusBadge(true);
            showToast();
            setTimeout(() => {
                closeApiKeyModal();
            }, 1000);
        } else {
            showModalError(data.error || 'Failed to validate API Key.');
            updateStatusBadge(false);
        }
    })
    .catch(err => {
        setModalLoading(false);
        console.error('Error setting API key:', err);
        showModalError('Connection error while validating the key.');
    });
}

function autoSubmitApiKey(api_key) {
    fetch('/api/set-gemini-key', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ api_key })
    })
    .then(res => res.json())
    .then(data => {
        if (data.ok) {
            updateStatusBadge(true);
        } else {
            localStorage.removeItem('gemini_api_key');
            updateStatusBadge(false);
            openApiKeyModal(true);
        }
    })
    .catch(err => {
        console.error('Error auto-submitting API key:', err);
        updateStatusBadge(false);
    });
}

function clearApiKey() {
    fetch('/api/set-gemini-key', {
        method: 'DELETE'
    })
    .then(res => res.json())
    .then(data => {
        if (data.ok) {
            localStorage.removeItem('gemini_api_key');
            apiKeyInput.value = '';
            showModalSuccess('API Key cleared.');
            updateStatusBadge(false);
            btnClearKey.style.display = 'none';
            setTimeout(() => {
                openApiKeyModal(true);
            }, 1000);
        }
    })
    .catch(err => {
        console.error('Error clearing API key:', err);
        showModalError('Failed to contact server to clear key.');
    });
}

function setModalLoading(loading) {
    const btnText = btnSaveKey.querySelector('.btn-text');
    const btnSpinner = btnSaveKey.querySelector('.btn-spinner');
    
    btnSaveKey.disabled = loading;
    apiKeyInput.disabled = loading;
    
    if (loading) {
        btnText.style.opacity = '0.5';
        btnSpinner.style.display = 'inline-block';
    } else {
        btnText.style.opacity = '1';
        btnSpinner.style.display = 'none';
    }
}

function showToast() {
    successToast.style.display = 'flex';
    successToast.style.opacity = '1';
    successToast.style.transition = 'opacity 0.5s ease';
    setTimeout(() => {
        successToast.style.opacity = '0';
        setTimeout(() => {
            successToast.style.display = 'none';
        }, 500);
    }, 3000);
}

