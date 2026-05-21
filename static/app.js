/**
 * IIT Jammu EE GraphRAG Chatbot — Frontend JavaScript
 *
 * Handles:
 *   - Sending/receiving chat messages via API
 *   - Rendering markdown-like formatting in responses
 *   - Auto-growing textarea
 *   - Typing indicator animation
 *   - Auto-scroll behavior
 */

// ─── State ──────────────────────────────────────────────────────────────────

let isLoading = false;

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

    // Add user message to chat
    addMessage(message, 'user');

    // Clear input
    chatInput.value = '';
    chatInput.style.height = 'auto';

    // Show typing indicator
    showTypingIndicator();

    // Disable send button
    setLoading(true);

    // Send to API
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
