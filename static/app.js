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
let DEPARTMENTS_METADATA = {};

const DEPT_WELCOME_DETAILS = {
    ee: {
        title: "Electrical Engineering",
        desc: "VLSI, Power Systems, Communications, RF & Microwave, Signal Processing, Power Electronics",
        chips: [
            "Who is the Head of Department?",
            "What are the main research areas?",
            "Tell me about antenna research",
            "Who supervises Ritujoy Biswas?"
        ]
    },
    cse: {
        title: "Computer Science & Engineering",
        desc: "Algorithms, Machine Learning, Computer Vision, Security, Cloud Computing, IoT",
        chips: [
            "Who is the Head of Department?",
            "What are the main research areas?",
            "Tell me about machine learning projects",
            "What are the placement statistics?"
        ]
    },
    civil_engineering: {
        title: "Civil Engineering",
        desc: "Structural, Geotechnical, Transportation, Water Resources, Environmental",
        chips: [
            "Who is the Head of Department?",
            "What civil labs are available?",
            "Tell me about structural research",
            "How many faculty are there?"
        ]
    },
    chemical: {
        title: "Chemical Engineering",
        desc: "Reaction Engineering, Thermodynamics, Transport Phenomena, Process Design",
        chips: [
            "Who is the Head of Department?",
            "What are the core research areas?",
            "List the faculty members",
            "What placement sectors recruit here?"
        ]
    },
    bsbe: {
        title: "Biosciences & Bioengineering",
        desc: "Computational Biology, Biomaterials, Biophysics, Bioinformatics",
        chips: [
            "Who is the Head of Department?",
            "What bio-labs are available?",
            "What projects are funded?",
            "Who is HOD of BSBE?"
        ]
    },
    chemistry: {
        title: "Chemistry",
        desc: "Organic, Inorganic, Physical, Analytical Chemistry, Catalysis",
        chips: [
            "Who is the Head of Department?",
            "What chemical research labs exist?",
            "What programmes are offered?"
        ]
    },
    hss: {
        title: "Humanities & Social Sciences",
        desc: "Economics, Literature, Sociology, Philosophy, Professional Communication",
        chips: [
            "Who is the HOD?",
            "What courses are offered by HSS?",
            "Who are the faculty members?"
        ]
    },
    idp: {
        title: "Interdisciplinary Programmes",
        desc: "Materials, Energy, Environment, Smart Systems",
        chips: [
            "What interdisciplinary areas are active?",
            "Who is the coordinator?"
        ]
    },
    materials: {
        title: "Materials Engineering",
        desc: "Metallurgy, Polymers, Nanomaterials, Computational Materials Science",
        chips: [
            "Who is the Head of Department?",
            "What research equipment is available?",
            "List the materials engineering faculty"
        ]
    },
    mechanical_engineering: {
        title: "Mechanical Engineering",
        desc: "Thermal, Design, Manufacturing, Robotics, Solid Mechanics",
        chips: [
            "Who is the Head of Department?",
            "What robotics research is happening?",
            "List the mechanical labs"
        ]
    },
    mathematics: {
        title: "Mathematics",
        desc: "Applied Mathematics, Statistics, Algebra, Differential Equations",
        chips: [
            "Who is the HOD of Mathematics?",
            "What math research fields are active?",
            "Who are the mathematics professors?"
        ]
    },
    physics: {
        title: "Physics",
        desc: "Condensed Matter Physics, High Energy Physics, Optics, Nanotechnology",
        chips: [
            "Who is the Head of Department?",
            "What experimental physics labs exist?",
            "Tell me about high energy physics research"
        ]
    }
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
    initDepartments();
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

function initDepartments() {
    const select = document.getElementById('dept-select');
    if (!select) return;

    fetch('/api/departments')
        .then(res => res.json())
        .then(depts => {
            select.innerHTML = '';
            depts.forEach(d => {
                const opt = document.createElement('option');
                opt.value = d.code;
                opt.textContent = `${d.name} ${d.ingested ? '●' : '○'}`;
                if (!d.ingested) {
                    opt.disabled = true;
                    opt.textContent += ' (Not Ingested)';
                }
                opt.selected = (d.code === 'ee');
                select.appendChild(opt);
            });
            
            depts.forEach(d => {
                DEPARTMENTS_METADATA[d.code] = d;
            });

            select.addEventListener('change', handleDeptChange);
            updateWelcomeMessage('ee');
        })
        .catch(err => console.error('Failed to load departments:', err));
}

function updateWelcomeMessage(deptCode) {
    const info = DEPT_WELCOME_DETAILS[deptCode] || {
        title: deptCode.toUpperCase(),
        desc: "Academic programmes, faculty, students, and research",
        chips: ["Who is the Head of Department?", "What are the main research areas?"]
    };

    const headerTitle = document.getElementById('header-title');
    if (headerTitle) {
        headerTitle.textContent = `${info.title} Assistant`;
    }

    const welcomeArea = document.querySelector('.welcome-message');
    if (welcomeArea) {
        welcomeArea.innerHTML = `
            <div class="message-avatar">
                <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                    <path d="M12 2L2 7l10 5 10-5-10-5z"/>
                    <path d="M2 17l10 5 10-5"/>
                    <path d="M2 12l10 5 10-5"/>
                </svg>
            </div>
            <div class="message-content">
                <div class="message-text">
                    <p>👋 Welcome! I'm the <strong>AI Assistant</strong> for the <strong>${info.title}</strong> at <strong>IIT Jammu</strong>.</p>
                    <p>I can help you with information about:</p>
                    <ul>
                        <li>👨‍🏫 <strong>Faculty</strong> — profiles, research interests, HOD</li>
                        <li>🎓 <strong>PhD Students</strong> — supervisors, research topics</li>
                        <li>🔬 <strong>Research Domains</strong> — ${info.desc}</li>
                        <li>💼 <strong>Projects & Placements</strong> — funded research, industrial records</li>
                        <li>🏛️ <strong>Labs & Facilities</strong> — departmental infrastructure</li>
                    </ul>
                    <p>Try asking something like:</p>
                </div>
                <div class="suggestion-chips">
                    ${info.chips.map(chip => `<button class="chip" onclick="sendSuggestion(this)">${chip}</button>`).join('')}
                </div>
            </div>
        `;
    }
}

function handleDeptChange() {
    const select = document.getElementById('dept-select');
    if (!select) return;
    const deptCode = select.value;

    const welcome = document.querySelector('.welcome-message');
    chatMessages.innerHTML = '';
    if (welcome) {
        chatMessages.appendChild(welcome);
    }

    updateWelcomeMessage(deptCode);

    const info = DEPT_WELCOME_DETAILS[deptCode] || { title: deptCode.toUpperCase() };
    addMessage(`*Switched context to **${info.title}** AI Assistant. Ask me anything about ${info.title} at IIT Jammu!*`, 'bot');
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

    const department = document.getElementById('dept-select')?.value || 'ee';

    fetch('/api/chat', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ message, department }),
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
