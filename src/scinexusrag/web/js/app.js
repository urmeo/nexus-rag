

const API_BASE = '';



function apiHeaders(extra = {}) {
    const key = state.apiKey;
    return key ? { ...extra, 'X-API-Key': key } : { ...extra };
}

function storedApiKey() {
    try {
        return localStorage.getItem('scinexusrag_api_key') || '';
    } catch {
        return '';
    }
}


const state = {
    documents: [],
    sources: [],
    llmModel: '',
    embeddingModel: '',
    lastResponse: null,
    lastQuery: '',
    isProcessing: false,
    apiKey: storedApiKey()
};

let documentRequest = 0;
let uploadQueue = Promise.resolve();
let uploadHideTimer = null;


const el = {
    
    modelStatus: document.getElementById('modelStatus'),
    apiKeyForm: document.getElementById('apiKeyForm'),
    apiKeyInput: document.getElementById('apiKeyInput'),
    apiKeySettings: document.getElementById('apiKeySettings'),
    documentsToggle: document.getElementById('documentsToggle'),

    
    sidebar: document.getElementById('sidebar'),
    docCount: document.getElementById('docCount'),
    uploadZone: document.getElementById('uploadZone'),
    fileInput: document.getElementById('fileInput'),
    uploadProgress: document.getElementById('uploadProgress'),
    progressFill: document.getElementById('progressFill'),
    progressText: document.getElementById('progressText'),
    documentList: document.getElementById('documentList'),
    emptyDocs: document.getElementById('emptyDocs'),

    
    messages: document.getElementById('messages'),
    welcome: document.getElementById('welcome'),
    chatInput: document.getElementById('chatInput'),
    sendBtn: document.getElementById('sendBtn'),
    statusHint: document.getElementById('statusHint'),

    
    sourcesPanel: document.getElementById('sourcesPanel'),
    closeSources: document.getElementById('closeSources'),
    sourcesList: document.getElementById('sourcesList'),

    
    llmStatus: document.getElementById('llmStatus'),
    embeddingStatus: document.getElementById('embeddingStatus'),
    docStatus: document.getElementById('docStatus'),
    chunkStatus: document.getElementById('chunkStatus'),

    
    toastContainer: document.getElementById('toastContainer')
};


document.addEventListener('DOMContentLoaded', init);

async function init() {
    setupEventListeners();
    el.apiKeyInput.value = state.apiKey;
    await checkHealth();
    await loadDocuments();
}

function setupEventListeners() {
    el.apiKeyForm.addEventListener('submit', saveApiKey);
    el.documentsToggle.addEventListener('click', () => {
        const open = el.sidebar.classList.toggle('open');
        el.documentsToggle.setAttribute('aria-expanded', String(open));
    });
    
    el.uploadZone.addEventListener('click', () => el.fileInput.click());
    el.uploadZone.addEventListener('keydown', (e) => {
        if (e.key === 'Enter' || e.key === ' ') {
            e.preventDefault();
            el.fileInput.click();
        }
    });
    el.fileInput.addEventListener('change', handleFileSelect);
    el.uploadZone.addEventListener('dragover', handleDragOver);
    el.uploadZone.addEventListener('dragleave', handleDragLeave);
    el.uploadZone.addEventListener('drop', handleDrop);

    
    el.chatInput.addEventListener('input', handleInputChange);
    el.chatInput.addEventListener('keydown', handleKeyDown);
    el.sendBtn.addEventListener('click', sendMessage);

    
    el.closeSources.addEventListener('click', closeSources);

    
    el.chatInput.addEventListener('input', autoResize);
}

async function saveApiKey(event) {
    event.preventDefault();
    state.apiKey = el.apiKeyInput.value.trim();
    
    documentRequest++;
    try {
        if (state.apiKey) localStorage.setItem('scinexusrag_api_key', state.apiKey);
        else localStorage.removeItem('scinexusrag_api_key');
    } catch {
        showToast('Browser storage is unavailable; the key is saved for this session.', 'info');
    }
    el.apiKeySettings.open = false;
    await checkHealth();
    await loadDocuments();
}

function handleDragOver(e) {
    e.preventDefault();
    el.uploadZone.classList.add('dragover');
}

function handleDragLeave(e) {
    e.preventDefault();
    el.uploadZone.classList.remove('dragover');
}

function handleDrop(e) {
    e.preventDefault();
    el.uploadZone.classList.remove('dragover');
    if (e.dataTransfer.files.length) {
        uploadFiles(e.dataTransfer.files);
    }
}

function handleFileSelect(e) {
    if (e.target.files.length) {
        uploadFiles(e.target.files);
    }
    e.target.value = '';
}

function handleKeyDown(e) {
    if (e.key === 'Enter' && !e.shiftKey) {
        e.preventDefault();
        sendMessage();
    }
}

function autoResize() {
    el.chatInput.style.height = 'auto';
    el.chatInput.style.height = Math.min(el.chatInput.scrollHeight, 120) + 'px';
}


async function checkHealth() {
    const key = state.apiKey;
    try {
        const res = await fetch(`${API_BASE}/api/health`, { headers: apiHeaders() });
        const data = await res.json();
        if (key !== state.apiKey) return;
        if (!res.ok) {
            updateConnectionStatus({ llm_available: false });
            if (res.status === 401) {
                el.modelStatus.querySelector('.model-name').textContent = 'API key required';
            }
            return;
        }

        state.llmModel = data.llm_model || '';
        state.embeddingModel = data.embedding_model || '';

        updateConnectionStatus(data);
        updateStatusBar(data);
    } catch (e) {
        if (key !== state.apiKey) return;
        updateConnectionStatus({ llm_available: false });
        updateStatusBar({});
    }
}

function updateConnectionStatus(data) {
    const statusDot = el.modelStatus.querySelector('.status-dot');
    const modelName = el.modelStatus.querySelector('.model-name');

    if (data.llm_available) {
        statusDot.className = 'status-dot connected';
        modelName.textContent = state.llmModel || 'Connected';
    } else {
        statusDot.className = 'status-dot disconnected';
        modelName.textContent = 'Disconnected';
    }
}

function updateStatusBar(data) {
    el.llmStatus.textContent = state.llmModel || '--';
    el.embeddingStatus.textContent = state.embeddingModel || '--';
    el.docStatus.textContent = data.total_documents || 0;
    el.chunkStatus.textContent = data.total_chunks || 0;
}


async function loadDocuments() {
    const request = ++documentRequest;
    try {
        const documents = [];
        let data;
        do {
            const res = await fetch(`${API_BASE}/api/documents?limit=1000&offset=${documents.length}`, { headers: apiHeaders() });
            data = await res.json();
            if (request !== documentRequest) return;
            if (!res.ok) {
                throw new Error(res.status === 401 ? 'Set a valid API key to load documents.' : 'Failed to load documents.');
            }
            const page = data.documents || [];
            documents.push(...page);
            if (!page.length) break;
        } while (documents.length < data.total_documents);

        state.documents = documents;
        renderDocuments();

        el.docCount.textContent = state.documents.length;
        el.docStatus.textContent = data.total_documents || state.documents.length;
        el.chunkStatus.textContent = data.total_chunks || 0;

        updateInputState();
    } catch (e) {
        if (request !== documentRequest) return;
        console.error('Failed to load documents:', e);
        showToast(e.message || 'Failed to load documents.', 'error');
    }
}

function renderDocuments() {
    if (state.documents.length === 0) {
        el.emptyDocs.style.display = 'flex';
        
        const items = el.documentList.querySelectorAll('.document-item');
        items.forEach(item => item.remove());
        return;
    }

    el.emptyDocs.style.display = 'none';

    
    
    const existingItems = el.documentList.querySelectorAll('.document-item');
    existingItems.forEach(item => item.remove());

    const frag = document.createDocumentFragment();
    state.documents.forEach(doc => {
        frag.appendChild(buildDocumentItem(doc));
    });
    el.documentList.appendChild(frag);
}

function buildDocumentItem(doc) {
    const ext = getFileExtension(doc.filename);
    const uploadTime = doc.uploaded_at ? formatTime(doc.uploaded_at) : '';

    const item = document.createElement('div');
    item.className = 'document-item';
    item.dataset.id = doc.id != null ? String(doc.id) : '';

    const icon = document.createElement('div');
    icon.className = `document-icon ${ext}`;
    icon.textContent = ext.toUpperCase();

    const info = document.createElement('div');
    info.className = 'document-info';

    const name = document.createElement('div');
    name.className = 'document-name';
    name.title = doc.filename || '';
    name.textContent = doc.filename || '';

    const meta = document.createElement('div');
    meta.className = 'document-meta';
    if (doc.chunk_count) {
        const chunks = document.createElement('span');
        chunks.textContent = `${doc.chunk_count} chunks`;
        meta.appendChild(chunks);
    }
    if (uploadTime) {
        const when = document.createElement('span');
        when.textContent = uploadTime;
        meta.appendChild(when);
    }

    info.appendChild(name);
    info.appendChild(meta);

    const del = document.createElement('button');
    del.className = 'document-delete';
    del.title = 'Delete';
    del.innerHTML = '<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">' +
        '<path d="M3 6h18M19 6v14a2 2 0 01-2 2H7a2 2 0 01-2-2V6m3 0V4a2 2 0 012-2h4a2 2 0 012 2v2"></path></svg>';
    del.addEventListener('click', () => deleteDocument(doc.id));

    item.appendChild(icon);
    item.appendChild(info);
    item.appendChild(del);
    return item;
}

function getFileExtension(filename) {
    if (!filename) return 'txt';
    return (filename.split('.').pop() || 'txt').toLowerCase();
}

function formatTime(isoString) {
    try {
        const date = new Date(isoString);
        return date.toLocaleDateString('en-US', { month: 'short', day: 'numeric' });
    } catch {
        return '';
    }
}


function uploadFiles(files) {
    const batch = Array.from(files);
    uploadQueue = uploadQueue.then(async () => {
        for (const file of batch) await uploadFile(file);
    });
    return uploadQueue;
}

async function uploadFile(file) {
    const ext = '.' + getFileExtension(file.name);
    if (!['.pdf', '.docx', '.txt', '.md'].includes(ext)) {
        showToast(`Unsupported file type: ${ext}`, 'error');
        return;
    }

    showUploadProgress(`Processing ${file.name}...`);

    const formData = new FormData();
    formData.append('file', file);

    try {
        const res = await fetch(`${API_BASE}/api/ingest`, {
            method: 'POST',
            headers: apiHeaders(),
            body: formData
        });

        const data = await res.json();

        if (res.ok && data.success) {
            showToast(`Added: ${file.name}`, 'success');
            await loadDocuments();
            await checkHealth();
        } else {
            showToast(data.detail || data.error || 'Upload failed', 'error');
        }
    } catch (e) {
        showToast('Upload error - check server connection', 'error');
    } finally {
        hideUploadProgress();
    }
}

async function deleteDocument(docId) {
    if (!confirm('Delete this document and all its chunks?')) return;

    try {
        const res = await fetch(`${API_BASE}/api/documents/${docId}`, {
            method: 'DELETE',
            headers: apiHeaders()
        });

        if (res.ok) {
            showToast('Document deleted', 'success');
            await loadDocuments();
            await checkHealth();
        } else {
            showToast('Delete failed', 'error');
        }
    } catch (e) {
        showToast('Delete error', 'error');
    }
}

function showUploadProgress(text) {
    clearTimeout(uploadHideTimer);
    el.uploadProgress.style.display = 'block';
    el.progressText.textContent = text;
    el.progressFill.style.width = '35%';
    el.progressFill.classList.add('indeterminate');
}

function hideUploadProgress() {
    el.progressFill.classList.remove('indeterminate');
    el.progressFill.style.width = '100%';
    uploadHideTimer = setTimeout(() => {
        el.uploadProgress.style.display = 'none';
        el.progressFill.style.width = '0%';
    }, 300);
}


function handleInputChange() {
    updateInputState();
}

function updateInputState() {
    const hasText = el.chatInput.value.trim().length > 0;
    const hasDocs = state.documents.length > 0;

    el.sendBtn.disabled = !hasText || !hasDocs || state.isProcessing;

    if (!hasDocs) {
        el.statusHint.textContent = 'Upload documents to start asking questions';
    } else if (state.isProcessing) {
        el.statusHint.textContent = 'Processing your question...';
    } else {
        el.statusHint.textContent = `${state.documents.length} document${state.documents.length !== 1 ? 's' : ''} loaded - ready to answer`;
    }
}


async function sendMessage() {
    const question = el.chatInput.value.trim();
    if (!question || state.documents.length === 0 || state.isProcessing) return;

    state.lastQuery = question;
    state.isProcessing = true;

    el.chatInput.value = '';
    el.chatInput.style.height = 'auto';
    updateInputState();

    
    if (el.welcome) {
        el.welcome.style.display = 'none';
    }

    
    closeSources();

    
    addUserMessage(question);

    
    const typingEl = showTypingIndicator();
    const startTime = Date.now();

    try {
        const res = await fetch(`${API_BASE}/api/query`, {
            method: 'POST',
            headers: apiHeaders({ 'Content-Type': 'application/json' }),
            body: JSON.stringify({ question })
        });

        const data = await res.json();

        
        typingEl.remove();

        const elapsed = ((Date.now() - startTime) / 1000).toFixed(1);

        if (res.ok) {
            state.lastResponse = data;
            addAssistantMessage(data, elapsed);

            updateSourcesPanel(data.sources || []);
        } else {
            const d = data.detail;
            const msg = Array.isArray(d)
                ? d.map(e => (e && e.msg) ? e.msg : String(e)).join('; ')
                : (d || 'An error occurred processing your question.');
            addErrorMessage(msg);
        }
    } catch (e) {
        typingEl.remove();
        addErrorMessage('Could not connect to server. Please check your connection.');
    } finally {
        state.isProcessing = false;
        updateInputState();
    }
}

function addUserMessage(content) {
    const div = document.createElement('div');
    div.className = 'message user';

    const time = formatMessageTime();

    div.innerHTML = `
        <div class="message-bubble">
            <div class="message-content">${esc(content)}</div>
        </div>
        <div class="message-time">${time}</div>
    `;

    el.messages.appendChild(div);
    scrollToBottom();
}

function addAssistantMessage(data, elapsed) {
    const div = document.createElement('div');
    div.className = 'message assistant';

    const time = formatMessageTime();
    const confidence = data.confidence || 0;
    const confClass = confidence >= 0.6 ? 'high' : confidence >= 0.4 ? 'medium' : 'low';
    const confPercent = Math.round(confidence * 100);
    const sourceCount = data.sources?.length || 0;

    
    const formattedContent = formatResponse(data.answer, data.sources || []);

    
    const warnings = data.warnings || [];
    const warningsBlock = warnings.length
        ? `<div class="answer-warnings">${warnings.map((w) => `<span>⚠ ${esc(w)}</span>`).join('')}</div>`
        : '';

    div.innerHTML = `
        <div class="message-bubble">
            <div class="message-content">${formattedContent}</div>
            ${warningsBlock}
            <div class="message-meta">
                <span class="meta-item">${elapsed}s</span>
                <span class="meta-item confidence ${confClass}">${confPercent}% conf</span>
                ${sourceCount > 0 ? `<button type="button" class="meta-item sources-link">${sourceCount} sources</button>` : ''}
            </div>
        </div>
        <div class="message-time">${time}</div>
        <div class="message-actions">
            <button class="action-btn copy-response" title="Copy response" aria-label="Copy response">
                <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                    <rect x="9" y="9" width="13" height="13" rx="2"></rect>
                    <path d="M5 15H4a2 2 0 01-2-2V4a2 2 0 012-2h9a2 2 0 012 2v1"></path>
                </svg>
            </button>
        </div>
    `;

    div.querySelector('.copy-response').addEventListener('click', () => copyResponse(data.answer));
    div.querySelector('.sources-link')?.addEventListener('click', () => {
        updateSourcesPanel(data.sources || []);
        openSources();
    });
    div.querySelectorAll('.citation').forEach(citation => {
        citation.addEventListener('click', () => {
            updateSourcesPanel(data.sources || []);
            highlightSource(Number(citation.dataset.sourceIndex));
        });
    });

    el.messages.appendChild(div);
    scrollToBottom();
}

function addErrorMessage(content) {
    const div = document.createElement('div');
    div.className = 'message assistant error';

    div.innerHTML = `
        <div class="message-bubble error">
            <div class="message-content">${esc(content)}</div>
        </div>
    `;

    el.messages.appendChild(div);
    scrollToBottom();
}

function formatResponse(text, sources) {
    if (!text) return '';

    let html = esc(text);

    
    html = html.replace(/\*\*(.+?)\*\*/g, '<strong>$1</strong>');

    
    html = html.replace(/\*([^*]+)\*/g, '<em>$1</em>');

    
    html = html.replace(/^[\-•]\s+(.+)$/gm, '<li>$1</li>');

    
    html = html.replace(/^\d+\.\s+(.+)$/gm, '<li>$1</li>');

    
    html = html.replace(/(<li>.*?<\/li>\s*)+/gs, match => `<ul>${match}</ul>`);

    
    html = html.replace(/\[(\d+)\]/g, (match, num) => {
        const idx = parseInt(num) - 1;
        if (idx >= 0 && idx < sources.length) {
            const src = sources[idx];
            const tooltip = src.filename || `Source ${num}`;
            return `<button type="button" class="citation" data-source-index="${idx}" title="${esc(tooltip)}">${num}</button>`;
        }
        return match;
    });

    
    html = html.split('\n\n').map(p => p.trim() ? `<p>${p}</p>` : '').join('');
    html = html.replace(/\n/g, '<br>');

    return html;
}

function showTypingIndicator() {
    const div = document.createElement('div');
    div.className = 'message assistant typing';
    div.innerHTML = `
        <div class="message-bubble">
            <div class="typing-indicator">
                <span></span>
                <span></span>
                <span></span>
            </div>
        </div>
    `;

    el.messages.appendChild(div);
    scrollToBottom();
    return div;
}

function formatMessageTime() {
    return new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
}

function scrollToBottom() {
    el.messages.scrollTop = el.messages.scrollHeight;
}


function updateSourcesPanel(sources) {
    state.sources = sources;

    el.sourcesList.innerHTML = sources.map((src, idx) => {
        const score = src.score || 0;
        const scorePercent = Math.round(score * 100);
        const scoreClass = score >= 0.6 ? 'high' : score >= 0.4 ? 'medium' : 'low';
        const content = src.content || src.text || '';
        const preview = content.substring(0, 200).replace(/\n/g, ' ');

        return `
            <div class="source-card" id="source-${idx}" onclick="toggleSourceExpand(${idx})">
                <div class="source-header">
                    <span class="source-number">${idx + 1}</span>
                    <span class="source-filename">${esc(src.filename || 'Unknown document')}</span>
                    <span class="source-score ${scoreClass}">${scorePercent}%</span>
                </div>
                ${src.section_title ? `<div class="source-section">${esc(src.section_title)}</div>` : ''}
                ${src.page ? `<div class="source-page">Page ${src.page}</div>` : ''}
                <div class="source-preview">${esc(preview)}${content.length > 200 ? '...' : ''}</div>
                <div class="source-full" style="display: none;">
                    <div class="source-full-content">${esc(content)}</div>
                    ${src.truncated ? '<div class="source-truncated-note">Preview — first 500 characters of the source passage.</div>' : ''}
                </div>
            </div>
        `;
    }).join('');
}

function openSources() {
    el.sourcesPanel.classList.add('open');
}

function closeSources() {
    el.sourcesPanel.classList.remove('open');
}

function toggleSourceExpand(idx) {
    const card = document.getElementById(`source-${idx}`);
    if (!card) return;

    const preview = card.querySelector('.source-preview');
    const full = card.querySelector('.source-full');

    if (full.style.display === 'none') {
        preview.style.display = 'none';
        full.style.display = 'block';
        card.classList.add('expanded');
    } else {
        preview.style.display = 'block';
        full.style.display = 'none';
        card.classList.remove('expanded');
    }
}

function highlightSource(idx) {
    
    openSources();

    
    document.querySelectorAll('.source-card.highlighted').forEach(el => {
        el.classList.remove('highlighted');
    });

    
    const card = document.getElementById(`source-${idx}`);
    if (card) {
        card.classList.add('highlighted');
        card.scrollIntoView({ behavior: 'smooth', block: 'center' });

        
        setTimeout(() => card.classList.remove('highlighted'), 2000);
    }
}


function askSuggestion(question) {
    el.chatInput.value = question;
    handleInputChange();
    sendMessage();
}


function copyResponse(answer = state.lastResponse?.answer) {
    if (answer) {
        navigator.clipboard.writeText(answer)
            .then(() => showToast('Copied to clipboard', 'success'))
            .catch(() => showToast('Failed to copy', 'error'));
    }
}


function showToast(message, type = 'info') {
    const toast = document.createElement('div');
    toast.className = `toast ${type}`;

    const icon = type === 'success'
        ? '<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M22 11.08V12a10 10 0 11-5.93-9.14"/><polyline points="22 4 12 14.01 9 11.01"/></svg>'
        : type === 'error'
        ? '<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="12" cy="12" r="10"/><line x1="15" y1="9" x2="9" y2="15"/><line x1="9" y1="9" x2="15" y2="15"/></svg>'
        : '<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="12" cy="12" r="10"/><line x1="12" y1="16" x2="12" y2="12"/><line x1="12" y1="8" x2="12.01" y2="8"/></svg>';

    toast.innerHTML = `
        ${icon}
        <span class="toast-message">${esc(message)}</span>
    `;

    el.toastContainer.appendChild(toast);

    
    setTimeout(() => {
        toast.classList.add('fade-out');
        setTimeout(() => toast.remove(), 300);
    }, 4000);
}


function esc(text) {
    if (!text) return '';
    return String(text)
        .replace(/&/g, '&amp;')
        .replace(/</g, '&lt;')
        .replace(/>/g, '&gt;')
        .replace(/"/g, '&quot;')
        .replace(/'/g, '&#39;');
}


window.highlightSource = highlightSource;
window.toggleSourceExpand = toggleSourceExpand;
window.openSources = openSources;
window.closeSources = closeSources;
window.copyResponse = copyResponse;
window.askSuggestion = askSuggestion;
