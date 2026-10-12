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
    llmModel: '',
    embeddingModel: '',
    isProcessing: false,
    apiKey: storedApiKey()
};

let documentRequest = 0;
let queryRequest = 0;
let uploadQueue = Promise.resolve();
let uploadHideTimer = null;
let sourceTrigger = null;

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

    el.chatInput.addEventListener('input', () => {
        updateInputState();
        autoResize();
    });
    el.chatInput.addEventListener('keydown', handleKeyDown);
    el.sendBtn.addEventListener('click', sendMessage);

    el.closeSources.addEventListener('click', closeSources);
    el.sourcesPanel.addEventListener('transitionend', event => {
        if (event.target === el.sourcesPanel && event.propertyName === 'transform' &&
            el.sourcesPanel.classList.contains('open') && document.activeElement === sourceTrigger) {
            el.closeSources.focus();
        }
    });
    document.addEventListener('keydown', event => {
        if (event.key !== 'Escape') return;
        if (el.sourcesPanel.classList.contains('open')) {
            closeSources();
        } else if (el.sidebar.classList.contains('open')) {
            el.sidebar.classList.remove('open');
            el.documentsToggle.setAttribute('aria-expanded', 'false');
            el.documentsToggle.focus();
        }
    });

    document.querySelectorAll('.suggestion').forEach(button => {
        button.addEventListener('click', () => {
            el.chatInput.value = button.dataset.question;
            updateInputState();
            sendMessage();
        });
    });
}

async function saveApiKey(event) {
    event.preventDefault();
    const key = el.apiKeyInput.value.trim();
    if (key !== state.apiKey) {
        queryRequest++;
        state.isProcessing = false;
        state.documents = [];
        state.llmModel = '';
        state.embeddingModel = '';
        renderDocuments();
        el.docCount.textContent = '0';
        updateStatusBar({});
        updateInputState();
        closeSources();
        updateSourcesPanel([]);
        el.messages.querySelectorAll('.message').forEach(message => message.remove());
        el.welcome.style.display = 'flex';
        el.chatInput.value = '';
        el.chatInput.style.height = 'auto';
    }
    state.apiKey = key;

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
    if (e.key === 'Enter' && !e.shiftKey && !e.isComposing) {
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
        const res = await fetch('/api/health', { headers: apiHeaders() });
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
    } catch {
        if (key !== state.apiKey) return;
        updateConnectionStatus({ llm_available: false });
        el.modelStatus.querySelector('.model-name').textContent = 'Server unavailable';
        updateStatusBar({});
    }
}

function updateConnectionStatus(data) {
    const statusDot = el.modelStatus.querySelector('.status-dot');
    const modelName = el.modelStatus.querySelector('.model-name');

    if (data.llm_available) {
        statusDot.className = 'status-dot connected';
        modelName.textContent = state.llmModel || 'Answer model available';
    } else {
        statusDot.className = 'status-dot disconnected';
        modelName.textContent = 'Answer model unavailable';
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
            const res = await fetch(`/api/documents?limit=1000&offset=${documents.length}`, { headers: apiHeaders() });
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
    el.emptyDocs.style.display = state.documents.length ? 'none' : 'flex';
    el.documentList.querySelectorAll('.document-item').forEach(item => item.remove());

    const fragment = document.createDocumentFragment();
    state.documents.forEach(document => {
        fragment.appendChild(buildDocumentItem(document));
    });
    el.documentList.appendChild(fragment);
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
        chunks.textContent = `${doc.chunk_count} passage${doc.chunk_count === 1 ? '' : 's'}`;
        meta.appendChild(chunks);
    }
    if (uploadTime) {
        const when = document.createElement('span');
        when.textContent = uploadTime;
        meta.appendChild(when);
    }

    info.appendChild(name);
    info.appendChild(meta);

    const deleteButton = document.createElement('button');
    deleteButton.className = 'document-delete';
    deleteButton.title = 'Delete document';
    deleteButton.setAttribute('aria-label', `Delete ${doc.filename || 'document'}`);
    deleteButton.innerHTML = '<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" aria-hidden="true">' +
        '<path d="M3 6h18M19 6v14a2 2 0 01-2 2H7a2 2 0 01-2-2V6m3 0V4a2 2 0 012-2h4a2 2 0 012 2v2"></path></svg>';
    deleteButton.addEventListener('click', () => deleteDocument(doc.id));

    item.appendChild(icon);
    item.appendChild(info);
    item.appendChild(deleteButton);
    return item;
}

function getFileExtension(filename) {
    if (!filename) return 'txt';
    return (filename.split('.').pop() || 'txt').toLowerCase();
}

function formatTime(isoString) {
    const date = new Date(isoString);
    return Number.isNaN(date.getTime()) ? '' : date.toLocaleDateString(undefined, { month: 'short', day: 'numeric' });
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

    showUploadProgress(`Uploading and indexing ${file.name}...`);

    const formData = new FormData();
    formData.append('file', file);

    try {
        const res = await fetch('/api/ingest', {
            method: 'POST',
            headers: apiHeaders(),
            body: formData
        });

        const data = await res.json();

        if (res.ok && data.success) {
            showToast(`Indexed ${file.name}`, 'success');
            await loadDocuments();
            await checkHealth();
        } else {
            showToast(apiError(data, 'Upload failed.'), 'error');
        }
    } catch {
        showToast('Could not upload the document. Check the server connection.', 'error');
    } finally {
        hideUploadProgress();
    }
}

async function deleteDocument(docId) {
    if (!confirm('Delete this document and its indexed passages?')) return;

    try {
        const res = await fetch(`/api/documents/${docId}`, {
            method: 'DELETE',
            headers: apiHeaders()
        });

        if (res.ok) {
            showToast('Document deleted', 'success');
            await loadDocuments();
            await checkHealth();
        } else {
            showToast('Could not delete the document.', 'error');
        }
    } catch {
        showToast('Could not delete the document. Check the server connection.', 'error');
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

function updateInputState() {
    const hasText = el.chatInput.value.trim().length > 0;
    const hasDocs = state.documents.length > 0;

    el.sendBtn.disabled = !hasText || !hasDocs || state.isProcessing;

    if (!hasDocs) {
        el.statusHint.textContent = 'Upload documents to start asking questions';
    } else if (state.isProcessing) {
        el.statusHint.textContent = 'Searching documents and generating an answer...';
    } else {
        el.statusHint.textContent = `${state.documents.length} document${state.documents.length !== 1 ? 's' : ''} indexed. Check the cited passages when reading an answer.`;
    }
}

async function sendMessage() {
    const question = el.chatInput.value.trim();
    if (!question || state.documents.length === 0 || state.isProcessing) return;

    state.isProcessing = true;
    const request = ++queryRequest;

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
        const res = await fetch('/api/query', {
            method: 'POST',
            headers: apiHeaders({ 'Content-Type': 'application/json' }),
            body: JSON.stringify({ question })
        });

        const data = await res.json();
        if (request !== queryRequest) return;

        typingEl.remove();

        const elapsed = ((Date.now() - startTime) / 1000).toFixed(1);

        if (res.ok) {
            addAssistantMessage(data, elapsed);

            updateSourcesPanel(data.sources || []);
        } else {
            addErrorMessage(apiError(data, 'Could not answer the question.'));
        }
    } catch {
        if (request !== queryRequest) return;
        typingEl.remove();
        addErrorMessage('Could not connect to the server. Check your connection.');
    } finally {
        if (request === queryRequest) {
            state.isProcessing = false;
            updateInputState();
        }
    }
}

function apiError(data, fallback) {
    const detail = data?.detail || data?.error;
    if (Array.isArray(detail)) {
        return detail.map(error => error?.msg || String(error)).join('; ') || fallback;
    }
    return typeof detail === 'string' ? detail : fallback;
}

function addUserMessage(content) {
    const div = document.createElement('div');
    div.className = 'message user';

    const time = formatMessageTime();

    div.innerHTML = `
        <div class="message-bubble">
            <div class="message-content">${escapeHtml(content)}</div>
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
    const confidenceClass = confidence >= 0.6 ? 'high' : confidence >= 0.4 ? 'medium' : 'low';
    const sourceCount = data.sources?.length || 0;

    const formattedContent = formatResponse(data.answer, data.sources || []);

    const warnings = data.warnings || [];
    const warningsBlock = warnings.length
        ? `<div class="answer-warnings">${warnings.map((w) => `<span>⚠ ${escapeHtml(w)}</span>`).join('')}</div>`
        : '';

    div.innerHTML = `
        <div class="message-bubble">
            <div class="message-content">${formattedContent}</div>
            ${warningsBlock}
            <div class="message-meta">
                <span class="meta-item">Answered in ${elapsed}s</span>
                <span class="meta-item confidence ${confidenceClass}" title="Based on source scores and citation coverage, or entailment checks when enabled. This score is not a probability of correctness.">Confidence score ${confidence.toFixed(2)}</span>
                ${sourceCount > 0 ? `<button type="button" class="meta-item sources-link">${sourceCount} source${sourceCount === 1 ? '' : 's'}</button>` : ''}
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
            <div class="message-content">${escapeHtml(content)}</div>
        </div>
    `;

    el.messages.appendChild(div);
    scrollToBottom();
}

function formatResponse(text, sources) {
    if (!text) return '';

    const blocks = [];
    let lines = [];
    let blockType = 'p';
    function flush() {
        if (!lines.length) return;
        blocks.push(`<${blockType}>${lines.join(blockType === 'p' ? '<br>' : '')}</${blockType}>`);
        lines = [];
    }

    for (const line of text.split('\n')) {
        if (!line.trim()) {
            flush();
            continue;
        }
        const item = line.match(/^\s*(?:([-•])|(\d+)\.)\s+(.+)$/);
        const type = item ? (item[2] ? 'ol' : 'ul') : 'p';
        if (type !== blockType) flush();
        blockType = type;
        const content = formatInline(item ? item[3] : line, sources);
        lines.push(item ? `<li${item[2] ? ` value="${item[2]}"` : ''}>${content}</li>` : content);
    }
    flush();
    return blocks.join('');
}

function formatInline(text, sources) {
    return escapeHtml(text)
        .replace(/\*\*(.+?)\*\*/g, '<strong>$1</strong>')
        .replace(/\*([^*]+)\*/g, '<em>$1</em>')
        .replace(/\[(\d+)\]/g, (match, num) => {
            const idx = parseInt(num) - 1;
            if (idx >= 0 && idx < sources.length) {
                const src = sources[idx];
                const tooltip = src.filename || `Source ${num}`;
                return `<button type="button" class="citation" data-source-index="${idx}" title="${escapeHtml(tooltip)}" aria-label="Open source ${num}: ${escapeHtml(tooltip)}">${num}</button>`;
            }
            return match;
        });
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
    el.sourcesList.innerHTML = sources.map((src, idx) => {
        const score = src.score || 0;
        const scoreClass = score >= 0.6 ? 'high' : score >= 0.4 ? 'medium' : 'low';
        const content = src.content || src.text || '';
        const preview = content.substring(0, 200).replace(/\n/g, ' ');

        return `
            <div class="source-card" id="source-${idx}" role="button" tabindex="0"
                 aria-expanded="false" aria-controls="source-content-${idx}">
                <div class="source-header">
                    <span class="source-number">${idx + 1}</span>
                    <span class="source-filename">${escapeHtml(src.filename || 'Unknown document')}</span>
                    <span class="source-score ${scoreClass}" title="The retrieval score ranks passages for this question. It is not a probability of correctness.">Match ${score.toFixed(2)}</span>
                </div>
                ${src.section_title ? `<div class="source-section">${escapeHtml(src.section_title)}</div>` : ''}
                ${src.page ? `<div class="source-page">Page ${escapeHtml(src.page)}</div>` : ''}
                <div class="source-preview">${escapeHtml(preview)}${content.length > 200 ? '...' : ''}</div>
                <div class="source-full" id="source-content-${idx}">
                    <div class="source-full-content">${escapeHtml(content)}</div>
                    ${src.truncated ? '<div class="source-truncated-note">Only the first 500 characters of this passage are shown.</div>' : ''}
                </div>
            </div>
        `;
    }).join('');
    el.sourcesList.querySelectorAll('.source-card').forEach(card => {
        const toggle = () => {
            const expanded = card.classList.toggle('expanded');
            card.setAttribute('aria-expanded', String(expanded));
        };
        card.addEventListener('click', toggle);
        card.addEventListener('keydown', event => {
            if (event.key === 'Enter' || event.key === ' ') {
                event.preventDefault();
                toggle();
            }
        });
    });
}

function openSources() {
    const wasOpen = el.sourcesPanel.classList.contains('open');
    if (!wasOpen) sourceTrigger = document.activeElement;
    el.sourcesPanel.classList.add('open');
    if (wasOpen) el.closeSources.focus();
}

function closeSources() {
    const wasOpen = el.sourcesPanel.classList.contains('open');
    el.sourcesPanel.classList.remove('open');
    if (wasOpen) sourceTrigger?.focus();
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

async function copyResponse(answer) {
    if (!answer) return;
    try {
        await navigator.clipboard.writeText(answer);
        showToast('Copied to clipboard', 'success');
    } catch {
        showToast('Could not copy the answer.', 'error');
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
        <span class="toast-message">${escapeHtml(message)}</span>
    `;

    el.toastContainer.appendChild(toast);

    setTimeout(() => {
        toast.classList.add('fade-out');
        setTimeout(() => toast.remove(), 300);
    }, 4000);
}

function escapeHtml(text) {
    if (!text) return '';
    return String(text)
        .replace(/&/g, '&amp;')
        .replace(/</g, '&lt;')
        .replace(/>/g, '&gt;')
        .replace(/"/g, '&quot;')
        .replace(/'/g, '&#39;');
}
