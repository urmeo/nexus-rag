function apiHeaders(extra = {}) {
    const key = state.apiKey;
    return key ? { ...extra, 'X-API-Key': key } : { ...extra };
}

try {
    localStorage.removeItem('scinexusrag_api_key');
} catch {}

const state = {
    documents: [],
    llmModel: '',
    embeddingModel: '',
    isProcessing: false,
    apiKey: ''
};

let documentRequest = 0;
let queryRequest = 0;
let healthRequest = 0;
let authSession = 0;
let draftVersion = 0;
let uploadQueue = Promise.resolve();
let uploadHideTimer = null;
let sourceTrigger = null;

const el = Object.fromEntries([
    'modelStatus', 'apiKeyForm', 'apiKeyInput', 'apiKeySettings', 'documentsToggle',
    'sidebar', 'docCount', 'uploadZone', 'fileInput', 'uploadProgress',
    'progressFill', 'progressText', 'documentList', 'emptyDocs',
    'messages', 'welcome', 'chatInput', 'sendBtn', 'statusHint',
    'sourcesPanel', 'closeSources', 'sourcesList',
    'llmStatus', 'embeddingStatus', 'docStatus', 'chunkStatus', 'toastContainer'
].map(id => [id, document.getElementById(id)]));

document.addEventListener('DOMContentLoaded', init);

async function init() {
    setupEventListeners();
    el.apiKeyInput.value = state.apiKey;
    await Promise.all([checkHealth(), loadDocuments()]);
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
        draftVersion++;
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
            el.chatInput.value = button.textContent;
            updateInputState();
            sendMessage();
        });
    });
}

async function saveApiKey(event) {
    event.preventDefault();
    const key = el.apiKeyInput.value.trim();
    if (key !== state.apiKey) {
        authSession++;
        uploadQueue = Promise.resolve();
        queryRequest++;
        state.isProcessing = false;
        state.documents = [];
        state.llmModel = '';
        state.embeddingModel = '';
        renderDocuments();
        el.docCount.textContent = '0';
        el.docStatus.textContent = '0';
        el.chunkStatus.textContent = '0';
        updateStatusBar();
        updateInputState();
        closeSources();
        updateSourcesPanel([]);
        el.messages.querySelectorAll('.message').forEach(message => message.remove());
        el.welcome.style.display = 'flex';
        el.chatInput.value = '';
        el.chatInput.style.height = 'auto';
        clearTimeout(uploadHideTimer);
        el.uploadProgress.style.display = 'none';
    }
    state.apiKey = key;
    el.apiKeyInput.value = '';

    documentRequest++;
    el.apiKeySettings.open = false;
    await Promise.all([checkHealth(), loadDocuments()]);
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
    handleDragLeave(e);
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
    const request = ++healthRequest;
    const key = state.apiKey;
    try {
        const res = await fetch('/api/health', { headers: apiHeaders() });
        const data = await res.json();
        if (request !== healthRequest || key !== state.apiKey) return;
        state.llmModel = typeof data?.llm_model === 'string' ? data.llm_model : '';
        state.embeddingModel = typeof data?.embedding_model === 'string' ? data.embedding_model : '';
        updateStatusBar();
        if (!res.ok) {
            updateConnectionStatus({ llm_available: false });
            el.modelStatus.querySelector('.model-name').textContent = res.status === 401
                ? 'API key required' : 'Server unavailable';
            return;
        }

        updateConnectionStatus(data);
    } catch {
        if (request !== healthRequest || key !== state.apiKey) return;
        state.llmModel = '';
        state.embeddingModel = '';
        updateConnectionStatus({ llm_available: false });
        el.modelStatus.querySelector('.model-name').textContent = 'Server unavailable';
        updateStatusBar();
    }
}

function updateConnectionStatus(data) {
    const statusDot = el.modelStatus.querySelector('.status-dot');
    const modelName = el.modelStatus.querySelector('.model-name');

    if (data.llm_available === true) {
        statusDot.className = 'status-dot connected';
        modelName.textContent = state.llmModel || 'Answer model available';
    } else {
        statusDot.className = 'status-dot disconnected';
        modelName.textContent = 'Answer model unavailable';
    }
}

function updateStatusBar() {
    el.llmStatus.textContent = state.llmModel || '--';
    el.embeddingStatus.textContent = state.embeddingModel || '--';
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

function createElement(tag, className = '', text = '') {
    const node = document.createElement(tag);
    node.className = className;
    node.textContent = text;
    return node;
}

function buildDocumentItem(doc) {
    const ext = getFileExtension(doc.filename);
    const uploadTime = doc.uploaded_at ? formatTime(doc.uploaded_at) : '';

    const item = createElement('div', 'document-item');
    item.dataset.id = doc.id != null ? String(doc.id) : '';

    const icon = createElement('div', `document-icon ${ext}`, ext.toUpperCase());
    const info = createElement('div', 'document-info');
    const name = createElement('div', 'document-name', doc.filename || '');
    name.title = doc.filename || '';

    const meta = createElement('div', 'document-meta');
    if (doc.chunk_count) {
        const label = `${doc.chunk_count} passage${doc.chunk_count === 1 ? '' : 's'}`;
        meta.appendChild(createElement('span', '', label));
    }
    if (uploadTime) {
        meta.appendChild(createElement('span', '', uploadTime));
    }

    info.appendChild(name);
    info.appendChild(meta);

    const deleteButton = createElement('button', 'document-delete');
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
    const session = authSession;
    uploadQueue = uploadQueue.then(async () => {
        for (const file of batch) {
            if (session !== authSession) return;
            await uploadFile(file, session);
        }
    });
    return uploadQueue;
}

async function uploadFile(file, session) {
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
        if (session !== authSession) return;

        if (res.ok && data.success) {
            showToast(`Indexed ${file.name}`, 'success');
            await loadDocuments();
            if (session === authSession) await checkHealth();
        } else {
            showToast(apiError(data, 'Upload failed.'), 'error');
        }
    } catch {
        if (session !== authSession) return;
        showToast('Could not upload the document. Check the server connection.', 'error');
    } finally {
        if (session === authSession) hideUploadProgress();
    }
}

async function deleteDocument(docId) {
    if (!confirm('Delete this document and its indexed passages?')) return;
    const session = authSession;

    try {
        const res = await fetch(`/api/documents/${docId}`, {
            method: 'DELETE',
            headers: apiHeaders()
        });
        if (session !== authSession) return;

        if (res.ok) {
            showToast('Document deleted', 'success');
            await loadDocuments();
            if (session === authSession) await checkHealth();
        } else {
            showToast('Could not delete the document.', 'error');
        }
    } catch {
        if (session !== authSession) return;
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
        el.statusHint.textContent = `${state.documents.length} document${state.documents.length !== 1 ? 's' : ''} indexed.`;
    }
}

async function sendMessage() {
    const question = el.chatInput.value.trim();
    if (!question || state.documents.length === 0 || state.isProcessing) return;

    state.isProcessing = true;
    const request = ++queryRequest;
    const draft = draftVersion;
    let answered = false;

    el.chatInput.value = '';
    el.chatInput.style.height = 'auto';
    updateInputState();

    el.welcome.style.display = 'none';

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
            answered = true;
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
            if (!answered && draft === draftVersion && !el.chatInput.value) {
                el.chatInput.value = question;
                autoResize();
            }
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

function appendMessage(className, markup) {
    const message = createElement('div', `message ${className}`);
    message.innerHTML = markup;
    el.messages.appendChild(message);
    el.messages.scrollTop = el.messages.scrollHeight;
    return message;
}

function addUserMessage(content) {
    appendMessage('user', `
        <div class="message-bubble">
            <div class="message-content">${escapeHtml(content)}</div>
        </div>
        <div class="message-time">${formatMessageTime()}</div>
    `);
}

function addAssistantMessage(data, elapsed) {
    const sources = data.sources || [];
    const confidence = data.confidence || 0;
    const confidenceClass = confidence >= 0.6 ? 'high' : confidence >= 0.4 ? 'medium' : 'low';
    const sourceCount = sources.length;

    const formattedContent = formatResponse(data.answer, sources);

    const warnings = data.warnings || [];
    const warningsBlock = warnings.length
        ? `<div class="answer-warnings">${warnings.map((w) => `<span>⚠ ${escapeHtml(w)}</span>`).join('')}</div>`
        : '';

    const div = appendMessage('assistant', `
        <div class="message-bubble">
            <div class="message-content">${formattedContent}</div>
            ${warningsBlock}
            <div class="message-meta">
                <span class="meta-item">Answered in ${elapsed}s</span>
                <span class="meta-item confidence ${confidenceClass}" title="Based on source scores and citation coverage, or entailment checks when enabled. This score is not a probability of correctness.">Confidence score ${confidence.toFixed(2)}</span>
                ${sourceCount > 0 ? `<button type="button" class="meta-item sources-link">${sourceCount} source${sourceCount === 1 ? '' : 's'}</button>` : ''}
            </div>
        </div>
        <div class="message-time">${formatMessageTime()}</div>
        <div class="message-actions">
            <button class="action-btn copy-response" title="Copy response" aria-label="Copy response">
                <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                    <rect x="9" y="9" width="13" height="13" rx="2"></rect>
                    <path d="M5 15H4a2 2 0 01-2-2V4a2 2 0 012-2h9a2 2 0 012 2v1"></path>
                </svg>
            </button>
        </div>
    `);

    div.querySelector('.copy-response').addEventListener('click', () => copyResponse(data.answer));
    div.querySelector('.sources-link')?.addEventListener('click', () => {
        updateSourcesPanel(sources);
        openSources();
    });
    div.querySelectorAll('.citation').forEach(citation => {
        citation.addEventListener('click', () => {
            updateSourcesPanel(sources);
            highlightSource(Number(citation.dataset.sourceIndex));
        });
    });
}

function addErrorMessage(content) {
    appendMessage('assistant error', `
        <div class="message-bubble error">
            <div class="message-content">${escapeHtml(content)}</div>
        </div>
    `);
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
            const idx = sources.findIndex((source, position) =>
                (source.index ?? position + 1) === Number(num));
            if (idx >= 0) {
                const src = sources[idx];
                const tooltip = src.filename || `Source ${num}`;
                return `<button type="button" class="citation" data-source-index="${idx}" title="${escapeHtml(tooltip)}" aria-label="Open source ${num}: ${escapeHtml(tooltip)}">${num}</button>`;
            }
            return match;
        });
}

function showTypingIndicator() {
    return appendMessage('assistant typing', `
        <div class="message-bubble">
            <div class="typing-indicator">
                <span></span>
                <span></span>
                <span></span>
            </div>
        </div>
    `);
}

function formatMessageTime() {
    return new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
}

function updateSourcesPanel(sources) {
    el.sourcesList.innerHTML = sources.map((src, idx) => {
        const score = src.score || 0;
        const scoreClass = score >= 0.6 ? 'high' : score >= 0.4 ? 'medium' : 'low';
        const content = src.content || src.text || '';
        const preview = content.substring(0, 200).replace(/\n/g, ' ');
        const page = src.page ?? src.page_number;

        return `
            <div class="source-card" id="source-${idx}" role="button" tabindex="0"
                 aria-expanded="false" aria-controls="source-content-${idx}">
                <div class="source-header">
                    <span class="source-number">${escapeHtml(src.index ?? idx + 1)}</span>
                    <span class="source-filename">${escapeHtml(src.filename || 'Unknown document')}</span>
                    <span class="source-score ${scoreClass}" title="The retrieval score ranks passages for this question. It is not a probability of correctness.">Match ${score.toFixed(2)}</span>
                </div>
                ${src.section_title ? `<div class="source-section">${escapeHtml(src.section_title)}</div>` : ''}
                ${page != null ? `<div class="source-page">Page ${escapeHtml(page)}</div>` : ''}
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
    const reducedMotion = window.matchMedia('(prefers-reduced-motion: reduce)').matches;
    if (wasOpen) el.closeSources.focus();
    else if (reducedMotion) {
        requestAnimationFrame(() => {
            if (el.sourcesPanel.classList.contains('open') && document.activeElement === sourceTrigger) {
                el.closeSources.focus();
            }
        });
    }
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
    const toast = createElement('div', `toast ${type}`);

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
    if (text == null) return '';
    return String(text)
        .replace(/&/g, '&amp;')
        .replace(/</g, '&lt;')
        .replace(/>/g, '&gt;')
        .replace(/"/g, '&quot;')
        .replace(/'/g, '&#39;');
}
