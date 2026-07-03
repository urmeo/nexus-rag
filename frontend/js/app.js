/**
 * NexusRAG Frontend Application
 * Premium UI with enhanced interactions
 */

const API_BASE = '';

// State
const state = {
    documents: [],
    sources: [],
    llmModel: '',
    embeddingModel: '',
    isConnected: false,
    lastResponse: null,
    lastQuery: '',
    isProcessing: false
};

// DOM Elements - matches new index.html structure
const el = {
    // Header
    modelStatus: document.getElementById('modelStatus'),

    // Sidebar
    sidebar: document.getElementById('sidebar'),
    docCount: document.getElementById('docCount'),
    uploadZone: document.getElementById('uploadZone'),
    fileInput: document.getElementById('fileInput'),
    uploadProgress: document.getElementById('uploadProgress'),
    progressFill: document.getElementById('progressFill'),
    progressText: document.getElementById('progressText'),
    documentList: document.getElementById('documentList'),
    emptyDocs: document.getElementById('emptyDocs'),

    // Chat
    messages: document.getElementById('messages'),
    welcome: document.getElementById('welcome'),
    chatInput: document.getElementById('chatInput'),
    sendBtn: document.getElementById('sendBtn'),
    statusHint: document.getElementById('statusHint'),

    // Sources Panel
    sourcesPanel: document.getElementById('sourcesPanel'),
    closeSources: document.getElementById('closeSources'),
    sourcesList: document.getElementById('sourcesList'),

    // Status Bar
    llmStatus: document.getElementById('llmStatus'),
    embeddingStatus: document.getElementById('embeddingStatus'),
    docStatus: document.getElementById('docStatus'),
    chunkStatus: document.getElementById('chunkStatus'),

    // Toast
    toastContainer: document.getElementById('toastContainer')
};

// Initialize
document.addEventListener('DOMContentLoaded', init);

async function init() {
    setupEventListeners();
    await checkHealth();
    await loadDocuments();
}

function setupEventListeners() {
    // Upload zone
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

    // Chat input
    el.chatInput.addEventListener('input', handleInputChange);
    el.chatInput.addEventListener('keydown', handleKeyDown);
    el.sendBtn.addEventListener('click', sendMessage);

    // Sources panel
    el.closeSources.addEventListener('click', closeSources);

    // Auto-resize textarea
    el.chatInput.addEventListener('input', autoResize);
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

// Health Check
async function checkHealth() {
    try {
        const res = await fetch(`${API_BASE}/api/health`);
        const data = await res.json();

        state.isConnected = res.ok && data.llm_available;
        state.llmModel = data.llm_model || '';
        state.embeddingModel = data.embedding_model || '';

        updateConnectionStatus(data);
        updateStatusBar(data);
    } catch (e) {
        state.isConnected = false;
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

// Documents
async function loadDocuments() {
    try {
        const res = await fetch(`${API_BASE}/api/documents`);
        const data = await res.json();

        state.documents = data.documents || [];
        renderDocuments();

        el.docCount.textContent = state.documents.length;
        el.docStatus.textContent = data.total_documents || state.documents.length;
        el.chunkStatus.textContent = data.total_chunks || 0;

        updateInputState();
    } catch (e) {
        console.error('Failed to load documents:', e);
    }
}

function renderDocuments() {
    if (state.documents.length === 0) {
        el.emptyDocs.style.display = 'flex';
        // Clear any document items but keep empty state
        const items = el.documentList.querySelectorAll('.document-item');
        items.forEach(item => item.remove());
        return;
    }

    el.emptyDocs.style.display = 'none';

    // Rebuild list via DOM construction so document ids/filenames are never
    // interpolated into markup (avoids HTML/attribute-injection sinks).
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

// File Upload
async function uploadFiles(files) {
    for (const file of files) {
        await uploadFile(file);
    }
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
            body: formData
        });

        const data = await res.json();

        if (res.ok && data.success) {
            showToast(`Added: ${file.name}`, 'success');
            await loadDocuments();
            await checkHealth();
        } else {
            showToast(data.error || 'Upload failed', 'error');
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
            method: 'DELETE'
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
    el.uploadProgress.style.display = 'block';
    el.progressText.textContent = text;
    el.progressFill.style.width = '0%';

    // Animate progress
    let progress = 0;
    const interval = setInterval(() => {
        progress += Math.random() * 15;
        if (progress > 90) {
            clearInterval(interval);
            progress = 90;
        }
        el.progressFill.style.width = progress + '%';
    }, 200);

    el.uploadProgress.dataset.interval = interval;
}

function hideUploadProgress() {
    const interval = el.uploadProgress.dataset.interval;
    if (interval) clearInterval(parseInt(interval));

    el.progressFill.style.width = '100%';
    setTimeout(() => {
        el.uploadProgress.style.display = 'none';
        el.progressFill.style.width = '0%';
    }, 300);
}

// Chat Input
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

// Chat
async function sendMessage() {
    const question = el.chatInput.value.trim();
    if (!question || state.documents.length === 0 || state.isProcessing) return;

    state.lastQuery = question;
    state.isProcessing = true;

    el.chatInput.value = '';
    el.chatInput.style.height = 'auto';
    updateInputState();

    // Hide welcome message
    if (el.welcome) {
        el.welcome.style.display = 'none';
    }

    // Close sources panel
    closeSources();

    // Add user message
    addUserMessage(question);

    // Show typing indicator
    const typingEl = showTypingIndicator();
    const startTime = Date.now();

    try {
        const res = await fetch(`${API_BASE}/api/query`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ question })
        });

        const data = await res.json();

        // Remove typing indicator
        typingEl.remove();

        const elapsed = ((Date.now() - startTime) / 1000).toFixed(1);

        if (res.ok) {
            state.lastResponse = data;
            addAssistantMessage(data, elapsed);

            if (data.sources?.length) {
                updateSourcesPanel(data.sources);
            }
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

    // Format content with markdown and citations
    const formattedContent = formatResponse(data.answer, data.sources || []);

    div.innerHTML = `
        <div class="message-bubble">
            <div class="message-content">${formattedContent}</div>
            <div class="message-meta">
                <span class="meta-item">${elapsed}s</span>
                <span class="meta-item confidence ${confClass}">${confPercent}% conf</span>
                ${sourceCount > 0 ? `<span class="meta-item sources-link" onclick="openSources()">${sourceCount} sources</span>` : ''}
            </div>
        </div>
        <div class="message-time">${time}</div>
        <div class="message-actions">
            <button class="action-btn" onclick="copyResponse()" title="Copy response">
                <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                    <rect x="9" y="9" width="13" height="13" rx="2"></rect>
                    <path d="M5 15H4a2 2 0 01-2-2V4a2 2 0 012-2h9a2 2 0 012 2v1"></path>
                </svg>
            </button>
        </div>
    `;

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

    // Format markdown bold
    html = html.replace(/\*\*(.+?)\*\*/g, '<strong>$1</strong>');

    // Format markdown italic
    html = html.replace(/\*([^*]+)\*/g, '<em>$1</em>');

    // Format bullet points
    html = html.replace(/^[\-•]\s+(.+)$/gm, '<li>$1</li>');

    // Format numbered lists (before wrapping, so these items get wrapped too)
    html = html.replace(/^\d+\.\s+(.+)$/gm, '<li>$1</li>');

    // Wrap consecutive list items in ul
    html = html.replace(/(<li>.*?<\/li>\s*)+/gs, match => `<ul>${match}</ul>`);

    // Format citations [1], [2], etc. - make them clickable
    html = html.replace(/\[(\d+)\]/g, (match, num) => {
        const idx = parseInt(num) - 1;
        if (idx >= 0 && idx < sources.length) {
            const src = sources[idx];
            const tooltip = src.filename || `Source ${num}`;
            return `<span class="citation" onclick="highlightSource(${idx})" title="${esc(tooltip)}">${num}</span>`;
        }
        return match;
    });

    // Convert line breaks to paragraphs for better spacing
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

// Sources Panel
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
    // Open sources panel
    openSources();

    // Remove previous highlights
    document.querySelectorAll('.source-card.highlighted').forEach(el => {
        el.classList.remove('highlighted');
    });

    // Highlight and scroll to source
    const card = document.getElementById(`source-${idx}`);
    if (card) {
        card.classList.add('highlighted');
        card.scrollIntoView({ behavior: 'smooth', block: 'center' });

        // Remove highlight after animation
        setTimeout(() => card.classList.remove('highlighted'), 2000);
    }
}

// Suggestion buttons
function askSuggestion(question) {
    el.chatInput.value = question;
    handleInputChange();
    sendMessage();
}

// Actions
function copyResponse() {
    if (state.lastResponse?.answer) {
        navigator.clipboard.writeText(state.lastResponse.answer)
            .then(() => showToast('Copied to clipboard', 'success'))
            .catch(() => showToast('Failed to copy', 'error'));
    }
}

// Toast notifications
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

    // Auto-remove after 4 seconds
    setTimeout(() => {
        toast.classList.add('fade-out');
        setTimeout(() => toast.remove(), 300);
    }, 4000);
}

// Utilities
function esc(text) {
    if (!text) return '';
    return String(text)
        .replace(/&/g, '&amp;')
        .replace(/</g, '&lt;')
        .replace(/>/g, '&gt;')
        .replace(/"/g, '&quot;')
        .replace(/'/g, '&#39;');
}

// Global function exports for onclick handlers
window.deleteDocument = deleteDocument;
window.highlightSource = highlightSource;
window.toggleSourceExpand = toggleSourceExpand;
window.openSources = openSources;
window.closeSources = closeSources;
window.copyResponse = copyResponse;
window.askSuggestion = askSuggestion;
