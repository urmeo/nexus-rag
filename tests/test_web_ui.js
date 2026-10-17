const assert = require('node:assert/strict');
const { readFileSync } = require('node:fs');
const { join } = require('node:path');
const test = require('node:test');
const vm = require('node:vm');

const webDir = join(__dirname, '..', 'src', 'scinexusrag', 'web');
const html = readFileSync(join(webDir, 'index.html'), 'utf8');
const script = readFileSync(join(webDir, 'js', 'app.js'), 'utf8');

function createApp(fetch, { storageBlocked = false, clipboardApi, reducedMotion = false } = {}) {
    const clipboard = [];
    const storage = new Map();
    const timers = new Map();
    const documentListeners = new Map();
    let timerId = 0;
    const decode = value => value.replace(/&quot;/g, '"').replace(/&#39;/g, "'")
        .replace(/&lt;/g, '<').replace(/&gt;/g, '>').replace(/&amp;/g, '&');

    class Element {
        constructor(tag = 'div') {
            this.tagName = tag;
            this.children = [];
            this.attributes = {};
            this.dataset = {};
            this.style = {};
            this.value = '';
            this.listeners = new Map();
            this.classes = new Set();
            this.classList = {
                add: (...names) => names.forEach(name => this.classes.add(name)),
                remove: (...names) => names.forEach(name => this.classes.delete(name)),
                contains: name => this.classes.has(name),
                toggle: name => {
                    if (this.classes.has(name)) { this.classes.delete(name); return false; }
                    this.classes.add(name); return true;
                }
            };
        }
        set className(value) { this.classes = new Set(value.split(/\s+/).filter(Boolean)); }
        get className() { return [...this.classes].join(' '); }
        set textContent(value) { this.text = String(value); this.children = []; }
        get textContent() { return (this.text || '') + this.children.map(child => child.textContent).join(''); }
        set innerHTML(value) {
            this.html = value;
            this.text = '';
            this.children = [];
            const stack = [this];
            for (const token of value.matchAll(/<\/?([a-z][\w-]*)\b[^>]*>|([^<]+)/gi)) {
                if (token[2]) { stack.at(-1).text = (stack.at(-1).text || '') + decode(token[2]); continue; }
                if (token[0].startsWith('</')) {
                    if (stack.length > 1 && stack.at(-1).tagName === token[1].toLowerCase()) stack.pop();
                    continue;
                }
                const child = new Element(token[1].toLowerCase());
                for (const attr of token[0].matchAll(/([\w-]+)="([^"]*)"/g)) child.setAttribute(attr[1], decode(attr[2]));
                stack.at(-1).appendChild(child);
                if (!['input', 'meta', 'link', 'br', 'img', 'hr'].includes(child.tagName) && !token[0].endsWith('/>')) stack.push(child);
            }
        }
        get innerHTML() { return this.html || ''; }
        setAttribute(name, value) {
            this.attributes[name] = String(value);
            if (name === 'id') this.id = value;
            if (name === 'class') this.className = value;
            if (name.startsWith('data-')) this.dataset[name.slice(5).replace(/-([a-z])/g, (_, letter) => letter.toUpperCase())] = value;
        }
        getAttribute(name) { return this.attributes[name]; }
        appendChild(child) {
            if (child.tagName === '#fragment') { [...child.children].forEach(item => this.appendChild(item)); return child; }
            child.parent = this;
            this.children.push(child);
            return child;
        }
        remove() { if (this.parent) this.parent.children = this.parent.children.filter(child => child !== this); }
        querySelectorAll(selector) {
            const matches = element => selector.startsWith('#') ? element.id === selector.slice(1)
                : selector.startsWith('.') ? selector.slice(1).split('.').every(name => element.classes.has(name))
                : element.tagName === selector;
            return this.children.flatMap(child => [...(matches(child) ? [child] : []), ...child.querySelectorAll(selector)]);
        }
        querySelector(selector) { return this.querySelectorAll(selector)[0] || null; }
        addEventListener(name, handler) {
            this.listeners.set(name, [...(this.listeners.get(name) || []), handler]);
        }
        async dispatch(name, event = {}) {
            for (const handler of this.listeners.get(name) || []) await handler({ target: this, preventDefault() {}, ...event });
            const inline = this.getAttribute(`on${name}`);
            if (inline) await vm.runInContext(inline, context);
        }
        click() { return this.dispatch('click'); }
        focus() { context.document.activeElement = this; }
        scrollIntoView() { this.scrolled = true; }
    }

    const body = new Element('body');
    body.innerHTML = html;
    const get = id => body.querySelector(`#${id}`);
    const context = vm.createContext({
        document: {
            getElementById: get,
            querySelectorAll: selector => body.querySelectorAll(selector),
            createElement: tag => new Element(tag),
            createDocumentFragment: () => new Element('#fragment'),
            addEventListener(name, handler) {
                documentListeners.set(name, [...(documentListeners.get(name) || []), handler]);
            }
        },
        window: { matchMedia: () => ({ matches: reducedMotion }) },
        navigator: {
            clipboard: clipboardApi === undefined
                ? { writeText: async text => clipboard.push(text) }
                : clipboardApi
        },
        localStorage: {
            getItem: key => { if (storageBlocked) throw new Error('Storage blocked'); return storage.get(key); },
            setItem: (key, value) => { if (storageBlocked) throw new Error('Storage blocked'); storage.set(key, value); },
            removeItem: key => { if (storageBlocked) throw new Error('Storage blocked'); storage.delete(key); }
        },
        FormData: class { constructor() { this.files = []; } append(name, value) { this.files.push(value); } },
        setTimeout: (callback, delay) => { timers.set(++timerId, { callback, delay }); return timerId; },
        requestAnimationFrame: callback => { timers.set(++timerId, { callback, delay: 'frame' }); return timerId; },
        clearTimeout: id => timers.delete(id),
        setInterval() { throw new Error('Progress should be indeterminate, not a fabricated percentage.'); },
        clearInterval() {},
        fetch,
        confirm: () => true,
        console: { error() {} }
    });
    vm.runInContext(script, context, { filename: 'frontend/js/app.js' });
    const api = vm.runInContext('({ init, checkHealth, loadDocuments, uploadFiles, sendMessage, state })', context);
    return { api, get, clipboard, storage, timers, document: context.document,
        dispatchDocument: async (name, event) => {
            for (const handler of documentListeners.get(name) || []) await handler(event);
        },
        runTimers: delay => {
            for (const [id, timer] of [...timers]) {
                if (timer.delay === delay) { timers.delete(id); timer.callback(); }
            }
        }
    };
}

function response(data, status = 200) { return { ok: status < 400, status, json: async () => data }; }
function deferred() {
    let resolve, reject;
    const promise = new Promise((done, failed) => { resolve = done; reject = failed; });
    return { promise, resolve, reject };
}
async function settle() { for (let i = 0; i < 20; i++) await Promise.resolve(); }
function documents(count, prefix = 'paper') {
    return Array.from({ length: count }, (_, index) => ({ id: `${prefix}-${index}`, filename: `${prefix}-${index}.txt` }));
}
const health = { llm_available: true, llm_model: 'local', embedding_model: 'embedding', total_documents: 1, total_chunks: 2 };

test('stale health responses cannot replace newer model status or document counts', async () => {
    for (const outcome of ['success', 'unauthorized', 'network failure']) {
        const oldHealth = deferred();
        const newHealth = deferred();
        const responses = [oldHealth, newHealth];
        const app = createApp(async () => responses.shift().promise);

        const oldRequest = app.api.checkHealth();
        const newRequest = app.api.checkHealth();
        newHealth.resolve(response({ ...health, llm_model: 'current-model', total_documents: 0, total_chunks: 0 }));
        await newRequest;

        if (outcome === 'success') {
            oldHealth.resolve(response({ ...health, llm_model: 'old-model', total_documents: 2, total_chunks: 12 }));
        } else if (outcome === 'unauthorized') {
            oldHealth.resolve(response({ detail: 'Expired API key' }, 401));
        } else {
            oldHealth.reject(new Error('Connection lost'));
        }
        await oldRequest;

        assert.equal(app.api.state.llmModel, 'current-model', outcome);
        assert.equal(app.get('modelStatus').querySelector('.model-name').textContent, 'current-model', outcome);
        assert.equal(app.get('docStatus').textContent, '0', outcome);
        assert.equal(app.get('chunkStatus').textContent, '0', outcome);
    }
});

test('health refreshes preserve counts from the latest document list', async () => {
    for (const outcome of ['success', 'network failure']) {
        const pendingHealth = deferred();
        const latestDocuments = outcome === 'success' ? [] : documents(2);
        const totalChunks = latestDocuments.length * 6;
        const app = createApp(async url => url === '/api/health'
            ? pendingHealth.promise
            : response({ documents: latestDocuments, total_documents: latestDocuments.length, total_chunks: totalChunks }));

        const healthRefresh = app.api.checkHealth();
        await app.api.loadDocuments();
        if (outcome === 'success') {
            pendingHealth.resolve(response({ ...health, total_documents: 2, total_chunks: 12 }));
        } else {
            pendingHealth.reject(new Error('Connection lost'));
        }
        await healthRefresh;

        assert.equal(app.api.state.documents.length, latestDocuments.length, outcome);
        assert.equal(app.get('docStatus').textContent, String(latestDocuments.length), outcome);
        assert.equal(app.get('chunkStatus').textContent, String(totalChunks), outcome);
        assert.equal(app.get('modelStatus').querySelector('.model-name').textContent,
            outcome === 'success' ? 'local' : 'Server unavailable', outcome);
    }
});

test('older answer controls copy its answer and restore its own citation sources', async () => {
    const answers = [
        { answer: 'Older finding [1]', sources: [{ filename: 'old.txt', content: 'Old passage', score: 0.8 }] },
        { answer: 'Newer finding [1]', sources: [{ filename: 'new.txt', content: 'New passage', score: 0.9 }] },
        { answer: 'No matching evidence', sources: [] }
    ];
    const app = createApp(async url => url.includes('/query') ? response(answers.shift()) : response({}));
    app.api.state.documents = documents(1);
    for (const question of ['old question', 'new question', 'unrelated question']) {
        app.get('chatInput').value = question;
        await app.api.sendMessage();
    }
    const messages = app.get('messages').querySelectorAll('.assistant');
    assert.equal(messages.length, 3);
    assert.equal(app.get('sourcesList').children.length, 0, 'an answer without sources clears stale source cards');

    await messages[0].querySelector('.action-btn').click();
    await messages[0].querySelector('.sources-link').click();
    assert.deepEqual(app.clipboard, ['Older finding [1]']);
    assert.match(app.get('sourcesList').textContent, /old.txt/);
    assert.doesNotMatch(app.get('sourcesList').textContent, /new.txt/);
    await messages[1].querySelector('.citation').click();
    assert.match(app.get('sourcesList').textContent, /new.txt/);
    await messages[0].querySelector('.citation').click();
    assert.match(app.get('sourcesList').textContent, /Old passage/);
    assert(app.get('source-0').classList.contains('highlighted'));
    assert(app.get('sourcesPanel').classList.contains('open'));
    await app.get('source-0').click();
    assert(app.get('source-0').classList.contains('expanded'));
    await app.get('source-0').click();
    assert(!app.get('source-0').classList.contains('expanded'));
});

test('citations and source cards preserve sparse API source indices', async () => {
    const app = createApp(async () => response({
        answer: 'First finding [2]. Second finding [4]. Unknown [1] and [5].',
        sources: [
            { index: 2, filename: 'first.txt', content: 'First passage.' },
            { index: 4, filename: 'second.txt', content: 'Second passage.' }
        ]
    }));
    app.api.state.documents = documents(1);
    app.get('chatInput').value = 'Question';
    await app.api.sendMessage();

    const citations = app.get('messages').querySelectorAll('.citation');
    assert.deepEqual(citations.map(citation => citation.textContent), ['2', '4']);
    assert.match(app.get('messages').textContent, /Unknown \[1\] and \[5\]/);
    await citations[1].click();
    assert.deepEqual(app.get('sourcesList').querySelectorAll('.source-number')
        .map(number => number.textContent), ['2', '4']);
    assert(app.get('source-1').classList.contains('highlighted'));
    assert(!app.get('source-0').classList.contains('highlighted'));
    assert(app.get('sourcesPanel').classList.contains('open'));
});

test('a suggested question sends the question displayed on its button', async () => {
    const questions = [];
    const app = createApp(async (url, options) => {
        if (url.endsWith('/health')) return response(health);
        if (url.startsWith('/api/documents')) return response({ documents: documents(1), total_documents: 1 });
        questions.push(JSON.parse(options.body).question);
        return response({ answer: 'The study used a survey.', confidence: 0.75, sources: [] });
    });
    await app.api.init();
    const button = app.get('welcome').querySelector('.suggestion');
    await button.click();
    await settle();
    assert.deepEqual(questions, [button.textContent]);
    assert.match(app.get('messages').textContent, /The study used a survey/);
});

test('Enter does not submit a question while text composition is active', async () => {
    const questions = [];
    const app = createApp(async (url, options) => {
        if (url.endsWith('/health')) return response(health);
        if (url.startsWith('/api/documents')) return response({ documents: documents(1), total_documents: 1 });
        questions.push(JSON.parse(options.body).question);
        return response({ answer: 'Answer', sources: [] });
    });
    await app.api.init();
    app.get('chatInput').value = 'Question';
    await app.get('chatInput').dispatch('keydown', { key: 'Enter', isComposing: true });
    assert.deepEqual(questions, []);
    assert.equal(app.get('chatInput').value, 'Question');

    await app.get('chatInput').dispatch('keydown', { key: 'Enter', isComposing: false });
    await settle();
    assert.deepEqual(questions, ['Question']);
});

for (const [scenario, clipboardApi] of [
    ['unavailable', null],
    ['denied', { writeText: async () => { throw new Error('Permission denied'); } }]
]) {
    test(`copying reports an error when the clipboard is ${scenario}`, async () => {
        const app = createApp(async () => response({ answer: 'Answer', sources: [] }), { clipboardApi });
        app.api.state.documents = documents(1);
        app.get('chatInput').value = 'Question';
        await app.api.sendMessage();

        await app.get('messages').querySelector('.copy-response').click();
        assert.match(app.get('toastContainer').textContent, /Could not copy the answer/);
        assert.deepEqual(app.clipboard, []);
    });
}

test('documents beyond the first API page remain visible and deletable', async () => {
    const calls = [];
    const app = createApp(async (url, options) => {
        calls.push({ url, options });
        if (options.method === 'DELETE') return response({});
        if (url.endsWith('/health')) return response(health);
        const offset = Number(new URL(url, 'http://localhost').searchParams.get('offset'));
        return response({ documents: documents(offset ? 1 : 1000, offset ? 'last' : 'first'), total_documents: 1001, total_chunks: 2002 });
    });
    await app.api.loadDocuments();
    assert.equal(app.get('documentList').querySelectorAll('.document-item').length, 1001);
    assert.equal(app.get('docCount').textContent, '1001');
    assert.equal(app.get('chunkStatus').textContent, '2002');
    assert(calls.some(call => call.url.endsWith('offset=1000')));
    await app.get('documentList').querySelectorAll('.document-item').at(-1).querySelector('.document-delete').click();
    assert(calls.some(call => call.options.method === 'DELETE' && call.url.endsWith('/last-0')));
});

test('a slow old document refresh cannot replace a newer corpus', async () => {
    const old = deferred();
    let count = 0;
    const app = createApp(async () => ++count === 1 ? old.promise : response({ documents: documents(1, 'new'), total_documents: 1 }));
    const pending = app.api.loadDocuments();
    await app.api.loadDocuments();
    old.resolve(response({ documents: documents(1, 'old'), total_documents: 1 }));
    await pending;
    assert.match(app.get('documentList').textContent, /new-0.txt/);
    assert.doesNotMatch(app.get('documentList').textContent, /old-0.txt/);
});

test('failed document refresh preserves the loaded corpus and explains authentication', async () => {
    let fail = false;
    const app = createApp(async () => fail ? response({ detail: 'Unauthorized' }, 401)
        : response({ documents: documents(1), total_documents: 1 }));
    await app.api.loadDocuments();
    fail = true;
    await app.api.loadDocuments();
    assert.equal(app.get('documentList').querySelectorAll('.document-item').length, 1);
    assert.match(app.get('toastContainer').textContent, /valid API key/);
});

test('successive upload batches remain queued and old completion cannot hide active progress', async () => {
    const first = deferred();
    const second = deferred();
    const uploaded = [];
    const app = createApp(async (url, options) => {
        if (url.endsWith('/ingest')) {
            uploaded.push(options.body.files[0].name);
            return uploaded.length === 1 ? first.promise : second.promise;
        }
        if (url.endsWith('/health')) return response(health);
        return response({ documents: documents(1), total_documents: 1 });
    });
    const batch1 = app.api.uploadFiles([{ name: 'first.txt' }]);
    const batch2 = app.api.uploadFiles([{ name: 'second.txt' }]);
    await settle();
    assert.deepEqual(uploaded, ['first.txt']);
    first.resolve(response({ success: true }));
    await batch1;
    await settle();
    assert.deepEqual(uploaded, ['first.txt', 'second.txt']);
    app.runTimers(300);
    assert.equal(app.get('uploadProgress').style.display, 'block');
    assert.match(app.get('progressText').textContent, /second.txt/);
    assert(app.get('progressFill').classList.contains('indeterminate'));
    second.resolve(response({ success: true }));
    await batch2;
    app.runTimers(300);
    assert.equal(app.get('uploadProgress').style.display, 'none');
});

test('API key control authenticates refreshes, can clear the key, and toggles mobile documents', async () => {
    const calls = [];
    const app = createApp(async (url, options) => {
        calls.push({ url, options });
        if (!options.headers['X-API-Key']) return response({ detail: 'Unauthorized' }, 401);
        return response(url.endsWith('/health') ? health : { documents: documents(1), total_documents: 1 });
    });
    await app.api.init();
    assert.match(app.get('modelStatus').textContent, /API key required/);
    app.get('apiKeyInput').value = 'test-only-key';
    await app.get('apiKeyForm').dispatch('submit');
    assert.equal(app.storage.get('scinexusrag_api_key'), 'test-only-key');
    assert(calls.slice(-2).every(call => call.options.headers['X-API-Key'] === 'test-only-key'));
    assert.equal(app.get('docCount').textContent, '1');
    await app.get('documentsToggle').click();
    assert(app.get('sidebar').classList.contains('open'));
    assert.equal(app.get('documentsToggle').getAttribute('aria-expanded'), 'true');
    await app.get('documentsToggle').click();
    assert.equal(app.get('documentsToggle').getAttribute('aria-expanded'), 'false');
    app.get('apiKeyInput').value = '';
    await app.get('apiKeyForm').dispatch('submit');
    assert(!app.storage.has('scinexusrag_api_key'));
    assert(calls.slice(-2).every(call => !call.options.headers['X-API-Key']));
});

test('blocked browser storage still permits a session API key', async () => {
    const calls = [];
    const app = createApp(async (url, options) => {
        calls.push(options);
        return response(url.endsWith('/health') ? health : { documents: [], total_documents: 0 });
    }, { storageBlocked: true });
    await app.api.init();
    app.get('apiKeyInput').value = 'session-only-key';
    await app.get('apiKeyForm').dispatch('submit');
    assert(calls.slice(-2).every(call => call.headers['X-API-Key'] === 'session-only-key'));
    assert.match(app.get('toastContainer').textContent, /saved for this session/);
});

test('untrusted answers and source fields stay text rather than executable markup', async () => {
    const app = createApp(async () => response({
        answer: '<img src=x onerror="alert(1)"> [1]',
        sources: [
            { filename: '"><img src=x onerror="alert(2)">', content: '<script>alert(3)</script>', page: '<img src=x onerror="alert(5)">' },
            { index: '<img src=x onerror="alert(6)">', filename: 'other.txt', content: 'Safe text.' }
        ],
        warnings: ['<svg onload="alert(4)">']
    }));
    app.api.state.documents = documents(1);
    app.get('chatInput').value = 'Question';
    await app.api.sendMessage();
    const answer = app.get('messages').querySelector('.assistant');
    assert.equal(answer.querySelectorAll('img').length, 0);
    assert.equal(answer.querySelectorAll('svg').length, 1, 'only the static copy icon is SVG');
    await answer.querySelector('.citation').click();
    assert.equal(app.get('sourcesList').querySelectorAll('script').length, 0);
    assert.equal(app.get('sourcesList').querySelectorAll('img').length, 0);
    assert.match(app.get('sourcesList').textContent, /<script>alert\(3\)<\/script>/);
    assert.equal(app.get('sourcesList').querySelectorAll('.source-number')[1].textContent,
        '<img src=x onerror="alert(6)">');
});

test('keyboard users can open, expand and close sources and return to the answer', async () => {
    const app = createApp(async url => {
        if (url.endsWith('/health')) return response(health);
        if (url.startsWith('/api/documents')) return response({ documents: documents(1), total_documents: 1 });
        return response({ answer: 'Finding [1]', sources: [{ filename: 'paper.txt', content: 'The complete source passage.' }] });
    });
    await app.api.init();
    app.get('chatInput').value = 'Question';
    await app.api.sendMessage();
    const button = app.get('messages').querySelector('.sources-link');
    button.focus();
    await button.click();
    await app.get('sourcesPanel').dispatch('transitionend', { propertyName: 'transform' });
    assert.equal(app.document.activeElement, app.get('closeSources'));

    const card = app.get('source-0');
    assert.equal(card.getAttribute('role'), 'button');
    assert.equal(card.getAttribute('tabindex'), '0');
    card.focus();
    await card.dispatch('keydown', { key: 'Enter' });
    assert.equal(card.getAttribute('aria-expanded'), 'true');
    assert(card.classList.contains('expanded'));
    assert.equal(card.getAttribute('aria-controls'), app.get('source-content-0').id);
    let prevented = false;
    await card.dispatch('keydown', { key: ' ', preventDefault() { prevented = true; } });
    assert(prevented);
    assert.equal(card.getAttribute('aria-expanded'), 'false');
    await app.dispatchDocument('keydown', { key: 'Escape' });
    assert(!app.get('sourcesPanel').classList.contains('open'));
    assert.equal(app.document.activeElement, button);
    assert.equal(app.get('messages').querySelector('.citation').getAttribute('aria-label'), 'Open source 1: paper.txt');
});

test('sources receive focus without a transition when reduced motion is enabled', async () => {
    const app = createApp(async () => response({
        answer: 'Finding [1]', sources: [{ filename: 'paper.txt', content: 'Evidence.' }]
    }), { reducedMotion: true });
    await app.api.init();
    app.api.state.documents = documents(1);
    app.get('chatInput').value = 'Question';
    await app.api.sendMessage();
    const button = app.get('messages').querySelector('.sources-link');
    button.focus();
    await button.click();

    app.runTimers('frame');
    assert.equal(app.document.activeElement, app.get('closeSources'));
    await app.dispatchDocument('keydown', { key: 'Escape' });
    assert.equal(app.document.activeElement, button);
    await button.click();
    app.get('chatInput').focus();
    app.runTimers('frame');
    assert.equal(app.document.activeElement, app.get('chatInput'));
});

test('opening sources preserves focus when the user moves to another control', async () => {
    const app = createApp(async () => response({
        answer: 'Finding [1]', sources: [{ filename: 'paper.txt', content: 'Evidence.' }]
    }));
    await app.api.init();
    app.api.state.documents = documents(1);
    app.get('chatInput').value = 'Question';
    await app.api.sendMessage();
    const button = app.get('messages').querySelector('.sources-link');
    button.focus();
    await button.click();
    await app.get('sourcesPanel').dispatch('transitionend', {
        propertyName: 'transform', target: app.get('source-0')
    });
    assert.equal(app.document.activeElement, button);
    app.get('chatInput').focus();
    await app.get('sourcesPanel').dispatch('transitionend', { propertyName: 'transform' });
    assert.equal(app.document.activeElement, app.get('chatInput'));
});

test('Escape closes the mobile document drawer and returns focus to its control', async () => {
    const app = createApp(async url => response(url.endsWith('/health') ? health : { documents: [] }));
    await app.api.init();
    await app.get('documentsToggle').click();
    assert(app.get('sidebar').classList.contains('open'));
    await app.dispatchDocument('keydown', { key: 'Escape' });
    assert(!app.get('sidebar').classList.contains('open'));
    assert.equal(app.get('documentsToggle').getAttribute('aria-expanded'), 'false');
    assert.equal(app.document.activeElement, app.get('documentsToggle'));
});

test('changing to an invalid API key clears the previous key’s corpus and model state', async () => {
    const app = createApp(async (url, options) => {
        if (options.headers['X-API-Key'] === 'invalid') return response({ detail: 'Unauthorized' }, 401);
        if (url.endsWith('/query')) return response({ answer: 'Earlier answer [1]', sources: [{ filename: 'old.txt', content: 'Old source' }] });
        return response(url.endsWith('/health') ? health : { documents: documents(1), total_documents: 1 });
    });
    await app.api.init();
    assert.equal(app.get('docCount').textContent, '1');
    assert.equal(app.get('llmStatus').textContent, 'local');
    app.get('chatInput').value = 'Question';
    await app.api.sendMessage();
    await app.get('messages').querySelector('.sources-link').click();
    assert(app.get('sourcesPanel').classList.contains('open'));
    app.get('apiKeyInput').value = 'invalid';
    await app.get('apiKeyForm').dispatch('submit');
    assert.equal(app.get('documentList').querySelectorAll('.document-item').length, 0);
    assert.equal(app.get('docCount').textContent, '0');
    assert.equal(app.get('chunkStatus').textContent, '0');
    assert.equal(app.get('llmStatus').textContent, '--');
    assert.equal(app.get('embeddingStatus').textContent, '--');
    assert(app.get('sendBtn').disabled);
    assert.match(app.get('modelStatus').textContent, /API key required/);
    assert.equal(app.get('messages').querySelectorAll('.message').length, 0);
    assert.equal(app.get('welcome').style.display, 'flex');
    assert.equal(app.get('sourcesList').children.length, 0);
    assert(!app.get('sourcesPanel').classList.contains('open'));
});

test('an old API key’s pending answer cannot restore old sources or unlock a newer request', async () => {
    const old = deferred();
    const current = deferred();
    let queries = 0;
    const app = createApp(async url => {
        if (url.endsWith('/query')) return ++queries === 1 ? old.promise : current.promise;
        return response(url.endsWith('/health') ? health : { documents: documents(1), total_documents: 1 });
    });
    await app.api.init();
    app.get('chatInput').value = 'Old question';
    const oldRequest = app.api.sendMessage();
    app.get('apiKeyInput').value = 'new-key';
    await app.get('apiKeyForm').dispatch('submit');
    app.get('chatInput').value = 'New question';
    const newRequest = app.api.sendMessage();
    old.resolve(response({ answer: 'Old answer', sources: [{ filename: 'old.txt', content: 'Old source' }] }));
    await oldRequest;
    assert.doesNotMatch(app.get('messages').textContent, /Old answer|Old question/);
    assert.equal(app.get('sourcesList').children.length, 0);
    assert(app.api.state.isProcessing);
    assert.equal(app.get('messages').querySelectorAll('.typing').length, 1);
    current.resolve(response({ answer: 'New answer', sources: [] }));
    await newRequest;
    assert.match(app.get('messages').textContent, /New answer/);
    assert(!app.api.state.isProcessing);
});

test('upload validation errors show the server’s explanation', async () => {
    const app = createApp(async () => response({ detail: [{ msg: 'File is empty.' }, { msg: 'Choose a supported document.' }] }, 422));
    await app.api.uploadFiles([{ name: 'empty.txt' }]);
    assert.match(app.get('toastContainer').textContent, /File is empty.; Choose a supported document./);
    assert.doesNotMatch(app.get('toastContainer').textContent, /\[object Object\]/);
});

test('query validation errors leave the input ready for another question', async () => {
    const app = createApp(async url => {
        if (url.endsWith('/health')) return response(health);
        if (url.startsWith('/api/documents')) return response({ documents: documents(1), total_documents: 1 });
        return response({ detail: [{ msg: 'Question is too long.' }] }, 422);
    });
    await app.api.init();
    app.get('chatInput').value = 'Question';
    await app.api.sendMessage();
    assert.match(app.get('messages').textContent, /Question is too long./);
    assert.equal(app.get('messages').querySelectorAll('.typing').length, 0);
    assert(!app.api.state.isProcessing);
    app.get('chatInput').value = 'Shorter question';
    await app.get('chatInput').dispatch('input');
    assert(!app.get('sendBtn').disabled);
    await app.get('sendBtn').click();
    await settle();
    assert.equal(app.get('messages').querySelectorAll('.message.error').length, 2);
});

test('answers preserve numbered steps, separate lists from paragraphs and keep citations usable', async () => {
    const app = createApp(async () => response({
        answer: 'Steps:\n\n3. Read the paper [1].\n4. Compare **results**.\n\n- Check *limitations*.\n- Keep the evidence.\n\nConclusion.\nSecond line.',
        sources: [{ filename: 'paper.txt', content: 'Evidence' }]
    }));
    app.api.state.documents = documents(1);
    app.get('chatInput').value = 'Question';
    await app.api.sendMessage();
    const content = app.get('messages').querySelector('.assistant').querySelector('.message-content');
    assert.deepEqual(content.children.map(child => child.tagName), ['p', 'ol', 'ul', 'p']);
    const steps = content.querySelector('ol').querySelectorAll('li');
    assert.deepEqual(steps.map(step => step.getAttribute('value')), ['3', '4']);
    assert.equal(content.querySelectorAll('strong').length, 1);
    assert.equal(content.querySelectorAll('em').length, 1);
    assert.equal(content.querySelectorAll('br').length, 1);
    await content.querySelector('.citation').click();
    assert.match(app.get('sourcesList').textContent, /Evidence/);
});
