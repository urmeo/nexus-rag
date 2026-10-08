const assert = require('node:assert/strict');
const { readFileSync } = require('node:fs');
const { join } = require('node:path');
const test = require('node:test');
const vm = require('node:vm');

const webDir = join(__dirname, '..', 'src', 'scinexusrag', 'web');
const html = readFileSync(join(webDir, 'index.html'), 'utf8');
const script = readFileSync(join(webDir, 'js', 'app.js'), 'utf8');

function createApp(fetch, { storageBlocked = false } = {}) {
    const clipboard = [];
    const storage = new Map();
    const timers = new Map();
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
            addEventListener() {}
        },
        window: {},
        navigator: { clipboard: { writeText: async text => clipboard.push(text) } },
        localStorage: {
            getItem: key => { if (storageBlocked) throw new Error('Storage blocked'); return storage.get(key); },
            setItem: (key, value) => { if (storageBlocked) throw new Error('Storage blocked'); storage.set(key, value); },
            removeItem: key => { if (storageBlocked) throw new Error('Storage blocked'); storage.delete(key); }
        },
        FormData: class { constructor() { this.files = []; } append(name, value) { this.files.push(value); } },
        setTimeout: (callback, delay) => { timers.set(++timerId, { callback, delay }); return timerId; },
        clearTimeout: id => timers.delete(id),
        setInterval() { throw new Error('Progress should be indeterminate, not a fabricated percentage.'); },
        clearInterval() {},
        fetch,
        confirm: () => true,
        console: { error() {} }
    });
    vm.runInContext(script, context, { filename: 'frontend/js/app.js' });
    const api = vm.runInContext('({ init, loadDocuments, uploadFiles, sendMessage, state })', context);
    return { api, get, clipboard, storage, timers,
        runTimers: delay => {
            for (const [id, timer] of [...timers]) {
                if (timer.delay === delay) { timers.delete(id); timer.callback(); }
            }
        }
    };
}

function response(data, status = 200) { return { ok: status < 400, status, json: async () => data }; }
function deferred() { let resolve; const promise = new Promise(done => { resolve = done; }); return { promise, resolve }; }
async function settle() { for (let i = 0; i < 20; i++) await Promise.resolve(); }
function documents(count, prefix = 'paper') {
    return Array.from({ length: count }, (_, index) => ({ id: `${prefix}-${index}`, filename: `${prefix}-${index}.txt` }));
}
const health = { llm_available: true, llm_model: 'local', embedding_model: 'embedding', total_documents: 1, total_chunks: 2 };

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
});

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

test('untrusted answers, filenames and source passages stay text rather than executable markup', async () => {
    const app = createApp(async () => response({
        answer: '<img src=x onerror="alert(1)"> [1]',
        sources: [{ filename: '"><img src=x onerror="alert(2)">', content: '<script>alert(3)</script>' }],
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
});
