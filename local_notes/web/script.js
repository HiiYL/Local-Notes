(() => {
  const input = document.getElementById('input');
  const runBtn = document.getElementById('run');
  const kEl = document.getElementById('k');
  const maxCharsEl = document.getElementById('maxChars');
  const providerEl = document.getElementById('provider');
  const llmModelEl = document.getElementById('llmModel');
  const statusEl = document.getElementById('status');
  const modeSearchBtn = document.getElementById('mode-search');
  const modeAskBtn = document.getElementById('mode-ask');
  const resultsEl = document.getElementById('results');
  const answerWrap = document.getElementById('answer');
  const sourcesEl = document.getElementById('sources');
  const answerText = document.getElementById('answerText');

  let mode = 'ask';

  function setMode(m) {
    mode = m;
    modeSearchBtn.classList.toggle('active', mode === 'search');
    modeAskBtn.classList.toggle('active', mode === 'ask');
    resultsEl.classList.toggle('hidden', mode !== 'search');
    answerWrap.classList.toggle('hidden', mode !== 'ask');
  }

  modeSearchBtn.addEventListener('click', () => setMode('search'));
  modeAskBtn.addEventListener('click', () => setMode('ask'));
  // default to Ask mode
  setMode('ask');

  async function search() {
    const q = input.value.trim();
    if (!q) return;
    statusEl.textContent = 'Searching…';
    resultsEl.innerHTML = '';
    try {
      const url = new URL('/search', window.location.origin);
      url.searchParams.set('q', q);
      url.searchParams.set('k', kEl.value || '6');
      url.searchParams.set('max_chars', maxCharsEl.value || '300');
      const resp = await fetch(url);
      if (!resp.ok) throw new Error(await resp.text());
      const items = await resp.json();
      items.forEach((r) => {
        const div = document.createElement('div');
        div.className = 'result';
        div.innerHTML = `
          <div class="title">${escapeHtml(r.title)}</div>
          <div class="snippet">${escapeHtml(r.text)}</div>
          <div class="meta">
            <span>Folder: ${escapeHtml(r.folder)}</span>
            <span>Chunk: ${r.chunk}</span>
            <span>Score: ${r.score.toFixed(4)}</span>
          </div>
        `;
        resultsEl.appendChild(div);
      });
    } catch (e) {
      resultsEl.innerHTML = `<div class="result">Error: ${escapeHtml(String(e))}</div>`;
    } finally {
      statusEl.textContent = '';
    }
  }

  async function ask() {
    const q = input.value.trim();
    if (!q) return;
    statusEl.textContent = 'Asking…';
    sourcesEl.innerHTML = '';
    let answerBuffer = '';
    renderMarkdown(answerBuffer);

    const body = {
      question: q,
      k: Number(kEl.value || '6'),
      provider: providerEl.value || 'ollama',
      llm_model: llmModelEl.value || null,
    };

    const es = new EventSourcePolyfill('/ask/stream', {
      headers: { 'Content-Type': 'application/json' },
      method: 'POST',
      payload: JSON.stringify(body),
    });

    es.addEventListener('sources', (ev) => {
      try {
        const arr = JSON.parse(ev.data || '[]');
        sourcesEl.innerHTML = arr.map((s) => (
          `<div class="result">`+
          `<div class="title"><strong>[${s.rank}]</strong> ${escapeHtml(s.title)}</div>`+
          `<div class="meta"><span>Folder: ${escapeHtml(s.folder)}</span>`+
          `<span>Chunk: ${s.chunk}</span></div>`+
          `</div>`
        )).join('');
      } catch {}
    });

    es.addEventListener('delta', (ev) => {
      answerBuffer += ev.data || '';
      renderMarkdown(answerBuffer);
    });

    es.addEventListener('done', () => {
      es.close();
      statusEl.textContent = '';
    });

    es.onerror = () => {
      statusEl.textContent = 'Error during streaming';
      es.close();
    };
  }

  runBtn.addEventListener('click', () => {
    if (mode === 'search') search(); else ask();
  });

  input.addEventListener('keydown', (e) => {
    if (e.key === 'Enter') {
      if (mode === 'search') search(); else ask();
    }
  });

  // Simple EventSource polyfill that supports POST payloads
  class EventSourcePolyfill {
    constructor(url, { headers = {}, method = 'GET', payload = null } = {}) {
      this.url = url;
      this.es = null;
      this.controller = new AbortController();
      this.listeners = new Map();
      this.connect({ headers, method, payload });
    }
    connect({ headers, method, payload }) {
      fetch(this.url, {
        method: method,
        headers: headers,
        body: payload,
        signal: this.controller.signal
      }).then(async (resp) => {
        if (!resp.ok || !resp.body) throw new Error('SSE failed');
        const reader = resp.body.getReader();
        const decoder = new TextDecoder();
        let buffer = '';
        while (true) {
          const { value, done } = await reader.read();
          if (done) break;
          buffer += decoder.decode(value, { stream: true });
          let idx;
          while ((idx = buffer.indexOf('\n\n')) >= 0) {
            const chunk = buffer.slice(0, idx);
            buffer = buffer.slice(idx + 2);
            this._dispatch(chunk);
          }
        }
        this._emit('done', {});
      }).catch((e) => {
        this._emit('error', e);
      });
    }
    addEventListener(type, handler) {
      if (!this.listeners.has(type)) this.listeners.set(type, []);
      this.listeners.get(type).push(handler);
    }
    _emit(type, ev) {
      (this.listeners.get(type) || []).forEach((h) => h(ev));
    }
    _dispatch(raw) {
      // parse simple SSE: event: <type> / data: <data>
      const lines = raw.split('\n');
      let type = 'message';
      let data = '';
      for (const line of lines) {
        if (line.startsWith('event: ')) type = line.slice(7).trim();
        else if (line.startsWith('data: ')) data += line.slice(6) + '\n';
      }
      if (data.endsWith('\n')) data = data.slice(0, -1);
      this._emit(type, { data });
    }
    close() { this.controller.abort(); }
  }

  function escapeHtml(s) {
    return s.replace(/[&<>\"']/g, (c) => ({'&':'&amp;','<':'&lt;','>':'&gt;','\"':'&quot;','\'':'&#39;'}[c]));
  }

  function renderMarkdown(mdText) {
    const html = DOMPurify.sanitize(marked.parse(mdText || ''));
    answerText.innerHTML = html;
  }
})();
