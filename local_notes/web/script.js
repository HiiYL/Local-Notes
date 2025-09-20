(() => {
  const input = document.getElementById('input');
  const sendBtn = document.getElementById('send');
  const stopBtn = document.getElementById('stop');
  const messagesEl = document.getElementById('messages');
  const statusEl = document.getElementById('status');
  const kEl = document.getElementById('k');
  const providerEl = document.getElementById('provider');
  const llmModelEl = document.getElementById('llmModel');
  const convListEl = document.getElementById('conversations');
  const newConvBtn = document.getElementById('newConv');

  let currentConv = null;
  let currentStream = null;
  // Load settings
  try {
    const saved = JSON.parse(localStorage.getItem('ln_settings') || '{}');
    if (saved.k) kEl.value = String(saved.k);
    if (saved.provider) providerEl.value = saved.provider;
    if (saved.llmModel) llmModelEl.value = saved.llmModel;
  } catch {}
  function saveSettings() {
    const s = { k: Number(kEl.value||'6'), provider: providerEl.value||'ollama', llmModel: llmModelEl.value||'' };
    localStorage.setItem('ln_settings', JSON.stringify(s));
  }
  kEl.addEventListener('change', saveSettings);
  providerEl.addEventListener('change', saveSettings);
  llmModelEl.addEventListener('change', saveSettings);

  function uuid() {
    return 'xxxxxxxx-xxxx-4xxx-yxxx-xxxxxxxxxxxx'.replace(/[xy]/g, c => {
      const r = Math.random()*16|0, v = c == 'x' ? r : (r&0x3|0x8);
      return v.toString(16);
    });
  }

  async function listConversations() {
    const resp = await fetch('/conv');
    const items = await resp.json();
    convListEl.innerHTML = '';
    items.forEach(c => {
      const div = document.createElement('div');
      div.className = 'conv-item' + (currentConv === c.id ? ' active' : '');
      div.textContent = c.title;
      div.onclick = () => selectConversation(c.id);
      convListEl.appendChild(div);
    });
  }

  async function createConversation() {
    const id = uuid();
    const resp = await fetch('/conv', {
      method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ id, title: 'New Conversation' })
    });
    if (!resp.ok) return;
    currentConv = id;
    await listConversations();
    await loadMessages();
    input.focus();
  }

  async function selectConversation(id) {
    currentConv = id;
    await listConversations();
    await loadMessages();
  }

  async function loadMessages() {
    if (!currentConv) return;
    messagesEl.innerHTML = '';
    const resp = await fetch(`/conv/${currentConv}/messages?limit=200`);
    const msgs = await resp.json();
    msgs.forEach(m => {
      const bubble = appendMessage(m.role, m.content, m.role === 'assistant');
      if (m.role === 'assistant') {
        const srcContainer = bubble.querySelector('.sources');
        if (srcContainer) {
          try {
            if (m.citations) {
              const arr = JSON.parse(m.citations || '[]');
              if (Array.isArray(arr) && arr.length) {
                srcContainer.innerHTML = '';
                arr.forEach(s => {
                  const chip = document.createElement('span');
                  chip.className = 'source-chip';
                  const label = `[${s.rank}] ${s.title || ''}`.trim();
                  chip.textContent = label;
                  if (s.text) {
                    const snippet = String(s.text);
                    chip.title = snippet.length > 500 ? (snippet.slice(0, 500) + '…') : snippet;
                  }
                  srcContainer.appendChild(chip);
                });
                return;
              }
            }
          } catch {}
          // fallback: restore citation chips by parsing [n] from the text
          const cited = Array.from((m.content || '').matchAll(/\[(\d+)\]/g)).map(x => Number(x[1]));
          if (cited.length) {
            srcContainer.innerHTML = '';
            [...new Set(cited)].forEach(n => {
              const chip = document.createElement('span');
              chip.className = 'source-chip';
              chip.textContent = `[${n}]`;
              srcContainer.appendChild(chip);
            });
          }
        }
      }
    });
    messagesEl.scrollTop = messagesEl.scrollHeight;
  }

  function appendMessage(role, content, renderMd=true) {
    const wrap = document.createElement('div');
    wrap.className = 'bubble ' + (role === 'user' ? 'user' : 'assistant');
    if (role === 'assistant') {
      const md = document.createElement('div');
      md.className = 'markdown';
      md.innerHTML = renderMd ? DOMPurify.sanitize(marked.parse(content || '')) : '';
      if (!renderMd) md.setAttribute('data-stream', '1');
      wrap.appendChild(md);
      const src = document.createElement('div');
      src.className = 'sources';
      wrap.appendChild(src);
    } else {
      wrap.textContent = content;
    }
    messagesEl.appendChild(wrap);
    return wrap;
  }

  function updateStreamingBubble(bubble, delta) {
    const md = bubble.querySelector('.markdown[data-stream="1"]');
    if (!md) return;
    const prev = md.getAttribute('data-buf') || '';
    const next = prev + (delta || '');
    md.setAttribute('data-buf', next);
    md.innerHTML = DOMPurify.sanitize(marked.parse(next));
    messagesEl.scrollTop = messagesEl.scrollHeight;
  }

  function finalizeStreamingBubble(bubble) {
    const md = bubble.querySelector('.markdown[data-stream="1"]');
    if (!md) return;
    const buf = md.getAttribute('data-buf') || '';
    md.removeAttribute('data-stream');
    md.removeAttribute('data-buf');
    // Ensure final render is set
    md.innerHTML = DOMPurify.sanitize(marked.parse(buf));
  }

  async function send() {
    const q = input.value.trim();
    if (!q) return;
    if (!currentConv) await createConversation();
    appendMessage('user', q, false);
    input.value = '';
    statusEl.textContent = 'Thinking…';
    sendBtn.disabled = true;
    stopBtn.classList.remove('hidden');

    const body = {
      question: q,
      k: Number(kEl.value || '6'),
      provider: providerEl.value || 'ollama',
      llm_model: llmModelEl.value || null,
    };

    const streamUrl = `/conv/${currentConv}/ask/stream`;
    const es = new EventSourcePolyfill(streamUrl, {
      headers: { 'Content-Type': 'application/json' },
      method: 'POST',
      payload: JSON.stringify(body),
    });
    currentStream = es;

    let assistantBubble = appendMessage('assistant', '', false);
    // attach sources chips on sources event
    const srcContainer = assistantBubble.querySelector('.sources');
    const attachSources = (arr) => {
      if (!srcContainer) return;
      srcContainer.innerHTML = '';
      (arr||[]).forEach(s => {
        const chip = document.createElement('span');
        chip.className = 'source-chip';
        chip.textContent = `[${s.rank}] ${s.title}`;
        if (s.text) {
          const snippet = String(s.text);
          chip.title = snippet.length > 500 ? (snippet.slice(0, 500) + '…') : snippet;
        }
        srcContainer.appendChild(chip);
      });
    };
    es.addEventListener('sources', (ev) => {
      // Initially show all, will be replaced by 'citations' when available
      try { attachSources(JSON.parse(ev.data||'[]')); } catch {}
    });
    es.addEventListener('citations', (ev) => {
      // Replace with cited-only sources
      try { attachSources(JSON.parse(ev.data||'[]')); } catch {}
    });
    es.addEventListener('delta', (ev) => {
      updateStreamingBubble(assistantBubble, ev.data);
    });
    es.addEventListener('done', () => {
      statusEl.textContent = '';
      finalizeStreamingBubble(assistantBubble);
      es.close();
      currentStream = null;
      sendBtn.disabled = false;
      stopBtn.classList.add('hidden');
    });
    es.onerror = () => {
      statusEl.textContent = 'Error during streaming';
      es.close();
      currentStream = null;
      sendBtn.disabled = false;
      stopBtn.classList.add('hidden');
    };
  }

  newConvBtn.addEventListener('click', createConversation);
  sendBtn.addEventListener('click', send);
  stopBtn.addEventListener('click', () => {
    if (currentStream) {
      currentStream.close();
      currentStream = null;
      sendBtn.disabled = false;
      stopBtn.classList.add('hidden');
      statusEl.textContent = 'Stopped';
      setTimeout(() => statusEl.textContent = '', 800);
    }
  });
  input.addEventListener('keydown', (e) => {
    if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); send(); }
  });

  // On load: list convos, create one if none
  (async () => {
    await listConversations();
    if (!convListEl.firstChild) {
      await createConversation();
    } else {
      // select first
      const resp = await fetch('/conv');
      const items = await resp.json();
      if (items.length) await selectConversation(items[0].id);
    }
  })();

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
})();
