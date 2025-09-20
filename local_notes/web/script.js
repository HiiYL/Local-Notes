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
  const settingsToggle = document.getElementById('settingsToggle');
  const controlsInline = document.getElementById('controlsInline');
  const recencyEl = document.getElementById('recency');
  const recencyValEl = document.getElementById('recencyVal');
  // Indexing overlay elements
  const reindexBtn = document.getElementById('reindexBtn');
  const indexOverlay = document.getElementById('indexOverlay');
  const ovScan = document.getElementById('ovScan');
  const ovScanTxt = document.getElementById('ovScanTxt');
  const ovPlan = document.getElementById('ovPlan');
  const ovPlanTxt = document.getElementById('ovPlanTxt');
  const ovFetch = document.getElementById('ovFetch');
  const ovFetchTxt = document.getElementById('ovFetchTxt');
  const ovEmbed = document.getElementById('ovEmbed');
  const ovEmbedTxt = document.getElementById('ovEmbedTxt');
  const ovCancel = document.getElementById('ovCancel');
  const ovClose = document.getElementById('ovClose');

  let currentConv = null;
  let currentStream = null;
  let currentQuestion = '';
  // Configure markdown rendering
  try {
    marked.setOptions({ gfm: true, breaks: true, smartLists: true });
  } catch {}
  // Load settings
  try {
    const saved = JSON.parse(localStorage.getItem('ln_settings') || '{}');
    if (saved.k) kEl.value = String(saved.k);
    if (saved.provider) providerEl.value = saved.provider;
    if (saved.llmModel) llmModelEl.value = saved.llmModel;
    if (typeof saved.recency === 'number') {
      recencyEl.value = String(saved.recency);
      recencyValEl.textContent = `${saved.recency}%`;
    }

  // --- Indexing overlay logic ---
  let indexStream = null;
  function openOverlay() { indexOverlay.classList.remove('hidden'); }
  function closeOverlay() { indexOverlay.classList.add('hidden'); }
  ovCancel && ovCancel.addEventListener('click', () => { if (indexStream) { indexStream.close(); indexStream = null; } ovCancel.classList.add('hidden'); ovClose.classList.remove('hidden'); });
  ovClose && ovClose.addEventListener('click', () => { closeOverlay(); ovClose.classList.add('hidden'); ovCancel.classList.remove('hidden'); });

  function resetOverlay() {
    ovScan.max = 0; ovScan.value = 0; ovScanTxt.textContent = '';
    ovPlan.max = 1; ovPlan.value = 0; ovPlanTxt.textContent = '';
    ovFetch.max = 1; ovFetch.value = 0; ovFetchTxt.textContent = '';
    ovEmbed.removeAttribute('max'); ovEmbed.value = 0; ovEmbedTxt.textContent = '';
    ovClose.classList.add('hidden'); ovCancel.classList.remove('hidden');
  }

  function startIndexing() {
    resetOverlay();
    openOverlay();
    // Start SSE POST to /index/stream
    indexStream = new EventSourcePolyfill('/index/stream', {
      headers: { 'Content-Type': 'application/json' },
      method: 'POST',
      payload: JSON.stringify({}),
    });
    indexStream.addEventListener('scan', (ev) => {
      try { const d = JSON.parse(ev.data||'{}'); ovScan.max = d.total || 0; ovScan.value = 0; ovScanTxt.textContent = `${ovScan.value}/${ovScan.max}`; } catch {}
    });
    indexStream.addEventListener('plan', (ev) => {
      try { const d = JSON.parse(ev.data||'{}'); ovPlan.value = 1; ovPlanTxt.textContent = `${d.changed||0} changed`; } catch {}
    });
    indexStream.addEventListener('fetch', (ev) => {
      try { const d = JSON.parse(ev.data||'{}'); ovFetch.value = 1; ovFetchTxt.textContent = `${d.got||0} bodies`; } catch {}
    });
    indexStream.addEventListener('note', (ev) => {
      // advance scan if known max
      if (ovScan.max > 0) { ovScan.value = Math.min(ovScan.max, ovScan.value + 1); ovScanTxt.textContent = `${ovScan.value}/${ovScan.max}`; }
    });
    indexStream.addEventListener('embed', (ev) => {
      try { const d = JSON.parse(ev.data||'{}'); ovEmbed.value += (d.inc||1); ovEmbedTxt.textContent = `${ovEmbed.value} chunks`; } catch { ovEmbed.value += 1; }
    });
    indexStream.addEventListener('save', () => {
      ovEmbedTxt.textContent = `${ovEmbed.value} chunks • saving...`;
    });
    indexStream.addEventListener('done', () => {
      ovEmbedTxt.textContent = `${ovEmbed.value} chunks • complete`;
      ovCancel.classList.add('hidden'); ovClose.classList.remove('hidden');
      indexStream && indexStream.close(); indexStream = null;
    });
    indexStream.onerror = () => {
      ovEmbedTxt.textContent = `error`;
      ovCancel.classList.add('hidden'); ovClose.classList.remove('hidden');
      indexStream && indexStream.close(); indexStream = null;
    };
  }
  } catch {}
  function saveSettings() {
    const s = { k: Number(kEl.value||'6'), provider: providerEl.value||'ollama', llmModel: llmModelEl.value||'', recency: Number(recencyEl.value||'10') };
    localStorage.setItem('ln_settings', JSON.stringify(s));
  }

  // Heuristic normalizer to improve streamed markdown readability
  function normalizeMd(text) {
    if (!text) return '';
    let t = text;
    // Ensure a newline before list markers if they follow text without a break: "text:* item" => "text:\n* item"
    t = t.replace(/(:)\s*\*/g, '$1\n*');
    // Also support dashes as list markers
    t = t.replace(/(:)\s*-\s+/g, '$1\n- ');
    // Ensure headings start on new lines: "text## Heading" => "text\n\n## Heading"
    t = t.replace(/([^\n])(#\s?#+\s)/g, '$1\n\n$2');
    // Add paragraph break between sentence endings and next capitalized sentence when missing space/newline
    t = t.replace(/([.!?])([A-Z])/g, '$1\n\n$2');
    // If list markers appear mid-line without newline, insert it
    t = t.replace(/([^\n])\s+(\*|-)\s+/g, '$1\n$2 ');
    // Compact multiple blank lines
    t = t.replace(/\n{3,}/g, '\n\n');
    return t;
  }

  function stripLeadingEcho(text) {
    if (!text || !currentQuestion) return text;
    const q = currentQuestion.trim();
    // Direct echo
    if (text.startsWith(q)) {
      return text.slice(q.length).replace(/^[:\-\s]*/, '');
    }
    // With quotes around question
    const qQuoted = `"${q}"`;
    if (text.startsWith(qQuoted)) {
      return text.slice(qQuoted.length).replace(/^[:\-\s]*/, '');
    }
    return text;
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
      // toolbar
      const toolbar = document.createElement('div');
      toolbar.className = 'assistant-toolbar';
      const copyBtn = document.createElement('button');
      copyBtn.className = 'secondary';
      copyBtn.textContent = 'Copy Answer';
      copyBtn.addEventListener('click', async () => {
        const md = wrap.querySelector('.markdown');
        try { await navigator.clipboard.writeText(md ? md.innerText : ''); copyBtn.textContent = 'Copied'; setTimeout(()=>copyBtn.textContent='Copy Answer', 1000);} catch {}
      });
      const expandAllBtn = document.createElement('button');
      expandAllBtn.className = 'secondary';
      expandAllBtn.textContent = 'Expand All';
      expandAllBtn.addEventListener('click', () => {
        const chips = wrap.querySelectorAll('.source-chip');
        chips.forEach(ch => ch.click());
      });
      const collapseAllBtn = document.createElement('button');
      collapseAllBtn.className = 'secondary';
      collapseAllBtn.textContent = 'Collapse All';
      collapseAllBtn.addEventListener('click', () => {
        const container = wrap.querySelector('.snippets');
        if (container) container.innerHTML = '';
      });
      toolbar.appendChild(copyBtn);
      toolbar.appendChild(expandAllBtn);
      toolbar.appendChild(collapseAllBtn);
      wrap.appendChild(toolbar);
      const md = document.createElement('div');
      md.className = 'markdown';
      md.innerHTML = renderMd ? DOMPurify.sanitize(marked.parse(content || '')) : '';
      if (!renderMd) md.setAttribute('data-stream', '1');
      wrap.appendChild(md);
      const src = document.createElement('div');
      src.className = 'sources';
      wrap.appendChild(src);
      // snippets container lives right after sources
      const snippets = document.createElement('div');
      snippets.className = 'snippets';
      wrap.appendChild(snippets);
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
    const cleaned = stripLeadingEcho(next);
    md.innerHTML = DOMPurify.sanitize(marked.parse(normalizeMd(cleaned)));
    messagesEl.scrollTop = messagesEl.scrollHeight;
  }

  function finalizeStreamingBubble(bubble) {
    const md = bubble.querySelector('.markdown[data-stream="1"]');
    if (!md) return;
    const buf = md.getAttribute('data-buf') || '';
    md.removeAttribute('data-stream');
    md.removeAttribute('data-buf');
    // Ensure final render is set
    const cleaned = stripLeadingEcho(buf);
    md.innerHTML = DOMPurify.sanitize(marked.parse(normalizeMd(cleaned)));
  }

  async function send() {
    const q = input.value.trim();
    if (!q) return;
    currentQuestion = q;
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
      recency_alpha: Math.max(0, Math.min(1, Number(recencyEl.value || '10')/100)),
    };

    const streamUrl = `/conv/${currentConv}/ask/stream`;
    const es = new EventSourcePolyfill(streamUrl, {
      headers: { 'Content-Type': 'application/json' },
      method: 'POST',
      payload: JSON.stringify(body),
    });
    currentStream = es;

    let assistantBubble = appendMessage('assistant', '', false);
    // show typing indicator until first token arrives
    let typing = document.createElement('div');
    typing.className = 'typing';
    typing.innerHTML = '<span class="dot"></span><span class="dot"></span><span class="dot"></span>';
    assistantBubble.appendChild(typing);
    // attach sources chips on sources event
    const srcContainer = assistantBubble.querySelector('.sources');
    const attachSources = (arr) => {
      if (!srcContainer) return;
      srcContainer.innerHTML = '';
      (arr||[]).forEach(s => {
        const chip = document.createElement('span');
        chip.className = 'source-chip';
        const chipLabel = `[${s.rank}] ${s.title || ''}` + (s.heading ? ` — ${s.heading}` : '');
        chip.textContent = chipLabel.trim();
        if (s.text) {
          const snippet = String(s.text);
          chip.title = snippet.length > 500 ? (snippet.slice(0, 500) + '…') : snippet;
        }
        // clickable expand/collapse
        chip.addEventListener('click', () => toggleSnippet(assistantBubble, s));
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
      if (typing && typing.parentElement) { typing.remove(); typing = null; }
      updateStreamingBubble(assistantBubble, ev.data);
    });
    es.addEventListener('done', () => {
      statusEl.textContent = '';
      if (typing && typing.parentElement) { typing.remove(); typing = null; }
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
  settingsToggle.addEventListener('click', () => {
    controlsInline.classList.toggle('hidden');
  });
  reindexBtn && reindexBtn.addEventListener('click', () => startIndexing());
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

  function toggleSnippet(bubble, source) {
    const container = bubble.querySelector('.snippets');
    if (!container) return;
    const id = `snip-${source.rank}-${source.chunk}`;
    let card = container.querySelector(`[data-id="${id}"]`);
    if (card) {
      card.remove();
      return;
    }
    card = document.createElement('div');
    card.className = 'snippet-card';
    card.setAttribute('data-id', id);
    const title = document.createElement('div');
    title.className = 'snippet-title';
    const updated = source.updated_at ? new Date(source.updated_at*1000) : null;
    const updatedStr = updated ? `  •  Updated: ${updated.toISOString().slice(0,10)}` : '';
    title.textContent = `${source.title}  •  Folder: ${source.folder}  •  Chunk: ${source.chunk}${updatedStr}`;
    const pre = document.createElement('pre');
    pre.textContent = source.text || '';
    const actions = document.createElement('div');
    actions.className = 'snippet-actions';
    const copyBtn = document.createElement('button');
    copyBtn.textContent = 'Copy';
    copyBtn.addEventListener('click', async () => {
      try { await navigator.clipboard.writeText(source.text || ''); copyBtn.textContent = 'Copied'; setTimeout(()=>copyBtn.textContent='Copy', 1000);} catch {}
    });
    actions.appendChild(copyBtn);
    card.appendChild(title);
    card.appendChild(pre);
    card.appendChild(actions);
    container.appendChild(card);
  }
})();
