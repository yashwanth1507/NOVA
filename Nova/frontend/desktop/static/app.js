/* ═══════════════════════════════════════════════════════════
   Nova Desktop — Main Application
   ═══════════════════════════════════════════════════════════ */

const API = '/api'
let sid = null           // current session ID
let docId = null         // active document ID
let docName = ''         // active document name
let activeChunkId = null // chunk used for quiz generation
let settingsTimer = null

// ── Session ↔ Document Context (localStorage) ────────────
// Persists which document was active per session so it can be
// restored when the user comes back to a session after a reload.
const SESSION_DOC_KEY = 'nova_session_doc' // JSON map: { sessionId → { docId, docName, chunkId } }

function _saveSessionDoc(sessionId, id, title, chunkId) {
  if (!sessionId) return
  try {
    const map = JSON.parse(localStorage.getItem(SESSION_DOC_KEY) || '{}')
    map[sessionId] = { docId: id, docName: title, chunkId: chunkId || null }
    // Keep map small — drop sessions older than the 20 most-recent entries
    const keys = Object.keys(map)
    if (keys.length > 20) delete map[keys[0]]
    localStorage.setItem(SESSION_DOC_KEY, JSON.stringify(map))
  } catch { /* storage quota */ }
}

function _loadSessionDoc(sessionId) {
  if (!sessionId) return null
  try {
    const map = JSON.parse(localStorage.getItem(SESSION_DOC_KEY) || '{}')
    return map[sessionId] || null
  } catch { return null }
}

function _clearSessionDoc(sessionId) {
  if (!sessionId) return
  try {
    const map = JSON.parse(localStorage.getItem(SESSION_DOC_KEY) || '{}')
    delete map[sessionId]
    localStorage.setItem(SESSION_DOC_KEY, JSON.stringify(map))
  } catch { /* ignore */ }
}

// ── Init ─────────────────────────────────────────────────

document.addEventListener('DOMContentLoaded', () => {
  loadSessions()
  loadDocuments()
  loadSettings()
  setupNetworkMonitor()
  setupDragDrop()
  setupChatInput()
  setupCopyButtons()
  setupQuizInteractions()
  checkModelStatusOnStartup()
})

// ── Startup Model Status Check ───────────────────────────
// Polls /models/status once on load; if the model is not installed
// yet shows a warning in the sidebar status bar so the user knows
// immediately without having to open Settings.
async function checkModelStatusOnStartup() {
  try {
    const result = await api('GET', '/models/status')
    if (result.state === 'missing' || (result.state !== 'ready' && !result.available)) {
      setStatus('Model not installed — open Settings to install', false)
      toast(
        '⚠️ Local model not installed. Go to Settings → Install model to download it.',
        'warning',
        8000
      )
    } else if (result.state === 'ready' && result.available) {
      setStatus('Ready')
    }
  } catch {
    // Model installer not reachable yet — server may still be starting
  }

  // Hybrid retrieval silently degrades to keyword search without MiniLM, so
  // surface the state instead of only logging it on the server.
  try {
    const embeddings = await api('GET', '/models/embeddings/status?probe=true')
    if (!embeddings.available) {
      toast(
        'ℹ️ Semantic search is off (embedding model not installed). Settings → Install embeddings enables it.',
        'info',
        8000
      )
    }
    renderEmbeddingStatus(embeddings)
  } catch {
    // Embedding installer not reachable yet
  }
}

// ── Toast Notifications ──────────────────────────────────

function toast(msg, type = 'info', duration = 4000) {
  const container = document.getElementById('toast-container')
  const el = document.createElement('div')
  el.className = `toast ${type}`
  const icon = type === 'error' ? '✕' : type === 'success' ? '✓' : 'ℹ'
  el.innerHTML = `<span class="toast-icon">${icon}</span><span>${esc(msg)}</span>`
  container.appendChild(el)
  setTimeout(() => {
    el.classList.add('removing')
    setTimeout(() => el.remove(), 200)
  }, duration)
}

// ── API Helper ───────────────────────────────────────────

async function api(method, path, body) {
  const opts = {
    method,
    headers: body ? { 'Content-Type': 'application/json' } : {},
    body: body ? JSON.stringify(body) : undefined
  }
  const res = await fetch(API + path, opts)
  if (!res.ok) {
    let msg
    try { const e = await res.json(); msg = e.detail || e.message || res.statusText }
    catch { msg = await res.text().catch(() => res.statusText) }
    throw new Error(msg)
  }
  return res.json()
}

// ── Loading ──────────────────────────────────────────────

function loading(on, text) {
  document.getElementById('loading').hidden = !on
  document.getElementById('loading-text').textContent = text || 'Processing...'
}

function setStatus(text, ok = true) {
  document.getElementById('status-text').textContent = text
  const dot = document.getElementById('status-dot')
  dot.className = 'status-dot'
  if (!ok) dot.classList.add('error')
}

// ── Network Monitor ──────────────────────────────────────

function setupNetworkMonitor() {
  const statusBar = document.getElementById('network-status')
  function update() {
    const engineBadge = document.getElementById('engine-badge')
    const isOnlineEngine = engineBadge && engineBadge.textContent !== 'local'
    if (!navigator.onLine && isOnlineEngine) {
      statusBar.classList.add('offline')
      statusBar.hidden = false
    } else {
      statusBar.classList.remove('offline')
      statusBar.hidden = true
    }
  }
  window.addEventListener('online', update)
  window.addEventListener('offline', update)
  update()
}

// ── Escaping ─────────────────────────────────────────────

function esc(s) {
  const d = document.createElement('div')
  d.textContent = s
  return d.innerHTML
}

// ── View Switching ───────────────────────────────────────

function switchView(view) {
  document.querySelectorAll('.view').forEach(el => el.classList.remove('active'))
  document.querySelectorAll('.nav-btn').forEach(el => el.classList.remove('active'))
  document.getElementById('view-' + view)?.classList.add('active')
  document.querySelector(`.nav-btn[data-view="${view}"]`)?.classList.add('active')
  // Remove any open chat model picker when switching views
  document.querySelector('.chat-model-picker')?.remove()
  if (view === 'quizzes') { loadQuizChunks(); loadQuizResults() }
  if (view === 'documents') loadDocuments()
}

// ── Sessions ─────────────────────────────────────────────

async function loadSessions() {
  try {
    const sessions = await api('GET', '/sessions')
    const sel = document.getElementById('session-select')
    sel.innerHTML = '<option value="">+ New session</option>'
    sessions.forEach(s => {
      const opt = document.createElement('option')
      opt.value = s.id
      const d = new Date(s.created_at * 1000)
      opt.textContent = `Session ${s.id.slice(0, 6)}… ${d.toLocaleDateString()}`
      sel.appendChild(opt)
    })
  } catch (e) {
    // Server might not be ready yet, that's OK
  }
}

async function newSession() {
  try {
    setStatus('Creating session…')
    const s = await api('POST', '/sessions', {})
    sid = s.id
    document.getElementById('session-select').value = ''
    document.getElementById('messages').innerHTML =
      '<div class="welcome"><div class="welcome-icon">💬</div><h3>New Session</h3><p>Ask a question to get started.</p></div>'
    loadSessions()
    document.getElementById('engine-badge').textContent = s.engine_used
    document.getElementById('btn-delete-session').hidden = false
    setStatus('Ready')
    toast('New session created', 'success', 2000)
  } catch (e) {
    toast('Failed to create session: ' + e.message, 'error')
    setStatus('Error', false)
  }
}

async function switchSession(id) {
  if (!id) {
    sid = null
    clearDoc()
    document.getElementById('messages').innerHTML =
      '<div class="welcome"><div class="welcome-icon">📚</div><h3>Welcome to Nova</h3><p>Upload a PDF and start asking questions.</p></div>'
    document.getElementById('btn-delete-session').hidden = true
    return
  }
  sid = id
  document.getElementById('btn-delete-session').hidden = false
  try {
    setStatus('Loading session…')
    const [msgs, s] = await Promise.all([
      api('GET', '/sessions/' + sid + '/messages'),
      api('GET', '/sessions/' + sid)
    ])
    renderMessages(msgs)
    document.getElementById('engine-badge').textContent = s.engine_used

    // Restore the document that was active in this session
    const saved = _loadSessionDoc(sid)
    if (saved && saved.docId) {
      // Verify the document still exists before restoring
      try {
        await api('GET', '/documents/' + saved.docId)
        setDoc(saved.docId, saved.docName, saved.chunkId)
        toast(`Restored document: "${saved.docName}"`, 'info', 2500)
      } catch {
        // Document was deleted — clear the stale entry
        _clearSessionDoc(sid)
        clearDoc()
      }
    } else {
      clearDoc()
    }

    setStatus('Ready')
  } catch (e) {
    toast('Failed to load session: ' + e.message, 'error')
    setStatus('Error', false)
  }
}

// ── Streamed Render Throttle ──────────────────────────────
// Re-rendering full Markdown on every token is O(n^2) DOM work. Batch it to
// one paint per animation frame instead.
let _streamRenderPending = false
let _streamRenderTarget = null
let _streamRenderText = ''

function scheduleStreamRender(msgEl, text) {
  _streamRenderTarget = msgEl
  _streamRenderText = text
  if (_streamRenderPending) return
  _streamRenderPending = true
  requestAnimationFrame(() => {
    _streamRenderPending = false
    if (_streamRenderTarget) updateMessageContent(_streamRenderTarget, _streamRenderText)
  })
}

function flushStreamRender() {
  _streamRenderPending = false
  if (_streamRenderTarget) updateMessageContent(_streamRenderTarget, _streamRenderText)
}

function resetStreamRender() {
  _streamRenderPending = false
  _streamRenderTarget = null
  _streamRenderText = ''
}

// ── Confirm Dialog ───────────────────────────────────────-

let _confirmCallback = null

function confirm(title, message, onYes, yesLabel) {
  document.getElementById('confirm-title').textContent = title
  document.getElementById('confirm-message').textContent = message
  document.getElementById('confirm-yes').textContent = yesLabel || 'Delete'
  document.getElementById('confirm-modal').hidden = false
  _confirmCallback = onYes
}

function closeConfirm() {
  document.getElementById('confirm-modal').hidden = true
  _confirmCallback = null
}

function confirmAction() {
  document.getElementById('confirm-modal').hidden = true
  if (_confirmCallback) _confirmCallback()
  _confirmCallback = null
}

// Close modal on overlay click
document.getElementById('confirm-modal').addEventListener('click', e => {
  if (e.target === e.currentTarget) closeConfirm()
})

async function deleteSession() {
  if (!sid) return
  confirm(
    'Delete Session',
    'This will permanently delete this session and all its messages and quizzes.',
    async () => {
      try {
        const deletedSid = sid
        await api('DELETE', '/sessions/' + deletedSid)
        _clearSessionDoc(deletedSid)
        sid = null
        document.getElementById('messages').innerHTML =
          '<div class="welcome"><div class="welcome-icon">📚</div><h3>Session deleted</h3><p>Start a new session to begin.</p></div>'
        document.getElementById('btn-delete-session').hidden = true
        loadSessions()
        toast('Session deleted', 'info', 2000)
      } catch (e) {
        toast('Failed to delete session: ' + e.message, 'error')
      }
    },
    'Delete'
  )
}

// ── Chat ─────────────────────────────────────────────────

function setupChatInput() {
  document.getElementById('chat-input').addEventListener('keydown', e => {
    if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); send() }
  })
}

function onChatInput(el) {
  el.style.height = 'auto'
  el.style.height = Math.min(el.scrollHeight, 120) + 'px'
  document.getElementById('send-btn').disabled = !el.value.trim()
}

async function send() {
  const el = document.getElementById('chat-input')
  const query = el.value.trim()
  const sendBtn = document.getElementById('send-btn')
  if (!query || sendBtn.disabled) return

  // Disable input while generating
  sendBtn.disabled = true
  el.disabled = true

  if (!sid) {
    try {
      const s = await api('POST', '/sessions', {})
      sid = s.id
      document.getElementById('btn-delete-session').hidden = false
      loadSessions()
    } catch (e) {
      toast('Create a session first', 'error')
      sendBtn.disabled = false
      el.disabled = false
      el.focus()
      return
    }
  }

  appendMessage('user', query)
  el.value = ''
  onChatInput(el)

  // Show typing indicator
  showTypingIndicator()
  setStatus('Thinking…', true)

  let assistantMsgEl = null
  let accumulatedText = ''

  try {
    const res = await fetch(API + '/chat/stream', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        session_id: sid,
        query: query,
        document_id: docId || undefined
      })
    })

    if (!res.ok || !res.body) {
      // Fallback to non-streaming /chat endpoint if stream not available
      const r = await api('POST', '/chat', {
        session_id: sid,
        query: query,
        document_id: docId || undefined
      })
      hideTypingIndicator()
      appendMessage('assistant', r.response)
      setStatus('Ready')
      return
    }

    const reader = res.body.getReader()
    const decoder = new TextDecoder('utf-8')
    let buffer = ''

    while (true) {
      const { done, value } = await reader.read()
      if (done) break

      buffer += decoder.decode(value, { stream: true })
      const lines = buffer.split('\n\n')
      buffer = lines.pop() // keep unparsed partial line

      for (const line of lines) {
        const trimmed = line.trim()
        if (!trimmed.startsWith('data:')) continue
        const jsonStr = trimmed.slice(5).trim()
        if (!jsonStr) continue

        // Parse and handle separately: a parse failure is a partial frame and
        // is ignorable, but a server-sent error must reach the user instead of
        // being swallowed by the same catch block.
        let data
        try {
          data = JSON.parse(jsonStr)
        } catch {
          continue
        }
        if (data.error) {
          throw new Error(data.error)
        }
        if (data.token) {
          if (!assistantMsgEl) {
            hideTypingIndicator()
            assistantMsgEl = appendMessage('assistant', '')
            setStatus('Generating…', true)
          }
          accumulatedText += data.token
          scheduleStreamRender(assistantMsgEl, accumulatedText)
        }
      }
    }

    flushStreamRender()
    if (!assistantMsgEl) {
      if (accumulatedText) {
        appendMessage('assistant', accumulatedText)
      } else {
        appendMessage('assistant', 'I could not find an answer in the uploaded documents.')
      }
    }
    setStatus('Ready')
  } catch (e) {
    hideTypingIndicator()
    const msg = e.message || ''
    let errMsg
    if (msg.includes('Failed to fetch') || msg.includes('NetworkError')) {
      errMsg = 'Connection lost — please check if the server is running.'
    } else if (msg.includes('timeout') || msg.includes('timed out')) {
      errMsg = 'Request timed out. The model is generating — try a simpler question.'
    } else {
      errMsg = msg
    }
    if (assistantMsgEl && accumulatedText) {
      resetStreamRender()
      updateMessageContent(assistantMsgEl, accumulatedText + '\n\n⚠️ ' + errMsg)
    } else {
      appendMessage('assistant', '⚠️ ' + errMsg)
    }
    setStatus('Error', false)
  } finally {
    resetStreamRender()
    sendBtn.disabled = false
    el.disabled = false
    el.focus()
  }
}

function updateMessageContent(msgEl, text) {
  if (!msgEl) return
  const container = document.getElementById('messages')
  msgEl.innerHTML = formatMessage(text) + '<div class="msg-timestamp">' + formatTimestamp(Date.now()) + '</div>'
  container.scrollTop = container.scrollHeight
}

function showTypingIndicator() {
  const container = document.getElementById('messages')
  if (document.getElementById('typing-indicator')) return
  const typingEl = document.createElement('div')
  typingEl.className = 'typing-indicator'
  typingEl.id = 'typing-indicator'
  typingEl.innerHTML = '<span class="typing-dot"></span><span class="typing-dot"></span><span class="typing-dot"></span>'
  container.appendChild(typingEl)
  container.scrollTop = container.scrollHeight
}

function hideTypingIndicator() {
  const el = document.getElementById('typing-indicator')
  if (el) el.remove()
}

function appendMessage(role, text) {
  const container = document.getElementById('messages')
  // Remove welcome if present
  const welcome = container.querySelector('.welcome')
  if (welcome) welcome.remove()

  const div = document.createElement('div')
  div.className = 'msg ' + role
  div.innerHTML = formatMessage(text) + '<div class="msg-timestamp">' + formatTimestamp(Date.now()) + '</div>'
  container.appendChild(div)
  container.scrollTop = container.scrollHeight
  return div
}

function formatMessage(text) {
  // Process code blocks BEFORE HTML escaping to avoid double-escape
  // Extract code blocks, replace with placeholders, escape text, then restore blocks
  const codeBlocks = []
  let h = text.replace(/```(\w*)\n([\s\S]*?)```/g, (match, lang, code) => {
    const idx = codeBlocks.length
    codeBlocks.push('<pre><code>' + esc(code) + '</code><button class="copy-btn" onclick="copyCode(this)">📋 Copy</button></pre>')
    return '%%CODEBLOCK_' + idx + '%%'
  })
  // Now escape the rest
  h = esc(h)
  // Restore code block placeholders (which are already safely escaped)
  h = h.replace(/%%CODEBLOCK_(\d+)%%/g, (match, idx) => codeBlocks[parseInt(idx)])
  // Inline code
  h = h.replace(/`([^`]+)`/g, '<code>$1</code>')
  // Bold
  h = h.replace(/\*\*(.+?)\*\*/g, '<strong>$1</strong>')
  // Paragraphs
  h = h.replace(/\n\n/g, '</p><p>')
  h = h.replace(/\n/g, '<br>')
  return '<p>' + h + '</p>'
}

function formatTimestamp(ts) {
  const d = new Date(ts)
  return d.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })
}

function copyCode(btn) {
  const code = btn.parentElement.querySelector('code')
  if (!code) return
  const text = code.textContent
  navigator.clipboard.writeText(text).then(() => {
    btn.textContent = '✓ Copied!'
    setTimeout(() => { btn.textContent = '📋 Copy' }, 2000)
  }).catch(() => {
    btn.textContent = 'Failed'
  })
}

function setupCopyButtons() {
  // Delegate copy button clicks for dynamically loaded messages
  document.getElementById('messages').addEventListener('click', e => {
    const btn = e.target.closest('.copy-btn')
    if (btn) copyCode(btn)
  })
}

function renderMessages(msgs) {
  const container = document.getElementById('messages')
  container.innerHTML = ''
  if (!msgs.length) {
    container.innerHTML =
      '<div class="welcome"><div class="welcome-icon">💬</div><h3>Session Started</h3><p>Ask a question to begin.</p></div>'
    return
  }
  msgs.forEach(m => appendMessage(m.role, m.content))
}

// ── Document Context ────────────────────────────────────

function setDoc(id, title, chunkId = null) {
  docId = id
  docName = title
  activeChunkId = chunkId
  const ind = document.getElementById('doc-indicator')
  if (ind) {
    ind.hidden = false
    const nameEl = document.getElementById('doc-name')
    if (nameEl) nameEl.textContent = title
  }
  const chatInput = document.getElementById('chat-input')
  if (chatInput) chatInput.placeholder = `Ask a question about ${title}...`
  // Persist so we can restore when the user comes back to this session
  _saveSessionDoc(sid, id, title, chunkId)
}

function clearDoc() {
  docId = null
  docName = ''
  activeChunkId = null
  const ind = document.getElementById('doc-indicator')
  if (ind) ind.hidden = true
  const chatInput = document.getElementById('chat-input')
  if (chatInput) chatInput.placeholder = 'Ask a question about your documents...'
  if (sid) _saveSessionDoc(sid, null, '', null)
}

// ── Documents ───────────────────────────────────────────

function setupDragDrop() {
  const zone = document.getElementById('drop-zone')
  zone.addEventListener('dragover', e => { e.preventDefault(); zone.classList.add('dragover') })
  zone.addEventListener('dragleave', () => zone.classList.remove('dragover'))
  zone.addEventListener('drop', e => {
    e.preventDefault()
    zone.classList.remove('dragover')
    const file = e.dataTransfer.files[0]
    if (file) uploadFile(file)
  })
}

async function uploadFile(file) {
  if (!file || !file.name.toLowerCase().endsWith('.pdf')) {
    toast('Please upload a PDF file.', 'error')
    return
  }

  loading(true, 'Uploading and processing…')
  const fd = new FormData()
  fd.append('file', file)

  try {
    const res = await fetch(API + '/documents/upload', { method: 'POST', body: fd })
    if (!res.ok) {
      const raw = await res.text().catch(() => res.statusText)
      let detail = raw
      try { detail = JSON.parse(raw).detail || raw } catch { /* plain-text error */ }
      throw new Error(detail)
    }
    const doc = await res.json()
    addDocumentCard(doc)
    // Leave the chunk unset: the quiz picker chooses the most useful section
    // instead of always the first one.
    setDoc(doc.id, doc.title, null)
    const chunkCount = doc.chunk_count ?? doc.chunks?.length ?? 0
    toast(`"${doc.title}" uploaded (${chunkCount} chunks)`, 'success', 3000)
    switchView('chat')
  } catch (e) {
    toast('Upload failed: ' + e.message, 'error')
  } finally {
    loading(false)
  }
}

async function loadDocuments() {
  try {
    const docs = await api('GET', '/documents')
    const list = document.getElementById('documents-list')
    list.innerHTML = ''
    docs.forEach(d => addDocumentCard(d))
  } catch (e) {
    // Server may not be ready
  }
}

function addDocumentCard(doc) {
  const list = document.getElementById('documents-list')
  // Remove empty state if present
  const empty = list.querySelector('.empty-state')
  if (empty) empty.remove()

  // Remove any stale card for this doc (e.g. after re-upload)
  const existing = list.querySelector(`.doc-card[data-doc-id="${doc.id}"]`)
  if (existing) existing.remove()

  const card = document.createElement('div')
  card.className = 'doc-card'
  card.dataset.docId = doc.id
  card.dataset.docTitle = doc.title
  card.dataset.docChunkId = doc.chunks?.[0]?.id || ''

  const isActive = docId === doc.id
  const chunkCount = doc.chunk_count ?? doc.chunks?.length ?? 0
  const chunkLabel = chunkCount === 1 ? '1 chunk' : `${chunkCount} chunks`

  card.innerHTML = `
    <div class="doc-card-info">
      <span class="doc-card-icon">📄</span>
      <div>
        <div class="doc-card-title">${esc(doc.title)}${isActive ? ' <span class="doc-active-badge">active</span>' : ''}</div>
        <div class="doc-card-meta">${chunkLabel} · ${esc(doc.source_filename)}</div>
      </div>
    </div>
    <div class="doc-card-actions">
      <button class="btn-use" title="Set as active document for chat">💬 Chat</button>
      <button class="btn-quiz" title="Set this document for quiz generation and go to Quizzes">📝 Quiz</button>
      <button class="btn-summary" title="Generate an AI summary of this document">📄 Summary</button>
      <button class="btn-danger btn-delete" title="Delete this document">🗑 Delete</button>
    </div>`
  list.appendChild(card)
}

// Event delegation for document card actions
// Uses data attributes instead of inline onclick to avoid XSS vectors
document.getElementById('documents-list').addEventListener('click', e => {
  const card = e.target.closest('.doc-card')
  if (!card) return
  const id = card.dataset.docId
  const title = card.dataset.docTitle
  const chunkId = card.dataset.docChunkId || null
  if (e.target.classList.contains('btn-use')) {
    useDoc(id, title, chunkId)
  } else if (e.target.classList.contains('btn-quiz')) {
    useDocForQuiz(id, title, chunkId)
  } else if (e.target.classList.contains('btn-summary')) {
    summarizeDoc(id, title)
  } else if (e.target.classList.contains('btn-delete')) {
    deleteDoc(id)
  }
})

async function summarizeDoc(id, title) {
  const modal = document.getElementById('summary-modal')
  const titleEl = document.getElementById('summary-title')
  const contentEl = document.getElementById('summary-content')
  titleEl.textContent = `Summary: ${title}`
  contentEl.textContent = 'Generating comprehensive summary from document…'
  modal.hidden = false
  loading(true, `Summarizing "${title}"…`)
  try {
    const res = await api('GET', `/documents/${id}/summary`)
    contentEl.textContent = res.summary || 'No summary generated.'
  } catch (e) {
    contentEl.textContent = `Could not generate summary: ${e.message}`
    toast('Summary failed: ' + e.message, 'error')
  } finally {
    loading(false)
  }
}

function closeSummaryModal() {
  document.getElementById('summary-modal').hidden = true
}

function useDoc(id, title, chunkId) {
  setDoc(id, title, chunkId)
  toast(`"${title}" set as active document`, 'info', 2000)
  switchView('chat')
}

async function useDocForQuiz(id, title, chunkId) {
  setDoc(id, title, null)
  toast(`"${title}" selected for quiz`, 'success', 2000)
  switchView('quizzes')
  await loadQuizChunks(chunkId)
  if (!activeChunkId) {
    toast(`"${title}" has no extractable text chunks — cannot generate a quiz.`, 'error')
    return
  }
  generateQuiz()
}

async function deleteDoc(id) {
  confirm(
    'Delete Document',
    'This will permanently delete this document and all its chunks. This cannot be undone.',
    async () => {
      try {
        await api('DELETE', '/documents/' + id)
        loadDocuments()
        if (docId === id) clearDoc()
        toast('Document deleted', 'info', 2000)
      } catch (e) {
        toast('Delete failed: ' + e.message, 'error')
      }
    },
    'Delete'
  )
}

// ── Quizzes ─────────────────────────────────────────────

// Pick the most useful default section: a heading-bearing, content-rich chunk
// beats a cover page or an almost-empty first chunk.
function pickDefaultChunk(chunks) {
  if (!chunks || !chunks.length) return null
  const score = c => (c.content_length || 0) + (c.heading ? 400 : 0)
  return chunks.reduce((best, c) => (score(c) > score(best) ? c : best)).id
}

async function loadQuizChunks(preferredId) {
  const select = document.getElementById('quiz-chunk-select')
  if (!select) return
  if (!docId) {
    select.innerHTML = '<option value="">No document selected</option>'
    select.disabled = true
    return
  }
  try {
    const data = await api('GET', `/documents/${docId}/chunks`)
    const chunks = data.chunks || []
    if (!chunks.length) {
      select.innerHTML = '<option value="">No chunks in this document</option>'
      select.disabled = true
      activeChunkId = null
      return
    }
    select.disabled = false
    select.innerHTML = chunks.map(c => {
      const heading = c.heading ? `${c.heading} · ` : ''
      const label = `${heading}#${(c.position ?? 0) + 1} · ${c.content_length ?? 0} chars`
      return `<option value="${esc(c.id)}">${esc(label)}</option>`
    }).join('')
    const requested = preferredId || activeChunkId
    const selected = chunks.some(c => c.id === requested) ? requested : pickDefaultChunk(chunks)
    if (selected) {
      select.value = selected
      activeChunkId = selected
    }
  } catch (e) {
    select.innerHTML = '<option value="">Could not load sections</option>'
    select.disabled = true
  }
}

function onQuizChunkChange(value) {
  activeChunkId = value || null
}

async function generateQuiz() {
  if (!docId) {
    toast(
      'No document selected. Go to Documents → click "📝 Quiz" on a document.',
      'warning',
      5000
    )
    switchView('documents')
    return
  }
  const select = document.getElementById('quiz-chunk-select')
  if (select && select.value) activeChunkId = select.value
  if (!activeChunkId) {
    await loadQuizChunks()
    if (select && select.value) activeChunkId = select.value
  }
  if (!activeChunkId) {
    toast(
      'This document has no extractable text chunks — cannot generate a quiz.',
      'error',
      5000
    )
    return
  }
  if (!sid) {
    try {
      const session = await api('POST', '/sessions', {})
      sid = session.id
      document.getElementById('btn-delete-session').hidden = false
      loadSessions()
    } catch (e) {
      toast('Could not create a session: ' + e.message, 'error')
      return
    }
  }

  loading(true, 'Generating quiz from "' + docName + '"…')
  const button = document.getElementById('btn-gen-quiz')
  button.disabled = true
  try {
    await api('POST', '/quizzes/generate', {
      session_id: sid,
      chunk_id: activeChunkId,
      num_questions: 3
    })
    toast('Quiz generated! ✅', 'success', 3000)
    await loadQuizResults()
  } catch (e) {
    toast('Quiz generation failed: ' + e.message, 'error')
  } finally {
    loading(false)
    button.disabled = false
  }
}

function setupQuizInteractions() {
  document.getElementById('quiz-content').addEventListener('click', e => {
    const option = e.target.closest('.quiz-option[data-quiz-id]')
    if (option) answerQuizOption(option)
  })
  document.getElementById('quiz-content').addEventListener('keydown', e => {
    if (e.key !== 'Enter' && e.key !== ' ') return
    const option = e.target.closest('.quiz-option[data-quiz-id]')
    if (!option) return
    e.preventDefault()
    answerQuizOption(option)
  })
}

async function answerQuizOption(option) {
  if (option.dataset.answered === 'true') return
  option.dataset.answered = 'true'
  try {
    await api('POST', '/quizzes/answer', {
      quiz_id: option.dataset.quizId,
      question_index: Number(option.dataset.questionIndex),
      student_answer: option.dataset.optionIndex,
      // Kept for API compatibility; the server validates against stored quiz data.
      correct_index: 0
    })
    toast('Answer saved', 'success', 1500)
    await loadQuizResults()
  } catch (e) {
    delete option.dataset.answered
    toast('Could not save answer: ' + e.message, 'error')
  }
}

async function loadQuizResults() {
  if (!sid) {
    document.getElementById('quiz-content').innerHTML = `
      <div class="empty-state">
        <div class="empty-icon">💬</div>
        <h3>No session active</h3>
        <p>Start a chat session first, then generate quizzes from your documents.</p>
        <button class="btn-primary" onclick="newSession().then(() => loadQuizResults())" style="margin-top:12px">Start Session</button>
      </div>`
    return
  }

  try {
    const results = await api('GET', `/sessions/${sid}/quiz-results`)
    const container = document.getElementById('quiz-content')

    if (!results.length) {
      const hasDoc = !!docId
      container.innerHTML = `
        <div class="empty-state">
          <div class="empty-icon">📝</div>
          <h3>No quizzes yet</h3>
          <p>${hasDoc
            ? `Click <strong>✨ Generate</strong> above to create a quiz from <em>${esc(docName)}</em>.`
            : 'Go to <strong>Documents</strong>, pick a PDF and click <strong>📝 Quiz</strong> to generate your first quiz.'}
          </p>
          <button class="btn-primary" onclick="${hasDoc ? 'generateQuiz()' : "switchView('documents')"}"
                  style="margin-top:12px">
            ${hasDoc ? '✨ Generate Quiz' : '📄 Go to Documents'}
          </button>
        </div>`
      return
    }

    container.innerHTML = ''

    // ── Calculate Session-wide Score & Mastery Metrics ──
    const totalQuizzes = results.length
    const totalQuestions = results.reduce((sum, r) => sum + (r.questions ? r.questions.length : 0), 0)
    const totalAnswered = results.reduce((sum, r) => sum + (r.answers ? r.answers.length : 0), 0)
    const totalCorrect = results.reduce((sum, r) => sum + (r.answers ? r.answers.filter(a => a.is_correct).length : 0), 0)
    const accuracyPct = totalAnswered > 0 ? Math.round(totalCorrect / totalAnswered * 100) : 0
    const masteryPct = totalQuestions > 0 ? Math.round(totalCorrect / totalQuestions * 100) : 0

    // Render Mastery Dashboard
    const dashboard = document.createElement('div')
    dashboard.className = 'score-dashboard'
    const statusText = accuracyPct >= 80 ? '🌟 Mastery Achieved' : accuracyPct >= 50 ? '📈 Good Understanding' : '📚 Learning in Progress'
    const accuracyBadgeClass = accuracyPct >= 80 ? 'badge-high' : accuracyPct >= 50 ? 'badge-mid' : 'badge-low'

    dashboard.innerHTML = `
      <div class="dashboard-header">
        <div>
          <div class="dashboard-title">Quiz Mastery Dashboard</div>
          <div class="dashboard-subtitle">${statusText} · ${totalQuizzes} ${totalQuizzes === 1 ? 'quiz' : 'quizzes'} generated</div>
        </div>
        <div class="dashboard-accuracy-badge ${accuracyBadgeClass}">
          ${accuracyPct}% Accuracy
        </div>
      </div>
      <div class="dashboard-grid">
        <div class="dash-stat">
          <div class="dash-stat-num">${totalQuizzes}</div>
          <div class="dash-stat-label">Quizzes</div>
        </div>
        <div class="dash-stat">
          <div class="dash-stat-num">${totalCorrect} / ${totalQuestions}</div>
          <div class="dash-stat-label">Correct Answers</div>
        </div>
        <div class="dash-stat">
          <div class="dash-stat-num">${totalAnswered} / ${totalQuestions}</div>
          <div class="dash-stat-label">Questions Completed</div>
        </div>
        <div class="dash-stat">
          <div class="dash-stat-num">${masteryPct}%</div>
          <div class="dash-stat-label">Total Mastery</div>
        </div>
      </div>
      <div class="mastery-bar-wrapper">
        <div class="mastery-progress-bar">
          <div class="mastery-progress-fill" style="width: ${masteryPct}%"></div>
        </div>
      </div>`
    container.appendChild(dashboard)

    results.forEach((r, idx) => {
      const card = document.createElement('div')
      card.className = 'quiz-card'

      const totalQ = r.questions.length
      const answeredQ = r.answers.length
      const correctQ = r.answers.filter(a => a.is_correct).length
      const pct = totalQ > 0 ? Math.round(correctQ / totalQ * 100) : 0
      const date = new Date(r.created_at * 1000).toLocaleString()
      const scoreClass = pct >= 80 ? 'score-high' : pct >= 50 ? 'score-mid' : 'score-low'
      const quizProgressPct = totalQ > 0 ? Math.round(answeredQ / totalQ * 100) : 0

      card.innerHTML = `
        <div class="quiz-header">
          <div>
            <h4>Quiz #${idx + 1}</h4>
            <div class="quiz-meta-date">${date}</div>
          </div>
          <span class="quiz-score ${scoreClass}">
            ${answeredQ < totalQ
              ? `${answeredQ}/${totalQ} answered`
              : `${correctQ}/${totalQ} correct (${pct}%)`
            }
          </span>
        </div>
        <div class="quiz-mini-bar">
          <div class="quiz-mini-fill" style="width:${quizProgressPct}%"></div>
        </div>
        ${r.questions.map((q, qi) => {
          const answer = r.answers.find(a => a.question_index === qi)
          return `<div class="quiz-question">
            <div class="q-text"><span class="q-num">${qi + 1}.</span> ${esc(q.question)}</div>
            <div class="quiz-options">
            ${q.options.map((opt, oi) => {
              const letter = ['A', 'B', 'C', 'D'][oi] || String(oi + 1)
              let cls = 'quiz-option'
              let icon = ''
              if (answer) {
                if (oi === q.correct_index) { cls += ' correct'; icon = '✅ ' }
                else if (oi === parseInt(answer.student_answer) && !answer.is_correct) { cls += ' wrong'; icon = '❌ ' }
              }
              const answerAttrs = answer
                ? 'data-answered="true"'
                : `data-quiz-id="${esc(r.quiz_id)}" data-question-index="${qi}" data-option-index="${oi}" role="button" tabindex="0" aria-label="Option ${letter}: ${esc(opt)}"`
              return `<div class="${cls}" ${answerAttrs}><span class="opt-badge">${letter}</span><span class="opt-text">${icon}${esc(opt)}</span></div>`
            }).join('')}
            </div>
            ${answer && q.explanation ? `<div class="q-explanation">💡 <strong>Explanation:</strong> ${esc(q.explanation)}</div>` : ''}
          </div>`
        }).join('')}`
      container.appendChild(card)
    })
  } catch (e) {
    document.getElementById('quiz-content').innerHTML =
      '<div class="empty-state"><div class="empty-icon">⚠️</div><h3>Could not load quizzes</h3><p>' + esc(e.message) + '</p></div>'
  }
}

// ── Settings ─────────────────────────────────────────────

async function loadSettings() {
  try {
    const s = v => document.getElementById(v)
    for (const [key, elId, fallback] of [
      ['engine', 'engine-select', 'local'],
      ['temperature', 'temp-range', '0.7'],
      ['context_size', 'ctx-size', '2048'],
      ['openai_api_key', 'openai-key-input', '']
    ]) {
      try {
        const el = s(elId)
        if (!el) continue
        const r = await api('GET', '/settings/' + key)
        el.value = r.value
      } catch {
        const el = s(elId)
        if (el) el.value = fallback
      }
    }
    const engineVal = s('engine-select').value
    document.getElementById('temp-val').textContent = s('temp-range').value
    document.getElementById('engine-badge').textContent = engineVal

    const keyGroup = document.getElementById('openai-key-group')
    if (keyGroup) keyGroup.style.display = engineVal === 'online' ? 'block' : 'none'

    // Load available models and update UI
    await loadModelChoices()
    loadModelStatus()
    loadEmbeddingStatus()
  } catch (e) {
    // Settings endpoint might not be available
  }
}

// ── Embedding Model (semantic search) ────────────────────

let embeddingPollTimer = null

function renderEmbeddingStatus(result) {
  const button = document.getElementById('embed-download-btn')
  const status = document.getElementById('embed-status')
  if (!button || !status) return
  const stopPolling = () => {
    if (embeddingPollTimer) { clearInterval(embeddingPollTimer); embeddingPollTimer = null }
  }
  if (result.state === 'ready' && result.available) {
    button.textContent = 'Installed ✓'
    button.disabled = true
    status.textContent = 'Semantic retrieval active'
    stopPolling()
  } else if (result.state === 'downloading') {
    button.textContent = 'Downloading…'
    button.disabled = true
    status.textContent = 'Downloading embedding model…'
    if (!embeddingPollTimer) {
      embeddingPollTimer = setInterval(() => loadEmbeddingStatus(false), 1500)
    }
  } else if (result.state === 'error') {
    button.textContent = 'Retry install'
    button.disabled = false
    status.textContent = result.error || 'Installation failed'
    stopPolling()
  } else {
    button.textContent = 'Install embeddings'
    button.disabled = false
    status.textContent = 'Not installed — keyword search only'
  }
}

async function loadEmbeddingStatus(probe = false) {
  const status = document.getElementById('embed-status')
  const button = document.getElementById('embed-download-btn')
  if (!status || !button) return
  try {
    const result = await api(
      'GET',
      `/models/embeddings/status${probe ? '?probe=true' : ''}`
    )
    renderEmbeddingStatus(result)
  } catch {
    status.textContent = 'Embedding installer unavailable'
  }
}

async function downloadEmbeddingModel() {
  const button = document.getElementById('embed-download-btn')
  if (!button) return
  button.disabled = true
  try {
    await api('POST', '/models/embeddings/download', {})
    toast('Embedding download started', 'info', 2500)
    await loadEmbeddingStatus(false)
    if (!embeddingPollTimer) {
      embeddingPollTimer = setInterval(() => loadEmbeddingStatus(false), 1500)
    }
  } catch (e) {
    button.disabled = false
    toast('Could not start embedding download: ' + e.message, 'error')
  }
}

// ── Model Selection ─────────────────────────────────────

let _pendingModelId = null

function updateChatModelBadge(name, modelId) {
  const badge = document.getElementById('chat-model-name')
  if (!badge) return
  const shortNames = { 'gemma-3-1b': '1B', 'gemma-3-270m': '270M' }
  badge.textContent = shortNames[modelId] || name || modelId || '…'
  badge.parentElement.title = `Model: ${name || modelId || 'Unknown'} — Click to switch`
}

async function loadModelChoices() {
  let activeId = 'gemma-3-1b'  // default fallback
  let models = []

  try {
    const data = await api('GET', '/models/available')
    activeId = data.active_model_id
    models = data.models || []

    const sel = document.getElementById('model-select')
    if (sel) {
      sel.innerHTML = ''
      models.forEach(m => {
        const opt = document.createElement('option')
        opt.value = m.id
        opt.textContent = `${m.name} — ${m.description}`
        if (m.active) opt.selected = true
        sel.appendChild(opt)
      })
      sel.dataset.activeModel = activeId
    }
  } catch (e) {
    // Server may not be ready yet — use defaults
    const sel = document.getElementById('model-select')
    if (sel) {
      sel.dataset.activeModel = activeId
    }
  }

  // Update chat badge
  const activeModel = models.find(m => m.id === activeId)
  const name = activeModel ? activeModel.name : (activeId === 'gemma-3-1b' ? 'Gemma 3 1B' : 'Gemma 3 270M')
  updateChatModelBadge(name, activeId)

  // Show/hide startup modal
  const hasModelBeenChosen = localStorage.getItem('nova_model_chosen')
  const startupModal = document.getElementById('startup-model-modal')
  if (startupModal) {
    if (!hasModelBeenChosen) {
      // Highlight the active model in startup modal
      document.querySelectorAll('.model-card').forEach(c => {
        c.classList.toggle('selected', c.dataset.model === activeId)
        c.querySelector('.model-card-check').textContent = c.dataset.model === activeId ? '✓' : ''
      })
      document.getElementById('startup-model-btn').textContent = `Start with ${activeModel?.name || name || activeId}`
    } else {
      startupModal.hidden = true
    }
  }
}

function selectStartupModel(modelId) {
  document.querySelectorAll('.model-card').forEach(c => {
    c.classList.toggle('selected', c.dataset.model === modelId)
    c.querySelector('.model-card-check').textContent = c.dataset.model === modelId ? '✓' : ''
  })
  const names = { 'gemma-3-1b': 'Gemma 3 1B', 'gemma-3-270m': 'Gemma 3 270M' }
  document.getElementById('startup-model-btn').textContent = `Start with ${names[modelId] || modelId}`
  document.getElementById('startup-model-btn').dataset.modelId = modelId
}

async function confirmStartupModel() {
  const btn = document.getElementById('startup-model-btn')
  const modelId = btn.dataset.modelId || 'gemma-3-1b'
  try {
    await api('PUT', '/settings/active_model', { value: modelId })
    localStorage.setItem('nova_model_chosen', 'true')
    document.getElementById('startup-model-modal').hidden = true
    const name = modelId === 'gemma-3-1b' ? 'Gemma 3 1B' : 'Gemma 3 270M'
    updateChatModelBadge(name, modelId)
    toast('Using ' + name, 'success', 2000)
    loadSettings()
  } catch (e) {
    // Even if the API fails, dismiss modal and use local default
    localStorage.setItem('nova_model_chosen', 'true')
    document.getElementById('startup-model-modal').hidden = true
    toast('Using default model', 'info', 2000)
    loadSettings()
  }
}

function skipStartupModel() {
  // Dismiss startup modal without any API call — use defaults
  localStorage.setItem('nova_model_chosen', 'true')
  document.getElementById('startup-model-modal').hidden = true
  toast('Using default model (Gemma 3 1B)', 'info', 2000)
  loadSettings()
}

function openModelSwitchFromChat() {
  // Open the model switch popup from the chat header badge
  const sel = document.getElementById('model-select')
  if (!sel) return
  const currentValue = sel.value
  const names = {
    'gemma-3-1b': 'Gemma 3 1B IT',
    'gemma-3-270m': 'Gemma 3 270M IT'
  }
  // Show a quick model picker in the chat header
  const badge = document.getElementById('chat-model-name')
  if (!badge) return

  // Toggle a small inline model switcher below the badge
  const existing = document.querySelector('.chat-model-picker')
  if (existing) {
    existing.remove()
    return
  }

  const picker = document.createElement('div')
  picker.className = 'chat-model-picker'
  picker.innerHTML = `
    <div class="chat-model-picker-item${currentValue === 'gemma-3-1b' ? ' active' : ''}" onclick="quickSwitchModel('gemma-3-1b')">
      <span class="cmp-icon">🧠</span>
      <div>
        <div class="cmp-name">Gemma 3 1B</div>
        <div class="cmp-desc">Best quality</div>
      </div>
      ${currentValue === 'gemma-3-1b' ? '<span class="cmp-check">✓</span>' : ''}
    </div>
    <div class="chat-model-picker-item${currentValue === 'gemma-3-270m' ? ' active' : ''}" onclick="quickSwitchModel('gemma-3-270m')">
      <span class="cmp-icon">⚡</span>
      <div>
        <div class="cmp-name">Gemma 3 270M</div>
        <div class="cmp-desc">Fast & lightweight</div>
      </div>
      ${currentValue === 'gemma-3-270m' ? '<span class="cmp-check">✓</span>' : ''}
    </div>
  `
  badge.parentElement.parentElement.appendChild(picker)
}

function quickSwitchModel(modelId) {
  // Show confirmation popup when switching from chat
  const sel = document.getElementById('model-select')
  const currentValue = sel?.dataset.activeModel || 'gemma-3-1b'
  if (currentValue === modelId) {
    // Already active, close picker
    document.querySelector('.chat-model-picker')?.remove()
    return
  }

  const names = { 'gemma-3-1b': 'Gemma 3 1B IT', 'gemma-3-270m': 'Gemma 3 270M IT' }
  document.getElementById('model-switch-title').textContent = `Switch to ${names[modelId] || modelId}?`
  document.getElementById('model-switch-message').innerHTML = `
    Changing the model will affect your current conversation.<br><br>
    <strong>${names[modelId] || modelId}</strong> needs to be loaded into memory, replacing the current model.
    Your chat history will be preserved, but the new model won't have the context of previous messages.
  `
  _pendingModelId = modelId
  document.querySelector('.chat-model-picker')?.remove()
  document.getElementById('model-switch-modal').hidden = false
}

function onModelSelectChange(value) {
  // Show confirmation popup when user changes the model in settings
  const sel = document.getElementById('model-select')
  const prevActive = sel.dataset.activeModel
  if (prevActive === value) return
  if (!prevActive) return

  const names = { 'gemma-3-1b': 'Gemma 3 1B IT', 'gemma-3-270m': 'Gemma 3 270M IT' }
  document.getElementById('model-switch-title').textContent = `Switch to ${names[value] || value}?`
  document.getElementById('model-switch-message').innerHTML = `
    Changing the model will affect your current conversation.<br><br>
    <strong>${names[value] || value}</strong> needs to be loaded into memory, which replaces the current model.
    Your chat history will be preserved, but the new model won't have the context of previous messages.
  `
  _pendingModelId = value
  document.getElementById('model-switch-modal').hidden = false
}

function closeModelSwitch() {
  document.getElementById('model-switch-modal').hidden = true
  _pendingModelId = null
  loadSettings()
}

async function confirmModelSwitch() {
  if (!_pendingModelId) return
  document.getElementById('model-switch-modal').hidden = true
  const modelId = _pendingModelId
  _pendingModelId = null

  loading(true, 'Switching model…')
  try {
    await api('PUT', '/settings/active_model', { value: modelId })
    const names = { 'gemma-3-1b': 'Gemma 3 1B', 'gemma-3-270m': 'Gemma 3 270M' }
    updateChatModelBadge(names[modelId] || modelId, modelId)
    toast('Switched to ' + (names[modelId] || modelId), 'success', 2500)
    await loadSettings()
  } catch (e) {
    toast('Failed to switch model: ' + e.message, 'error')
    await loadSettings()
  } finally {
    loading(false)
  }
}

let modelPollTimer = null

function formatModelBytes(value) {
  if (!value) return '0 MB'
  return (value / (1024 * 1024)).toFixed(0) + ' MB'
}

async function loadModelStatus() {
  const button = document.getElementById('model-download-btn')
  const status = document.getElementById('model-download-status')
  const installStatus = document.getElementById('model-install-status')
  if (!button || !status) return
  try {
    const result = await api('GET', '/models/status')
    const filename = result.filename || ''
    if (result.state === 'ready' && result.available) {
      button.textContent = 'Installed ✓'
      button.disabled = true
      status.textContent = `${filename} is ready`
      if (installStatus) installStatus.textContent = '✅ Model ready'
    } else if (result.state === 'downloading') {
      button.textContent = 'Downloading...'
      button.disabled = true
      const progress = result.total_bytes
        ? `${result.progress}% (${formatModelBytes(result.downloaded_bytes)} / ${formatModelBytes(result.total_bytes)})`
        : `Downloading (${formatModelBytes(result.downloaded_bytes)})`
      status.textContent = progress
      if (installStatus) installStatus.textContent = '⏳ ' + progress
      if (!modelPollTimer) modelPollTimer = setInterval(loadModelStatus, 1000)
    } else if (result.state === 'error') {
      button.textContent = 'Retry install'
      button.disabled = false
      status.textContent = result.error || 'Installation failed'
      if (installStatus) installStatus.textContent = '❌ ' + (result.error || 'Failed')
      if (modelPollTimer) { clearInterval(modelPollTimer); modelPollTimer = null }
    } else {
      button.textContent = 'Install model'
      button.disabled = false
      status.textContent = `${filename || 'Model'} is not installed yet`
      if (installStatus) installStatus.textContent = '⬇️ Needs download'
    }
  } catch (e) {
    status.textContent = 'Model installer unavailable'
    if (installStatus) installStatus.textContent = ''
  }
}

async function downloadModel() {
  const button = document.getElementById('model-download-btn')
  if (!button) return
  button.disabled = true
  try {
    await api('POST', '/models/download', {})
    await loadModelStatus()
    toast('Model download started', 'info', 2500)
  } catch (e) {
    button.disabled = false
    toast(e.message, 'error')
  }
}

const saveCache = {}
async function saveSetting(key, value) {
  // Debounce rapid saves from range input
  const now = Date.now()
  if (saveCache[key] && now - saveCache[key] < 100) return
  saveCache[key] = now

  try {
    await api('PUT', '/settings/' + key, { value: String(value) })
    if (key === 'engine') {
      document.getElementById('engine-badge').textContent = value
      const keyGroup = document.getElementById('openai-key-group')
      if (keyGroup) keyGroup.style.display = value === 'online' ? 'block' : 'none'
      toast('Engine switched to: ' + value, 'info', 2000)
    } else if (key === 'openai_api_key') {
      toast('OpenAI API key saved', 'success', 2000)
    } else if (key === 'temperature') {
      toast('Temperature: ' + value, 'info', 1500)
    } else if (key === 'context_size') {
      toast('Context size: ' + value + ' tokens', 'info', 1500)
    }
  } catch (e) {
    toast('Failed to save setting: ' + e.message, 'error')
  }
}

function saveSettingDelayed(key, value) {
  clearTimeout(settingsTimer)
  settingsTimer = setTimeout(() => saveSetting(key, value), 500)
}
