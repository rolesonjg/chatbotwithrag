const $ = (selector) => document.querySelector(selector);
const documentList = $('#documentList');
const answerArea = $('#answerArea');
const welcome = $('#welcome');
let documents = [];
let toastTimer;

function toast(message, isError = false) {
  const el = $('#toast');
  el.textContent = message;
  el.className = `toast show${isError ? ' error' : ''}`;
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => { el.className = 'toast'; }, 3500);
}

async function api(url, options = {}) {
  const response = await fetch(url, options);
  let data;
  try { data = await response.json(); } catch { data = {}; }
  if (!response.ok) throw new Error(data.detail || data.message || `Request failed (${response.status})`);
  return data;
}

function esc(value) {
  return String(value ?? '').replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
}

function statusClass(status) {
  return ({'Uploaded':'uploaded','Ingested':'ingested','Failed':'failed','Ingesting':'ingesting'})[status] || 'uploaded';
}

function selectedNames() {
  return [...document.querySelectorAll('.doc-checkbox:checked')].map(input => input.dataset.name);
}

function updateSelectionSummary() {
  const selected = selectedNames();
  $('#selectionSummary').textContent = selected.length ? `${selected.length} document${selected.length === 1 ? '' : 's'} selected` : 'All ingested documents';
}

function renderDocuments() {
  $('#docCount').textContent = `${documents.length} file${documents.length === 1 ? '' : 's'}`;
  const totalIndexed = documents.filter(d => d.status === 'Ingested').length;
  if (!documents.length) {
    documentList.innerHTML = '<div class="empty-state">No PDFs yet. Upload a document to get started.</div>';
  } else {
    documentList.innerHTML = documents.map(doc => {
      const ingested = doc.status === 'Ingested';
      const size = doc.size < 1024 * 1024 ? `${Math.max(1, Math.round(doc.size / 1024))} KB` : `${(doc.size / (1024 * 1024)).toFixed(1)} MB`;
      return `<article class="document-card">
        <div class="doc-top">
          <input class="doc-checkbox" type="checkbox" data-name="${esc(doc.name)}" aria-label="Search ${esc(doc.name)}" ${ingested ? '' : 'disabled'} title="Only ingested documents can be selected">
          <div class="doc-info"><div class="doc-name">${esc(doc.name)}</div><div class="doc-meta">${doc.pages ?? '—'} pages · ${size}</div></div>
          <span class="status status-${statusClass(doc.status)}">${esc(doc.status)}</span>
        </div>
        <div class="doc-actions">
          ${!ingested ? `<button class="mini-button ingest" data-action="ingest" data-name="${esc(doc.name)}">${doc.status === 'Ingesting' ? 'Ingesting…' : 'Ingest'}</button>` : '<span class="mini-button" style="cursor:default">Indexed</span>'}
          <button class="mini-button" data-action="rename" data-name="${esc(doc.name)}">Rename</button>
          <button class="mini-button delete" data-action="delete" data-name="${esc(doc.name)}">Delete</button>
        </div>
      </article>`;
    }).join('');
  }
  document.querySelectorAll('.doc-checkbox').forEach(input => input.addEventListener('change', updateSelectionSummary));
  updateSelectionSummary();
}

async function loadDocuments() {
  try {
    const data = await api('/api/documents');
    documents = data.documents || [];
    $('#chunkCount').textContent = `${data.indexed_chunks || 0} indexed chunks`;
    renderDocuments();
  } catch (error) {
    documentList.innerHTML = `<div class="empty-state">Could not load documents: ${esc(error.message)}</div>`;
    toast(error.message, true);
  }
}

async function checkHealth() {
  const badge = $('#ollamaStatus');
  try {
    const health = await api('/api/health');
    badge.className = `service-status ${health.ollama ? 'online' : 'offline'}`;
    badge.innerHTML = `<span class="dot"></span><span>${health.ollama ? `Ollama connected · ${esc(health.model)}` : 'Ollama unavailable'}</span>`;
  } catch {
    badge.className = 'service-status offline';
    badge.innerHTML = '<span class="dot"></span><span>Backend unavailable</span>';
  }
}

$('#uploadForm').addEventListener('submit', async event => {
  event.preventDefault();
  const input = $('#pdfFile');
  if (!input.files.length) return;
  const button = event.currentTarget.querySelector('button');
  button.disabled = true; button.textContent = 'Uploading…';
  try {
    const form = new FormData(); form.append('file', input.files[0]);
    const data = await api('/api/documents', {method:'POST', body:form});
    toast(data.message || 'PDF uploaded.'); input.value = '';
    await loadDocuments();
  } catch (error) { toast(error.message, true); }
  finally { button.disabled = false; button.textContent = 'Upload'; }
});

documentList.addEventListener('click', async event => {
  const button = event.target.closest('button[data-action]');
  if (!button) return;
  const name = button.dataset.name;
  const action = button.dataset.action;
  if (action === 'ingest') {
    button.disabled = true; button.textContent = 'Ingesting…';
    toast(`Ingesting ${name}. This may take a while…`);
    try {
      const data = await api(`/api/documents/${encodeURIComponent(name)}/ingest`, {method:'POST'});
      toast(`Ingested ${name}: ${data.chunks} chunks created.`);
      await loadDocuments();
    } catch (error) {
      toast(error.message, true); await loadDocuments();
    }
  }
  if (action === 'delete') {
    if (!confirm(`Delete “${name}” and remove its indexed chunks? This cannot be undone.`)) return;
    button.disabled = true;
    try { const data = await api(`/api/documents/${encodeURIComponent(name)}`, {method:'DELETE'}); toast(data.message); await loadDocuments(); }
    catch (error) { toast(error.message, true); }
  }
  if (action === 'rename') {
    const suggested = name.replace(/\.pdf$/i, '') + '-renamed.pdf';
    const newName = prompt('Enter the new filename (must end in .pdf):', suggested);
    if (!newName || newName.trim() === name) return;
    try {
      const data = await api(`/api/documents/${encodeURIComponent(name)}`, {method:'PATCH', headers:{'Content-Type':'application/json'}, body:JSON.stringify({new_name:newName.trim()})});
      toast(data.message); await loadDocuments();
    } catch (error) { toast(error.message, true); }
  }
});

$('#refreshButton').addEventListener('click', loadDocuments);
$('#selectAllButton').addEventListener('click', () => {
  const inputs = [...document.querySelectorAll('.doc-checkbox:not(:disabled)')];
  const allChecked = inputs.length > 0 && inputs.every(input => input.checked);
  inputs.forEach(input => { input.checked = !allChecked; });
  updateSelectionSummary();
});

$('#resetButton').addEventListener('click', async () => {
  if (!confirm('Reset the entire vector database and extracted index? Uploaded PDF files will remain, but you will need to ingest them again.')) return;
  const button = $('#resetButton'); button.disabled = true; button.textContent = 'Resetting…';
  try { const data = await api('/api/reset', {method:'POST'}); toast(data.message); answerArea.innerHTML = ''; welcome.hidden = false; await loadDocuments(); }
  catch (error) { toast(error.message, true); }
  finally { button.disabled = false; button.textContent = 'Reset database'; }
});

$('#questionForm').addEventListener('submit', async event => {
  event.preventDefault();
  const question = $('#questionInput').value.trim();
  if (!question) return;
  const chosen = selectedNames();
  const indexedAvailable = documents.some(doc => doc.status === 'Ingested');
  if (!indexedAvailable) { toast('Upload and ingest at least one PDF first.', true); return; }
  const button = $('#askButton'); button.disabled = true;
  welcome.hidden = true;
  answerArea.innerHTML = '<div class="thinking"><span class="spinner"></span><span>Searching documents and asking the local model…</span></div>';
  try {
    const result = await api('/api/ask', {method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify({question, document_names:chosen})});
    renderAnswer(result);
    $('#questionInput').value = '';
  } catch (error) {
    answerArea.innerHTML = `<div class="warning">${esc(error.message)}<br><br>Check that Ollama is running and that your configured model is installed.</div>`;
  } finally { button.disabled = false; }
});

function renderAnswer(result) {
  const points = Array.isArray(result.key_points) && result.key_points.length
    ? `<ul class="key-points">${result.key_points.map(point => `<li>${esc(point)}</li>`).join('')}</ul>` : '';
  const sources = Array.isArray(result.sources) && result.sources.length
    ? `<div class="sources"><h4>Sources · ${result.sources.length}</h4>${result.sources.map(source => {
      const start = source.page_start ?? '?'; const end = source.page_end ?? start;
      const pageText = start === end ? `Page ${start}` : `Pages ${start}–${end}`;
      return `<div class="source-item"><span class="source-page">${esc(pageText)}</span><div><div class="source-name">${esc(source.document)}</div><div class="source-section">${esc(source.section || 'Document passage')}</div></div></div>`;
    }).join('')}</div>` : '';
  const warning = result.context_sufficient === false
    ? `<div class="warning">Not enough evidence was retrieved to answer confidently. Try selecting another ingested document or asking a more specific question.</div>` : '';
  answerArea.innerHTML = `<article class="answer-card"><div class="answer-label">Answer</div><div class="answer-text">${esc(result.answer || 'No answer returned.')}</div>${points}${sources}</article>${warning}`;
}

document.querySelectorAll('[data-question]').forEach(button => button.addEventListener('click', () => {
  $('#questionInput').value = button.dataset.question;
  $('#questionInput').focus();
}));

loadDocuments();
checkHealth();
setInterval(checkHealth, 15000);
