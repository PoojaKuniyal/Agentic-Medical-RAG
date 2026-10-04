/**
 * MediAI — Clinical Evidence Synthesis Platform Frontend Client
 * 
 * Interacts purely with the FastAPI REST Endpoints:
 *  - GET  /health
 *  - GET  /collections
 *  - POST /chat
 *  - POST /ingest
 */

const API_BASE_URL = window.location.origin;


// State management
let currentSessionId = 'session-' + Date.now();
let sessions = [
  { id: currentSessionId, title: 'New Clinical Query', messages: [], inspectorData: null }
];

// DOM Elements
const chatFeed = document.getElementById('chat-feed');
const chatForm = document.getElementById('chat-form');
const userInput = document.getElementById('user-input');
const sendBtn = document.getElementById('send-btn');
const newChatBtn = document.getElementById('new-chat-btn');
const sessionList = document.getElementById('session-list');
const collectionsList = document.getElementById('collections-list');
const backendStatusDot = document.getElementById('backend-status-dot');
const backendStatusText = document.getElementById('backend-status-text');
const reingestBtn = document.getElementById('reingest-btn');
const timeline = document.getElementById('execution-timeline');
const inspectorStatus = document.getElementById('inspector-status');
const evidenceSupportDisplay = document.getElementById('evidence-support-display');
const followupList = document.getElementById('followup-list');
const citationsList = document.getElementById('citations-list');
const activeSessionTitle = document.getElementById('active-session-title');


// Initialize App
document.addEventListener('DOMContentLoaded', () => {
  checkBackendHealth();
  fetchCollections();
  renderSessionList();
  renderCurrentSession();

  // Poll backend health every 30 seconds
  setInterval(checkBackendHealth, 30000);

  // Form Submission
  chatForm.addEventListener('submit', handleFormSubmit);

  // Handle textarea enter submit (shift+enter for new line)
  userInput.addEventListener('keydown', (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      chatForm.requestSubmit();
    }
  });

  newChatBtn.addEventListener('click', createNewSession);
  reingestBtn.addEventListener('click', triggerReingest);

  // Sample Query Buttons via Event Delegation on chatFeed
  chatFeed.addEventListener('click', (e) => {
    const btn = e.target.closest('.sample-query-btn');
    if (btn) {
      userInput.value = btn.textContent.trim();
      chatForm.requestSubmit();
    }
  });
});

// ── Backend Communications ──────────────────────────────────────────────────

async function checkBackendHealth() {
  try {
    const res = await fetch(`${API_BASE_URL}/health`);
    if (res.ok) {
      const data = await res.json();
      backendStatusDot.className = 'status-dot dot-online';
      backendStatusText.textContent = `Online (${data.status})`;
    } else {
      throw new Error('Unhealthy');
    }
  } catch (err) {
    backendStatusDot.className = 'status-dot dot-offline';
    backendStatusText.textContent = 'Backend Offline';
  }
}

async function fetchCollections() {
  try {
    const res = await fetch(`${API_BASE_URL}/collections`);
    if (res.ok) {
      const data = await res.json();
      renderCollections(data.collections || []);
    }
  } catch (err) {
    collectionsList.innerHTML = '<div class="collection-item">Failed to load collections</div>';
  }
}

async function triggerReingest() {
  reingestBtn.disabled = true;
  reingestBtn.innerHTML = '<span class="spinner"></span> Syncing...';
  try {
    const res = await fetch(`${API_BASE_URL}/ingest`, { method: 'POST' });
    if (res.ok) {
      const data = await res.json();
      alert(`Knowledge ingestion complete: ${data.indexed} indexed, ${data.skipped} skipped.`);
      fetchCollections();
    } else {
      alert('Ingestion failed.');
    }
  } catch (err) {
    alert('Ingestion error: ' + err.message);
  } finally {
    reingestBtn.disabled = false;
    reingestBtn.innerHTML = `
      <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
        <path d="M21.5 2v6h-6M21.34 15.57a10 10 0 1 1-.57-8.38l5.67-5.67"/>
      </svg> Sync Knowledge`;
  }
}

function typewriteText(element, fullText, speedMs = 14, onComplete = null) {
  element.classList.add('streaming');
  const words = fullText.split(' ');
  let currentWordIdx = 0;
  element.textContent = '';

  const interval = setInterval(() => {
    if (currentWordIdx < words.length) {
      element.textContent += (currentWordIdx > 0 ? ' ' : '') + words[currentWordIdx];
      currentWordIdx++;
      chatFeed.scrollTop = chatFeed.scrollHeight;
    } else {
      clearInterval(interval);
      element.classList.remove('streaming');
      if (onComplete) onComplete();
    }
  }, speedMs);
}

async function handleFormSubmit(e) {
  e.preventDefault();
  const query = userInput.value.trim();
  if (!query) return;

  const session = sessions.find(s => s.id === currentSessionId);
  if (!session) return;

  // Clear input
  userInput.value = '';
  userInput.style.height = 'auto';

  // Update session title if default
  if (session.title === 'New Clinical Query') {
    session.title = query.length > 35 ? query.substring(0, 35) + '...' : query;
    renderSessionList();
    if (activeSessionTitle) activeSessionTitle.textContent = session.title;
  }

  // Remove welcome card if present
  const welcomeCard = chatFeed.querySelector('.welcome-card');
  if (welcomeCard) welcomeCard.remove();

  // Save & Render User Message
  session.messages.push({ sender: 'user', text: query });
  appendMessage('user', query);

  // Show Timeline & Disable Send
  sendBtn.disabled = true;
  showExecutionTimeline();

  try {
    // Send REST API Request to /chat
    const res = await fetch(`${API_BASE_URL}/chat`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        query: query,
        session_id: currentSessionId
      })
    });

    if (!res.ok) {
      let detailMsg = `HTTP error ${res.status}`;
      try {
        const errorJson = await res.json();
        if (errorJson.detail) {
          detailMsg = typeof errorJson.detail === 'string' ? errorJson.detail : JSON.stringify(errorJson.detail);
        }
      } catch (e) { }
      throw new Error(detailMsg);
    }

    const data = await res.json();
    hideExecutionTimeline();

    if (!data.is_safe && data.safety_response) {
      // Query blocked by guardrail
      const msgText = `⚠️ **Query Blocked by Guardrail**\n\nReason: ${data.safety_response.reason}\n${data.safety_response.message}`;
      session.messages.push({
        sender: 'assistant',
        text: msgText,
        cached: data.cached,
        similarity: data.similarity_score
      });
      session.inspectorData = null;
      if (currentSessionId === session.id) {
        appendMessage('assistant', msgText, { cached: data.cached, similarity: data.similarity_score }, true);
        resetInspector();
      }
    } else if (data.response) {
      // Successful clinical summary response
      const resp = data.response;
      session.messages.push({
        sender: 'assistant',
        text: resp.summary,
        inspectorData: resp,
        cached: data.cached,
        similarity: data.similarity_score
      });
      session.inspectorData = resp;
      if (currentSessionId === session.id) {
        appendMessage('assistant', resp.summary, { cached: data.cached, similarity: data.similarity_score }, true, () => {
          updateInspector(resp);
        });
        updateInspector(resp);
      }
    } else {
      const msgText = 'Unable to generate synthesis for this query.';
      session.messages.push({ sender: 'assistant', text: msgText });
      if (currentSessionId === session.id) {
        appendMessage('assistant', msgText);
      }
    }
  } catch (err) {
    console.error('Chat error:', err);
    hideExecutionTimeline();
    const errText = `⚠️ **MediAI Service Error**: ${err.message}`;
    session.messages.push({ sender: 'assistant', text: errText });
    if (currentSessionId === session.id) {
      appendMessage('assistant', errText);
    }
  } finally {
    hideExecutionTimeline();
    sendBtn.disabled = false;
  }
}

// ── UI Rendering Helpers ────────────────────────────────────────────────────

function renderCurrentSession() {
  const session = sessions.find(s => s.id === currentSessionId);
  if (!session) return;

  if (activeSessionTitle) {
    activeSessionTitle.textContent = session.title;
  }

  chatFeed.innerHTML = '';

  if (!session.messages || session.messages.length === 0) {
    chatFeed.innerHTML = `
      <div class="welcome-card">
        <div class="welcome-icon">
          <svg width="40" height="40" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5">
            <path d="M19 14c1.49-1.46 3-3.21 3-5.5A5.5 5.5 0 0 0 16.5 3c-1.76 0-3 .5-4.5 2-1.5-1.5-2.74-2-4.5-2A5.5 5.5 0 0 0 2 8.5c0 2.3 1.5 4.05 3 5.5l7 7Z"/>
            <path d="M12 5v14M5 12h14"/>
          </svg>
        </div>
        <h3>Clinical Evidence Assistant</h3>
        <p>Ask evidence-based clinical questions about guidelines, treatment algorithms, and biomedical literature. Answers are synthesized directly from verified sources.</p>
        <div class="sample-queries">
          <button class="sample-query-btn">Which adults with type 2 diabetes should be offered continuous glucose monitoring (CGM) according to Quality Statement 3?</button>
          <button class="sample-query-btn">What are the three main benefits of SGLT-2 inhibitors for adults with type 2 diabetes?</button>
          <button class="sample-query-btn">What conditions must be met for finerenone to be recommended for adults with stage 3 or 4 chronic kidney disease associated with type 2 diabetes?</button>
        </div>
      </div>
    `;
    resetInspector();
  } else {
    session.messages.forEach(msg => {
      appendMessage(msg.sender, msg.text, { cached: msg.cached, similarity: msg.similarity }, false);
    });

    if (session.inspectorData) {
      updateInspector(session.inspectorData);
    } else {
      resetInspector();
    }
  }
}

function stripMarkdownForUI(text) {
  if (!text) return '';
  // Strip bold (**text** or __text__)
  let cleaned = text.replace(/(\*\*|__)(.*?)\1/g, '$2');
  // Strip italics (*text* or _text_)
  cleaned = cleaned.replace(/(\*|_)(.*?)\1/g, '$2');
  // Strip headers (#, ##, ###)
  cleaned = cleaned.replace(/^#{1,6}\s+/gm, '');
  // Strip backticks
  cleaned = cleaned.replace(/`([^`]+)`/g, '$1');
  // Strip inline [REF-xxx] tags
  cleaned = cleaned.replace(/\[REF-\d+\]/gi, '');
  // Strip debug metadata strings like (source: rank 1, PMID: unavailable; PDF reference: ...)
  cleaned = cleaned.replace(/\s*\(\s*(source|rank|PMID|PDF reference|collection):.*?\)/gi, '');
  cleaned = cleaned.replace(/\b(PMID:\s*unavailable|PDF reference:\s*[\w.-]+|source:\s*rank\s*\d+)\b/gi, '');
  return cleaned.trim();
}

function appendMessage(sender, text, metadata = {}, animate = false, onComplete = null) {
  const msgDiv = document.createElement('div');
  msgDiv.className = `message message-${sender}`;

  const senderLabel = document.createElement('span');
  senderLabel.className = 'message-sender';
  if (sender === 'user') {
    senderLabel.textContent = 'Clinical Practitioner';
  } else {
    if (metadata.cached) {
      const simPercent = metadata.similarity ? ` (${(metadata.similarity * 100).toFixed(1)}% match)` : '';
      senderLabel.innerHTML = `MediAI Synthesis <span style="display:inline-flex; align-items:center; gap:3px; background:rgba(16, 185, 129, 0.15); color:#10b981; border:1px solid rgba(16,185,129,0.3); border-radius:12px; padding:1px 8px; font-size:11px; margin-left:8px; font-weight:500;">⚡ Semantic Cache Hit${simPercent}</span>`;
    } else {
      senderLabel.textContent = 'MediAI Synthesis';
    }
  }

  const bubble = document.createElement('div');
  bubble.className = 'message-bubble';

  const summarySec = document.createElement('div');
  summarySec.className = 'summary-section';
  const cleanedText = sender === 'assistant' ? stripMarkdownForUI(text) : text;

  bubble.appendChild(summarySec);
  msgDiv.appendChild(senderLabel);
  msgDiv.appendChild(bubble);

  chatFeed.appendChild(msgDiv);
  chatFeed.scrollTop = chatFeed.scrollHeight;

  if (animate && sender === 'assistant' && cleanedText) {
    typewriteText(summarySec, cleanedText, 14, onComplete);
  } else {
    summarySec.textContent = cleanedText;
    if (onComplete) onComplete();
  }
}


function updateInspector(resp) {
  inspectorStatus.textContent = 'Active Response';
  inspectorStatus.className = 'badge badge-clinical';

  // Evidence Support Metrics
  if (evidenceSupportDisplay) {
    const support = resp.evidence_support || {};
    const total = support.total_sources_count || 0;
    const guidelines = support.guideline_passages_count || 0;
    const pubmed = support.pubmed_studies_count || 0;
    const yearRange = support.publication_year_range ? ` (${support.publication_year_range})` : '';
    const dist = support.study_type_distribution || {};

    let distHtml = '';
    for (const [type, count] of Object.entries(dist)) {
      distHtml += `<span class="support-type-badge">${escapeHtml(type)}: ${count}</span> `;
    }

    evidenceSupportDisplay.innerHTML = `
      <div class="support-stats-summary">
        <div class="support-stat-chip"><strong>${total}</strong> Total Sources</div>
        <div class="support-stat-chip"><strong>${guidelines}</strong> Guidelines</div>
        <div class="support-stat-chip"><strong>${pubmed}</strong> Studies${yearRange}</div>
      </div>
      ${distHtml ? `<div class="support-dist-container">${distHtml}</div>` : ''}
    `;
  }

  // Follow-up Questions
  followupList.innerHTML = '';
  if (resp.follow_up_questions && resp.follow_up_questions.length > 0) {
    resp.follow_up_questions.forEach(q => {
      const btn = document.createElement('div');
      btn.className = 'followup-item';
      btn.textContent = q;
      btn.addEventListener('click', () => {
        userInput.value = q;
        chatForm.requestSubmit();
      });
      followupList.appendChild(btn);
    });
  } else {
    followupList.innerHTML = '<span class="empty-state-text">No follow-up questions.</span>';
  }

  // Citations
  citationsList.innerHTML = '';
  if (resp.citations && resp.citations.length > 0) {
    resp.citations.forEach(c => {
      const refId = (c.reference_id || 'REF').toUpperCase();
      const card = document.createElement('div');
      card.className = 'citation-card';
      card.id = `citation-card-${refId}`;

      const isPubmed = c.source_type === 'pubmed' || (c.pmid && String(c.pmid).toLowerCase() !== 'unavailable');

      let metaItems = [];
      if (isPubmed) {
        if (c.pmid && String(c.pmid).toLowerCase() !== 'unavailable') {
          metaItems.push(`PMID: <strong>${escapeHtml(String(c.pmid))}</strong>`);
        }
        let journalYear = [c.journal, c.year ? `(${c.year})` : ''].filter(Boolean).join(' ');
        if (journalYear) {
          metaItems.push(escapeHtml(journalYear));
        }
        if (c.doi && String(c.doi).toLowerCase() !== 'unavailable') {
          metaItems.push(`DOI: ${escapeHtml(String(c.doi))}`);
        }
      } else {
        // Local PDF evidence
        const srcLabel = c.source_type === 'clinical_guideline' ? 'Clinical Guideline' : (c.source_type || 'Clinical Guideline');
        metaItems.push(`Source: <strong>${escapeHtml(srcLabel)}</strong>`);
        if (c.page !== null && c.page !== undefined && String(c.page).toLowerCase() !== 'unavailable') {
          metaItems.push(`Page ${escapeHtml(String(c.page))}`);
        }
      }

      card.innerHTML = `
        <div class="citation-ref">${refId}</div>
        <div style="font-weight: 500; font-size: 0.825rem; color: var(--text-main); margin-bottom: 4px;">${escapeHtml(c.title || 'Untitled Evidence Source')}</div>
        <div style="color: var(--text-muted); font-size: 0.75rem;">
          ${metaItems.join(' • ')}
        </div>
      `;
      citationsList.appendChild(card);
    });
  } else {
    citationsList.innerHTML = '<span class="empty-state-text">No citations generated.</span>';
  }
}

function highlightCitationCard(refId) {
  const card = document.getElementById(`citation-card-${refId}`);
  if (card) {
    document.querySelectorAll('.citation-card').forEach(c => c.classList.remove('highlighted'));
    card.classList.add('highlighted');
    card.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
  }
}
window.highlightCitationCard = highlightCitationCard;

function resetInspector() {
  inspectorStatus.textContent = 'Idle';
  inspectorStatus.className = 'badge';
  if (evidenceSupportDisplay) {
    evidenceSupportDisplay.innerHTML = '<span class="empty-state-text">No active retrieval metrics.</span>';
  }
  followupList.innerHTML = '<span class="empty-state-text">No active response loaded.</span>';
  citationsList.innerHTML = '<span class="empty-state-text">No citations generated yet.</span>';
}


function renderCollections(collections) {
  if (!collections || collections.length === 0) {
    collectionsList.innerHTML = '<div class="collection-item">No collections found.</div>';
    return;
  }

  collectionsList.innerHTML = '';
  collections.forEach(col => {
    const badge = document.createElement('div');
    badge.className = 'collection-badge';
    badge.innerHTML = `
      <span>${escapeHtml(col.name)}</span>
      <span class="collection-count">${col.document_count} chunks</span>
    `;
    collectionsList.appendChild(badge);
  });
}

function renderSessionList() {
  sessionList.innerHTML = '';
  sessions.forEach(sess => {
    const item = document.createElement('div');
    item.className = `session-item ${sess.id === currentSessionId ? 'active' : ''}`;
    item.textContent = sess.title;
    item.addEventListener('click', () => switchSession(sess.id));
    sessionList.appendChild(item);
  });
}

function createNewSession() {
  currentSessionId = 'session-' + Date.now();
  sessions.unshift({
    id: currentSessionId,
    title: 'New Clinical Query',
    messages: [],
    inspectorData: null
  });
  renderSessionList();
  renderCurrentSession();
}

function switchSession(id) {
  currentSessionId = id;
  renderSessionList();
  renderCurrentSession();
}

// ── Agent Timeline Step Simulation ──────────────────────────────────────────

let timelineInterval = null;

function showExecutionTimeline() {
  timeline.classList.remove('hidden');
  const steps = ['guardrail', 'planner', 'retrieval', 'ranking', 'reasoning', 'reflection', 'summary'];

  // Reset steps
  steps.forEach(s => {
    const el = document.getElementById(`step-${s}`);
    if (el) el.className = 'step-chip';
  });

  let currentIdx = 0;
  document.getElementById(`step-${steps[0]}`).className = 'step-chip active';

  timelineInterval = setInterval(() => {
    if (currentIdx < steps.length - 1) {
      document.getElementById(`step-${steps[currentIdx]}`).className = 'step-chip completed';
      currentIdx++;
      document.getElementById(`step-${steps[currentIdx]}`).className = 'step-chip active';
    }
  }, 600);
}

function hideExecutionTimeline() {
  if (timelineInterval) clearInterval(timelineInterval);
  const steps = ['guardrail', 'planner', 'retrieval', 'ranking', 'reasoning', 'reflection', 'summary'];
  steps.forEach(s => {
    const el = document.getElementById(`step-${s}`);
    if (el) el.className = 'step-chip completed';
  });
  setTimeout(() => {
    timeline.classList.add('hidden');
  }, 500);
}

function escapeHtml(str) {
  return str.replace(/[&<>"']/g, function (m) {
    return {
      '&': '&amp;',
      '<': '&lt;',
      '>': '&gt;',
      '"': '&quot;',
      "'": '&#039;'
    }[m];
  });
}
