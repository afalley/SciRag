// JavaScript
function renderNow(el) {
  if (window.renderAnswerMath) {
    window.renderAnswerMath(el);
  }
}

// Lightweight gallery renderer for source images
function renderSourcesWithImages(sources) {
  const srcEl = document.getElementById('sources');
  if (!Array.isArray(sources) || sources.length === 0) {
    srcEl.innerHTML = '';
    return;
  }
  const html = sources.map(s => {
    const pills = `<span class="source-pill">${s.source_name} · chunk ${s.chunk_index}${typeof s.score === 'number' ? ` · score ${s.score.toFixed(3)}` : ''}${typeof s.page === 'number' && s.page >= 0 ? ` · page ${s.page}` : ''}</span>`;
    const imgs = Array.isArray(s.images) && s.images.length
      ? `<div class="source-images">` + s.images.map(img => {
          const alt = img.alt || `${s.source_name} p${img.page}`;
          const thumb = img.thumb;
          const full = img.full || thumb;
          return `<a href="${full}" target="_blank" rel="noopener noreferrer">
                    <img src="${thumb}" loading="lazy" decoding="async" alt="${alt}" />
                  </a>`;
        }).join('') + `</div>`
      : '';
    return `<div class="source-block">${pills}${imgs}</div>`;
  }).join('');
  srcEl.innerHTML = html;
}

async function ask() {
  const queryEl = document.getElementById('query');
  const topkEl = document.getElementById('topk');
  const ansEl = document.getElementById('answer');

  const query = (queryEl.value || '').trim();
  const top_k = parseInt(topkEl.value || '5', 10);

  if (!query) {
    ansEl.textContent = 'Please enter a question.';
    document.getElementById('sources').innerHTML = '';
    renderNow(ansEl);
    return;
  }

  ansEl.textContent = 'Thinking...';
  document.getElementById('sources').innerHTML = '';
  renderNow(ansEl);

  try {
    const res = await fetch('/query', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ query, top_k })
    });

    if (!res.ok) {
      const err = await res.json().catch(() => ({}));
      ansEl.textContent = 'Error: ' + (err.detail || res.statusText);
      renderNow(ansEl);
      return;
    }

    const data = await res.json();

    ansEl.textContent = data.answer || '(no answer)';
    renderNow(ansEl);

    renderSourcesWithImages(data.sources || []);
  } catch (e) {
    ansEl.textContent = 'Network error: ' + e;
    renderNow(ansEl);
  }
}

document.addEventListener('DOMContentLoaded', () => {
  const btn = document.getElementById('ask');
  if (btn) btn.addEventListener('click', ask);
  // Initial typeset for any math on load
  const ansEl = document.getElementById('answer');
  renderNow(ansEl);
});