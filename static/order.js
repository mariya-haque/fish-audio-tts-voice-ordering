// Browser half of the ordering bot.
//
// You type; the server answers with tokens and PCM; the cart panel tracks what
// the bot has actually added. Same shape as app.js on the text-to-speech page,
// with a cart on the side. No microphone.
//
// The playback rate comes from the server's `ready` message rather than being
// hardcoded, because an AudioContext built at the wrong rate plays everything
// back at the wrong pitch.

const els = {
  transcript: document.getElementById('transcript'),
  talk: document.getElementById('talk'),
  level: document.querySelector('#level > div'),
  text: document.getElementById('text'),
  stop: document.getElementById('stop'),
  reset: document.getElementById('reset'),
  unblock: document.getElementById('unblock'),
  dot: document.getElementById('dot'),
  status: document.getElementById('statustext'),
  restaurant: document.getElementById('restaurant'),
  lines: document.getElementById('lines'),
  totals: document.getElementById('totals'),
  needed: document.getElementById('needed'),
  placed: document.getElementById('placed'),
  latency: document.getElementById('latency'),
};

const sessionId = Math.random().toString(36).slice(2, 10);
let ws = null;
let playbackContext = null;
let playbackNode = null;
let sampleRate = 44100;
let micRate = 16000;
let assistantBubble = null;
let speaking = false;

// Push-to-talk state.
let recordContext = null;
let recordNode = null;
let micStream = null;
let holding = false;

// --- transcript --------------------------------------------------------------
function atBottom() {
  const el = els.transcript;
  return el.scrollHeight - el.scrollTop - el.clientHeight < 60;
}

function scroll(wasAtBottom) {
  if (wasAtBottom) els.transcript.scrollTop = els.transcript.scrollHeight;
}

function bubble(role, text, pending) {
  const wasAtBottom = atBottom();
  const el = document.createElement('div');
  el.className = `bubble ${role}${pending ? ' pending' : ''}`;
  el.textContent = text;
  els.transcript.appendChild(el);
  scroll(wasAtBottom);
  return el;
}

function event(text, isError) {
  const wasAtBottom = atBottom();
  const el = document.createElement('div');
  el.className = `event${isError ? ' err' : ''}`;
  el.textContent = text;
  els.transcript.appendChild(el);
  scroll(wasAtBottom);
}

function setStatus(text, state) {
  els.status.textContent = text;
  els.dot.className = state || '';
}

// --- cart panel --------------------------------------------------------------
function escapeHtml(text) {
  const div = document.createElement('div');
  div.textContent = text == null ? '' : String(text);
  return div.innerHTML;
}

function renderCart(msg) {
  const cart = msg.cart || {};
  const lines = cart.lines || [];
  const currency = cart.currency || '';

  els.lines.innerHTML = '';
  if (!lines.length) {
    els.lines.innerHTML = '<div id="empty">Nothing yet.</div>';
  } else {
    for (const line of lines) {
      const el = document.createElement('div');
      el.className = 'line';
      const detail = [line.choice, ...(line.extras || []), line.special_instructions]
        .filter(Boolean).join(', ');
      el.innerHTML =
        `<span class="amt">${line.line_total}</span>` +
        `<span class="qty">${line.quantity > 1 ? line.quantity + ' x ' : ''}</span>` +
        `${escapeHtml(line.name)}` +
        (detail ? `<div class="opts">${escapeHtml(detail)}</div>` : '');
      els.lines.appendChild(el);
    }
  }

  const rows = [];
  if (lines.length) {
    rows.push(`<div class="row"><span>Subtotal</span><span>${cart.subtotal} ${currency}</span></div>`);
    if (cart.delivery_fee) {
      rows.push(`<div class="row"><span>Delivery</span><span>${cart.delivery_fee} ${currency}</span></div>`);
    }
    rows.push(`<div class="row total"><span>Total</span><span>${cart.total} ${currency}</span></div>`);
  }
  rows.push(`<div class="row"><span style="color:var(--muted)">${msg.order_type || ''}</span><span></span></div>`);
  els.totals.innerHTML = rows.join('');

  const needed = msg.still_needed || [];
  els.needed.textContent = needed.length && lines.length
    ? `Still needed: ${needed.join(', ')}`
    : '';
}

// --- audio -------------------------------------------------------------------
async function buildPlayback() {
  if (playbackNode) return;
  playbackContext = new AudioContext({ sampleRate });
  await playbackContext.audioWorklet.addModule('audio-playback.worklet.js');
  playbackNode = new AudioWorkletNode(playbackContext, 'pcm-playback', {
    numberOfInputs: 0,
    outputChannelCount: [1],
  });
  playbackNode.connect(playbackContext.destination);
}

// Browsers refuse to start audio without a gesture. Typing counts, so the
// button is only shown if a context somehow starts suspended anyway.
async function unlockAudio() {
  await buildPlayback();
  if (playbackContext.state === 'suspended') await playbackContext.resume();
  els.unblock.style.display = playbackContext.state === 'running' ? 'none' : '';
}

// A sample can straddle two WebSocket frames; keeping the odd byte stops
// `new Int16Array(buffer)` throwing on a non-even length.
let carry = null;

function pushAudio(buffer) {
  if (!playbackNode) return;
  let bytes = new Uint8Array(buffer);
  if (carry) {
    const merged = new Uint8Array(carry.length + bytes.length);
    merged.set(carry, 0);
    merged.set(bytes, carry.length);
    bytes = merged;
    carry = null;
  }
  const usable = bytes.length - (bytes.length % 2);
  if (usable < bytes.length) carry = bytes.slice(usable);
  if (!usable) return;

  const payload = bytes.buffer.slice(bytes.byteOffset, bytes.byteOffset + usable);
  playbackNode.port.postMessage({ type: 'chunk', payload }, [payload]);
}

function stopPlayback() {
  carry = null;
  if (playbackNode) playbackNode.port.postMessage({ type: 'stop' });
}

// --- microphone (push to talk) ----------------------------------------------
// Releasing the button is the end-of-speech signal. Fish ASR is a batch
// endpoint with no voice activity detection, and a held button is an exact
// answer to "have they finished?" where silence detection is a guess that cuts
// people off mid-sentence.
async function openMic() {
  if (recordNode) return true;
  try {
    micStream = await navigator.mediaDevices.getUserMedia({
      audio: {
        channelCount: 1,
        echoCancellation: true,
        noiseSuppression: true,
        autoGainControl: true,
      },
    });
  } catch (err) {
    event(`microphone blocked: ${err.name}`, true);
    return false;
  }

  // Asking for the recogniser's rate up front makes the browser resample the
  // 48 kHz microphone for us, which is far cheaper than doing it in JS.
  recordContext = new AudioContext({ sampleRate: micRate });
  await recordContext.audioWorklet.addModule('audio-recorder.worklet.js');
  const source = recordContext.createMediaStreamSource(micStream);
  recordNode = new AudioWorkletNode(recordContext, 'pcm-recorder');

  recordNode.port.onmessage = (e) => {
    const chunk = e.data;
    if (!chunk || !holding || !ws || ws.readyState !== WebSocket.OPEN) return;
    ws.send(chunk.buffer);

    let peak = 0;
    for (let i = 0; i < chunk.length; i += 16) {
      const v = Math.abs(chunk[i]);
      if (v > peak) peak = v;
    }
    els.level.style.width = `${Math.min(100, (peak / 32768) * 260)}%`;
  };

  source.connect(recordNode);
  recordNode.connect(recordContext.destination);
  return true;
}

async function startTalking() {
  if (holding) return;
  await unlockAudio();
  if (!(await openMic())) return;
  if (recordContext.state === 'suspended') await recordContext.resume();

  // Talking over the bot cuts it off, same as sending a new message.
  stopPlayback();
  holding = true;
  els.talk.classList.add('on');
  els.talk.textContent = 'Listening…';
  send({ type: 'speech_start' });
}

function stopTalking() {
  if (!holding) return;
  holding = false;
  els.talk.classList.remove('on');
  els.talk.textContent = 'Hold to talk';
  els.level.style.width = '0%';
  send({ type: 'speech_end' });
}

// --- socket ------------------------------------------------------------------
function connect() {
  const scheme = location.protocol === 'https:' ? 'wss' : 'ws';
  ws = new WebSocket(`${scheme}://${location.host}/ws/order/${sessionId}`);
  ws.binaryType = 'arraybuffer';

  ws.onopen = () => setStatus('connected', 'live');

  ws.onmessage = (e) => {
    if (e.data instanceof ArrayBuffer) { pushAudio(e.data); return; }
    let msg;
    try { msg = JSON.parse(e.data); } catch { return; }
    handle(msg);
  };

  ws.onclose = () => {
    setStatus('reconnecting', 'error');
    els.text.disabled = true;
    setTimeout(connect, 1500);
  };

  ws.onerror = () => setStatus('connection error', 'error');
}

function handle(msg) {
  switch (msg.type) {
    case 'ready':
      sampleRate = msg.playback_sample_rate || 44100;
      micRate = msg.mic_sample_rate || 16000;
      setStatus(`${msg.llm_model} · ${msg.tts_model} · ${msg.voice || 'stock voice'}`, 'live');
      els.restaurant.textContent =
        `${msg.restaurant} — ${msg.items} items in ${msg.categories} categories`;
      els.text.disabled = false;
      els.talk.disabled = false;
      els.text.focus();
      break;

    case 'transcribing':
      event(`heard ${msg.seconds}s, transcribing…`);
      break;

    case 'transcript':
      if (msg.role === 'user') bubble('user', msg.text);
      break;

    case 'token': {
      const wasAtBottom = atBottom();
      if (!assistantBubble) assistantBubble = bubble('assistant', '', true);
      assistantBubble.textContent += msg.text;
      scroll(wasAtBottom);
      break;
    }

    case 'tool':
      event(`${msg.name}(${Object.keys(msg.arguments || {}).join(', ')})`);
      break;

    case 'cart':
      renderCart(msg);
      if (msg.order_id) {
        els.placed.style.display = 'block';
        els.placed.textContent = `Order placed — ${msg.order_id}`;
      }
      break;

    case 'order_placed':
      event(`order placed: ${msg.order_id} — ${msg.total} ${msg.currency || ''}`);
      break;

    case 'audio_start':
      speaking = true;
      els.stop.disabled = false;
      break;

    case 'audio_end':
      speaking = false;
      els.stop.disabled = true;
      if (assistantBubble) { assistantBubble.classList.remove('pending'); assistantBubble = null; }
      break;

    case 'timing':
      els.latency.textContent =
        `first token ${msg.ttft_ms ?? '-'}ms · first audio ${msg.ttfa_ms ?? '-'}ms · ` +
        `total ${msg.total_ms}ms · ${msg.audio_seconds}s spoken` +
        (msg.tools && msg.tools.length ? ` · ${msg.tools.join(', ')}` : '');
      break;

    case 'interrupted':
      stopPlayback();
      speaking = false;
      els.stop.disabled = true;
      if (assistantBubble) { assistantBubble.classList.remove('pending'); assistantBubble = null; }
      break;

    case 'info':
      event(msg.info);
      break;

    case 'error':
      event(msg.error, true);
      break;
  }
}

function send(payload) {
  if (ws && ws.readyState === WebSocket.OPEN) ws.send(JSON.stringify(payload));
}

// --- wiring ------------------------------------------------------------------
async function submit() {
  const text = els.text.value.trim();
  if (!text) return;
  els.text.value = '';
  await unlockAudio();
  stopPlayback();
  send({ type: 'text', text });
}

els.text.addEventListener('keydown', (e) => {
  if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); submit(); }
});

// Mouse, touch and spacebar all drive the same hold. `pointerleave` and the
// window-level `pointerup` matter: dragging off the button or releasing
// outside it would otherwise leave the mic recording forever.
els.talk.addEventListener('pointerdown', (e) => { e.preventDefault(); startTalking(); });
els.talk.addEventListener('pointerleave', stopTalking);
window.addEventListener('pointerup', stopTalking);
window.addEventListener('blur', stopTalking);

window.addEventListener('keydown', (e) => {
  if (e.code !== 'Space' || e.repeat) return;
  if (document.activeElement === els.text) return;   // typing a space
  e.preventDefault();
  startTalking();
});
window.addEventListener('keyup', (e) => {
  if (e.code === 'Space' && document.activeElement !== els.text) stopTalking();
});

els.stop.addEventListener('click', () => { stopPlayback(); send({ type: 'interrupt' }); });
els.unblock.addEventListener('click', unlockAudio);

els.reset.addEventListener('click', () => {
  stopPlayback();
  els.transcript.innerHTML = '';
  els.placed.style.display = 'none';
  els.latency.textContent = '';
  assistantBubble = null;
  send({ type: 'reset' });
});

connect();
