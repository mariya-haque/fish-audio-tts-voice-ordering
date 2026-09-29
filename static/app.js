// Browser half of the text-to-speech agent.
//
// You type; the server streams back the reply as text tokens and, concurrently,
// raw PCM audio chunks. Those chunks go into an AudioWorklet running at the
// server's synthesis rate, so playback starts before the reply is finished.

const state = {
  ws: null,
  sessionId: Math.random().toString(36).slice(2, 10),
  playbackCtx: null,
  playbackNode: null,
  playbackRate: 44100,
  speaking: false,
  assistantEl: null,
  // Diagnostics — surfaced in the UI so "I can't hear it" becomes answerable.
  chunks: 0,
  bytes: 0,
  // A chunk boundary can land mid-sample; an odd-length buffer makes
  // `new Int16Array(buf)` throw inside the worklet and kills it silently.
  leftover: null,
};

const el = (id) => document.getElementById(id);

function setStatus(text, kind = '') {
  const node = el('status');
  node.textContent = text;
  node.className = `status ${kind}`;
}

function log(msg) {
  const node = el('log');
  const line = document.createElement('div');
  line.textContent = `${new Date().toLocaleTimeString()}  ${msg}`;
  node.appendChild(line);
  node.scrollTop = node.scrollHeight;
}

function updateAudioInfo(extra = '') {
  const ctx = state.playbackCtx;
  const bits = [
    `context: ${ctx ? ctx.state : 'not created'}`,
    ctx ? `${ctx.sampleRate} Hz` : `expects ${state.playbackRate} Hz`,
    `${state.chunks} chunks`,
    `${(state.bytes / 1024).toFixed(0)} KB`,
  ];
  if (extra) bits.push(extra);
  el('audio-info').textContent = bits.join(' · ');
}

// --- transcript -------------------------------------------------------------

function addBubble(role, text) {
  const wrap = el('transcript');
  const bubble = document.createElement('div');
  bubble.className = `bubble ${role}`;
  bubble.textContent = text;
  wrap.appendChild(bubble);
  wrap.scrollTop = wrap.scrollHeight;
  return bubble;
}

function appendToAssistant(text) {
  if (!state.assistantEl) state.assistantEl = addBubble('assistant', '');
  state.assistantEl.textContent += text;
  el('transcript').scrollTop = el('transcript').scrollHeight;
}

// --- playback ---------------------------------------------------------------

async function buildPlayback() {
  if (state.playbackCtx && state.playbackCtx.sampleRate === state.playbackRate) return;
  if (state.playbackCtx) await state.playbackCtx.close();

  // Matching the context rate to the synthesis rate avoids any resampling.
  const ctx = new AudioContext({ sampleRate: state.playbackRate });
  await ctx.audioWorklet.addModule('audio-playback.worklet.js');

  const node = new AudioWorkletNode(ctx, 'pcm-playback', {
    numberOfInputs: 0,
    numberOfOutputs: 1,
    outputChannelCount: [1],
  });
  node.connect(ctx.destination);

  ctx.onstatechange = () => {
    log(`audio context → ${ctx.state}`);
    updateAudioInfo();
    reflectBlocked();
  };

  state.playbackCtx = ctx;
  state.playbackNode = node;
  log(`audio ready at ${ctx.sampleRate} Hz (${ctx.state})`);
  updateAudioInfo();
}

/** Browsers only honour resume() inside a real user gesture, so call this
 *  synchronously from the click/keypress handler — never after an await. */
function unlockAudio() {
  const ctx = state.playbackCtx;
  if (ctx && ctx.state !== 'running') {
    ctx.resume().then(reflectBlocked, () => {});
  }
}

function reflectBlocked() {
  const blocked = !state.playbackCtx || state.playbackCtx.state !== 'running';
  el('unblock').hidden = !blocked;
}

function stopPlayback() {
  if (state.playbackNode) state.playbackNode.port.postMessage({ type: 'stop' });
}

/** Feed raw PCM bytes to the worklet, carrying any half-sample across chunks. */
function pushAudio(buffer) {
  let bytes = new Uint8Array(buffer);

  if (state.leftover) {
    const merged = new Uint8Array(state.leftover.length + bytes.length);
    merged.set(state.leftover, 0);
    merged.set(bytes, state.leftover.length);
    bytes = merged;
    state.leftover = null;
  }

  if (bytes.length % 2 === 1) {
    state.leftover = bytes.slice(bytes.length - 1);
    bytes = bytes.subarray(0, bytes.length - 1);
    log('carried a half-sample to the next chunk');
  }
  if (bytes.length === 0) return;

  // Copy into a fresh buffer so it can be transferred without detaching a view.
  const payload = bytes.slice().buffer;
  state.playbackNode.port.postMessage({ type: 'chunk', payload }, [payload]);
}

// --- self test ---------------------------------------------------------------

/** Play a short tone through the exact same worklet path as real speech.
 *  If you hear this but not the agent, the problem is the stream, not output. */
async function testTone() {
  unlockAudio();
  await buildPlayback();
  unlockAudio();

  const rate = state.playbackCtx.sampleRate;
  const samples = Math.floor(rate * 0.6);
  const pcm = new Int16Array(samples);
  for (let i = 0; i < samples; i++) {
    // Fade both ends so it does not click.
    const env = Math.min(1, i / (rate * 0.02), (samples - i) / (rate * 0.02));
    pcm[i] = Math.round(Math.sin((2 * Math.PI * 440 * i) / rate) * 8000 * env);
  }
  state.playbackNode.port.postMessage(
    { type: 'chunk', payload: pcm.buffer }, [pcm.buffer],
  );
  log(`test tone: 440 Hz, 0.6s at ${rate} Hz`);
  setStatus('test tone playing — if you hear nothing, check the system volume/output device', 'live');
  updateAudioInfo('tone sent');
}

// --- websocket --------------------------------------------------------------

function connect() {
  const proto = location.protocol === 'https:' ? 'wss' : 'ws';
  const ws = new WebSocket(`${proto}://${location.host}/ws/${state.sessionId}`);
  ws.binaryType = 'arraybuffer';
  state.ws = ws;

  ws.onopen = () => setStatus('ready', 'ok');
  ws.onerror = () => setStatus('websocket error', 'err');
  ws.onclose = () => {
    setStatus('disconnected — reconnecting…', 'warn');
    setTimeout(connect, 1500);
  };

  ws.onmessage = (event) => {
    if (event.data instanceof ArrayBuffer) {
      state.chunks += 1;
      state.bytes += event.data.byteLength;
      if (!state.playbackNode) {
        log('AUDIO ARRIVED BUT PLAYBACK IS NOT READY — press "Test sound" once');
        return;
      }
      pushAudio(event.data);
      if (state.chunks % 4 === 1) updateAudioInfo();
      return;
    }
    handleControl(JSON.parse(event.data));
  };
}

async function handleControl(msg) {
  switch (msg.type) {
    case 'ready': {
      state.playbackRate = msg.playback_sample_rate;
      const kb = msg.rag_enabled ? `${msg.kb_chunks} kb chunks` : 'rag off';
      el('meta').textContent =
        `${msg.llm_model} · ${msg.tts_model} · ${msg.voice || 'stock voice'} · ` +
        `${msg.playback_sample_rate} Hz · ${kb}`;
      if (msg.missing_keys && msg.missing_keys.length) {
        setStatus(`missing ${msg.missing_keys.join(', ')} in .env`, 'err');
      }
      // Build it now (suspended is fine) so audio never arrives before the node.
      await buildPlayback();
      reflectBlocked();
      break;
    }

    case 'transcript':
      state.assistantEl = null;
      addBubble('user', msg.text);
      break;

    case 'context':
      log(`retrieved ${msg.count} chunk(s) from ${msg.sources.join(', ')}`);
      break;

    case 'token':
      appendToAssistant(msg.text);
      break;

    case 'audio_start':
      state.speaking = true;
      state.chunks = 0;
      state.bytes = 0;
      state.leftover = null;
      setStatus(msg.verbatim ? 'speaking your text…' : 'speaking…', 'live');
      el('stop').disabled = false;
      break;

    case 'audio_end': {
      state.speaking = false;
      el('stop').disabled = true;
      const ctx = state.playbackCtx;
      if (state.chunks === 0) {
        setStatus('the server sent no audio for that turn', 'err');
      } else if (!ctx || ctx.state !== 'running') {
        setStatus('audio received but the browser is blocking playback — press "Enable sound"', 'err');
      } else {
        setStatus('ready', 'ok');
      }
      updateAudioInfo('done');
      break;
    }

    case 'interrupted':
      state.speaking = false;
      stopPlayback();
      setStatus('stopped', 'warn');
      el('stop').disabled = true;
      break;

    case 'timing': {
      const parts = [];
      if (msg.rag_ms) parts.push(`rag ${msg.rag_ms}ms`);
      if (msg.ttft_ms != null) parts.push(`first token ${msg.ttft_ms}ms`);
      if (msg.ttfa_ms != null) parts.push(`first audio ${msg.ttfa_ms}ms`);
      parts.push(`total ${msg.total_ms}ms`);
      parts.push(`${msg.audio_seconds}s of speech`);
      el('latency').textContent = parts.join(' · ');
      break;
    }

    case 'info':
      log(msg.info);
      break;

    case 'error':
      log(`ERROR: ${msg.error}`);
      setStatus(msg.error, 'err');
      el('stop').disabled = true;
      break;
  }
}

function sendControl(payload) {
  if (state.ws && state.ws.readyState === WebSocket.OPEN) {
    state.ws.send(JSON.stringify(payload));
  }
}

// --- wiring -----------------------------------------------------------------

async function submit() {
  const input = el('text-input');
  const text = input.value.trim();
  if (!text) return;

  unlockAudio();          // synchronous, inside the gesture
  await buildPlayback();  // no-op once built
  unlockAudio();
  stopPlayback();

  input.value = '';
  state.assistantEl = null;
  el('latency').textContent = '';
  setStatus('thinking…', 'live');

  sendControl({ type: 'text', text, verbatim: el('verbatim').checked });
}

el('composer').addEventListener('submit', (event) => {
  event.preventDefault();
  submit();
});

// Enter sends, Shift+Enter makes a new line.
el('text-input').addEventListener('keydown', (event) => {
  if (event.key === 'Enter' && !event.shiftKey) {
    event.preventDefault();
    submit();
  }
});

el('unblock').addEventListener('click', async () => {
  unlockAudio();
  await buildPlayback();
  unlockAudio();
  reflectBlocked();
});

el('testtone').addEventListener('click', testTone);

el('stop').addEventListener('click', () => {
  sendControl({ type: 'interrupt' });
  stopPlayback();
});

el('reset').addEventListener('click', () => {
  sendControl({ type: 'reset' });
  stopPlayback();
  el('transcript').innerHTML = '';
  el('latency').textContent = '';
  state.assistantEl = null;
});

el('verbatim').addEventListener('change', (event) => {
  el('mode-hint').textContent = event.target.checked
    ? 'Speaks your text exactly as written. No AI.'
    : 'The assistant answers, and the answer is spoken as it is written.';
});

updateAudioInfo();
connect();
