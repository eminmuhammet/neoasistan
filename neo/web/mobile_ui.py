from __future__ import annotations

# Served at GET / (no token required for the page shell itself -- only the
# actual /message calls the page makes need the token, which the user pastes
# once and the page remembers in localStorage). Dark/green theme to match
# the desktop app; safe-area insets so it sits properly under the iPhone
# notch/home-indicator once added to the home screen as a PWA.
PAGE_HTML = """<!doctype html>
<html lang="tr">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover, maximum-scale=1">
<meta name="theme-color" content="#050a07">
<meta name="apple-mobile-web-app-capable" content="yes">
<meta name="apple-mobile-web-app-status-bar-style" content="black-translucent">
<meta name="apple-mobile-web-app-title" content="NEO">
<link rel="manifest" href="/manifest.json">
<link rel="apple-touch-icon" href="/icon.png">
<title>NEO</title>
<style>
  :root { color-scheme: dark; }
  * { box-sizing: border-box; -webkit-tap-highlight-color: transparent; }
  html, body { height: 100%; margin: 0; }
  body {
    background: #050a07;
    color: #d8ffe8;
    font: 15px/1.45 -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    display: flex; flex-direction: column;
    padding-top: env(safe-area-inset-top);
    padding-bottom: env(safe-area-inset-bottom);
  }
  header {
    padding: 14px 18px 10px;
    text-align: center;
    border-bottom: 1px solid #113322;
  }
  header .dot { color: #35e08a; }
  header h1 {
    margin: 0; font-size: 15px; font-weight: 700; letter-spacing: 4px;
    color: #7bffb0;
  }
  header p { margin: 2px 0 0; font-size: 11px; color: #4a8a68; letter-spacing: 1px; }
  #log {
    flex: 1; overflow-y: auto; padding: 16px; display: flex; flex-direction: column; gap: 10px;
    -webkit-overflow-scrolling: touch;
  }
  .msg { max-width: 84%; padding: 10px 14px; border-radius: 16px; white-space: pre-wrap; word-wrap: break-word; }
  .me { align-self: flex-end; background: #14351f; color: #eafff2; border-bottom-right-radius: 4px; }
  .neo { align-self: flex-start; background: #0d1a13; border: 1px solid #1c3d29; color: #cdf5df; border-bottom-left-radius: 4px; }
  .sys { align-self: center; color: #4a8a68; font-size: 12px; text-align: center; }
  form {
    display: flex; gap: 8px; padding: 10px 12px calc(10px + env(safe-area-inset-bottom));
    border-top: 1px solid #113322; background: #050a07;
  }
  input[type=text] {
    flex: 1; background: #0d1a13; border: 1px solid #1c3d29; border-radius: 20px;
    padding: 11px 16px; color: #eafff2; font-size: 16px; outline: none;
  }
  button {
    background: #1c8a53; color: #eafff2; border: none; border-radius: 20px;
    padding: 0 20px; font-size: 15px; font-weight: 600;
  }
  button:disabled { opacity: 0.5; }
  #tokenOverlay {
    position: fixed; inset: 0; background: #050a07; display: flex; flex-direction: column;
    align-items: center; justify-content: center; gap: 14px; padding: 24px; text-align: center;
  }
  #tokenOverlay input {
    width: 100%; max-width: 320px; background: #0d1a13; border: 1px solid #1c3d29;
    border-radius: 10px; padding: 12px; color: #eafff2; font-size: 15px;
  }
  #tokenOverlay button { padding: 12px 24px; border-radius: 10px; }
  [hidden] { display: none !important; }
</style>
</head>
<body>

<div id="tokenOverlay">
  <h1 style="color:#7bffb0;letter-spacing:3px;">N E O</h1>
  <p style="color:#8fd6ac;max-width:320px;">
    Bilgisayarındaki <code>data/web_token.txt</code> dosyasındaki tokeni yapıştır.
  </p>
  <input id="tokenInput" type="text" placeholder="token" autocapitalize="off" autocorrect="off" spellcheck="false">
  <button id="tokenSave">Bağlan</button>
  <p id="tokenError" style="color:#ff6b6b;" hidden>Geçersiz token ya da NEO'ya ulaşılamıyor.</p>
</div>

<header>
  <h1><span class="dot">●</span> NEO</h1>
  <p>KİŞİSEL YAPAY ZEKÂ ASİSTANI</p>
</header>
<div id="log"></div>
<form id="form">
  <input id="text" type="text" placeholder="NEO'ya bir şey söyle..." autocomplete="off" autocapitalize="sentences">
  <button id="send" type="submit">Gönder</button>
</form>

<script>
const log = document.getElementById('log');
const form = document.getElementById('form');
const textInput = document.getElementById('text');
const sendBtn = document.getElementById('send');
const overlay = document.getElementById('tokenOverlay');
const tokenInput = document.getElementById('tokenInput');
const tokenError = document.getElementById('tokenError');

function addMsg(who, text) {
  const div = document.createElement('div');
  div.className = 'msg ' + who;
  div.textContent = text;
  log.appendChild(div);
  log.scrollTop = log.scrollHeight;
}
function addSys(text) {
  const div = document.createElement('div');
  div.className = 'msg sys';
  div.textContent = text;
  log.appendChild(div);
  log.scrollTop = log.scrollHeight;
}

function getToken() { return localStorage.getItem('neo_token') || ''; }

async function checkToken(token) {
  const r = await fetch('/status', { headers: { 'x-neo-token': token } });
  return r.ok;
}

async function trySavedToken() {
  const token = getToken();
  if (!token) return;
  if (await checkToken(token)) {
    overlay.hidden = true;
    addSys('Bağlandı.');
  }
}

document.getElementById('tokenSave').addEventListener('click', async () => {
  const token = tokenInput.value.trim();
  if (!token) return;
  tokenError.hidden = true;
  if (await checkToken(token)) {
    localStorage.setItem('neo_token', token);
    overlay.hidden = true;
    addSys('Bağlandı.');
  } else {
    tokenError.hidden = false;
  }
});

form.addEventListener('submit', async (e) => {
  e.preventDefault();
  const text = textInput.value.trim();
  if (!text) return;
  addMsg('me', text);
  textInput.value = '';
  sendBtn.disabled = true;
  try {
    const r = await fetch('/message', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json', 'x-neo-token': getToken() },
      body: JSON.stringify({ text }),
    });
    if (r.status === 401) {
      addSys('Token geçersiz oldu, tekrar bağlanman gerekiyor.');
      localStorage.removeItem('neo_token');
      overlay.hidden = false;
      return;
    }
    const data = await r.json();
    addMsg('neo', data.reply || data.detail || 'Yanıt alınamadı.');
  } catch (err) {
    addSys('Bağlantı hatası: ' + err.message);
  } finally {
    sendBtn.disabled = false;
  }
});

if ('serviceWorker' in navigator) {
  navigator.serviceWorker.register('/sw.js').catch(() => {});
}

trySavedToken();
</script>
</body>
</html>
"""

MANIFEST_JSON = """{
  "name": "NEO",
  "short_name": "NEO",
  "start_url": "/",
  "display": "standalone",
  "background_color": "#050a07",
  "theme_color": "#050a07",
  "icons": [{"src": "/icon.png", "sizes": "512x512", "type": "image/png"}]
}
"""

# Minimal: caches the app shell so re-opening from the home screen icon
# doesn't show a blank/error page on a flaky connection. Deliberately not
# caching /message or /status -- those must always hit the real NEO.
SERVICE_WORKER_JS = """
const SHELL_CACHE = 'neo-shell-v1';
self.addEventListener('install', (event) => {
  event.waitUntil(caches.open(SHELL_CACHE).then((c) => c.add('/')));
  self.skipWaiting();
});
self.addEventListener('fetch', (event) => {
  if (event.request.method !== 'GET' || new URL(event.request.url).pathname !== '/') return;
  event.respondWith(
    fetch(event.request).catch(() => caches.match('/'))
  );
});
"""


def build_icon_png() -> bytes:
    """A simple 512x512 rounded-square NEO mark (green ring + dot on dark
    ground) generated at runtime with Pillow, which is already a project
    dependency -- no separate image asset to keep in sync with the theme."""
    import io

    from PIL import Image, ImageDraw

    size = 512
    img = Image.new("RGBA", (size, size), (5, 10, 7, 255))
    draw = ImageDraw.Draw(img)
    margin = 40
    draw.rounded_rectangle(
        [0, 0, size, size], radius=110, fill=(5, 10, 7, 255)
    )
    draw.ellipse(
        [margin, margin, size - margin, size - margin],
        outline=(53, 224, 138, 255),
        width=22,
    )
    center = size // 2
    r = 56
    draw.ellipse([center - r, center - r, center + r, center + r], fill=(53, 224, 138, 255))
    buffer = io.BytesIO()
    img.save(buffer, format="PNG")
    return buffer.getvalue()
