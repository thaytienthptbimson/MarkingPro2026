// Lưu sẵn thư viện (OpenCV, Tesseract, SheetJS) để các lần mở sau tải gần như tức thì
const C = 'omr-v1', HOSTS = ['docs.opencv.org', 'cdn.jsdelivr.net', 'cdnjs.cloudflare.com'];
self.addEventListener('install', () => self.skipWaiting());
self.addEventListener('activate', e => e.waitUntil(clients.claim()));
self.addEventListener('fetch', e => {
  if (e.request.method != 'GET' || !HOSTS.includes(new URL(e.request.url).hostname)) return;
  e.respondWith(caches.open(C).then(async c => {
    const hit = await c.match(e.request); if (hit) return hit;
    const res = await fetch(e.request); if (res.ok || res.type == 'opaque') c.put(e.request, res.clone());
    return res;
  }));
});
