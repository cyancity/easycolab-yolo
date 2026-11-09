// 只缓存界面外壳，便于 PWA 安装和秒开；/api 与 /v1 永远走网络
const SHELL = "easycolab-shell-v2";
const FILES = ["/", "/app.css", "/tokens.css", "/app.js", "/api.js", "/util.js", "/icons.js", "/views.js", "/sheets.js", "/icon-192.png"];

self.addEventListener("install", (e) => {
  e.waitUntil(caches.open(SHELL).then((c) => c.addAll(FILES)).then(() => self.skipWaiting()));
});
self.addEventListener("activate", (e) => {
  e.waitUntil(caches.keys().then((ks) => Promise.all(ks.filter((k) => k !== SHELL).map((k) => caches.delete(k)))).then(() => self.clients.claim()));
});
self.addEventListener("fetch", (e) => {
  const url = new URL(e.request.url);
  if (url.origin !== location.origin || url.pathname.startsWith("/api/") || url.pathname.startsWith("/v1/")) return;
  // 外壳网络优先，离线时退回缓存，保证改完代码刷新即生效
  e.respondWith(fetch(e.request).then((r) => {
    // 只缓存成功响应，避免把 404/500 存成外壳
    if (r.ok) { const copy = r.clone(); caches.open(SHELL).then((c) => c.put(e.request, copy)); }
    return r;
  }).catch(() => caches.match(e.request)));
});
