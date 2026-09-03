/**
 * Service worker for MTG Scanner.
 *
 * Buckets:
 *   app    - the SPA shell + hashed assets: stale-while-revalidate
 *   models - /models/* (detector ONNX) and the onnxruntime wasm: cache-first
 *   never  - /api/* and /scans/* always go to the network
 */

/// <reference lib="webworker" />
const sw = /** @type {ServiceWorkerGlobalScope} */ (self);

const APP_CACHE = 'mtg-scanner-app-v2';
const MODEL_CACHE = 'mtg-scanner-models-v2';
const PRECACHE = ['/', '/manifest.webmanifest'];

sw.addEventListener('install', (event) => {
	event.waitUntil(
		(async () => {
			const cache = await caches.open(APP_CACHE);
			await cache.addAll(PRECACHE).catch(() => {});
			await sw.skipWaiting();
		})(),
	);
});

sw.addEventListener('activate', (event) => {
	event.waitUntil(
		(async () => {
			const keep = new Set([APP_CACHE, MODEL_CACHE]);
			await Promise.all((await caches.keys()).filter((k) => !keep.has(k)).map((k) => caches.delete(k)));
			await sw.clients.claim();
		})(),
	);
});

/** @param {URL} url */
function route(url) {
	if (url.origin !== sw.location.origin) return null;
	if (url.pathname.startsWith('/api/') || url.pathname.startsWith('/scans/')) return null;
	if (url.pathname.startsWith('/models/') || url.pathname.endsWith('.wasm')) {
		return { cache: MODEL_CACHE, strategy: 'cache-first' };
	}
	return { cache: APP_CACHE, strategy: 'swr' };
}

sw.addEventListener('fetch', (event) => {
	const req = event.request;
	if (req.method !== 'GET') return;
	const r = route(new URL(req.url));
	if (!r) return;

	event.respondWith(
		(async () => {
			const cache = await caches.open(r.cache);
			// Navigations all resolve to the SPA shell.
			const key = req.mode === 'navigate' ? '/' : req;
			const cached = await cache.match(key);

			if (r.strategy === 'cache-first') {
				if (cached) return cached;
				const fresh = await fetch(req);
				if (fresh.ok) cache.put(key, fresh.clone());
				return fresh;
			}

			const network = fetch(req)
				.then((res) => {
					if (res.ok) cache.put(key, res.clone());
					return res;
				})
				.catch(() => cached);
			return cached ?? network;
		})(),
	);
});
