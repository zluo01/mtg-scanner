/** Register the service worker after first paint (production only). */
export function registerServiceWorker(): void {
	if (!import.meta.env.PROD || !('serviceWorker' in navigator)) return;
	const register = () => {
		navigator.serviceWorker.register('/sw.js', { scope: '/' }).catch((err) => {
			console.warn('Service worker registration failed:', err);
		});
	};
	if (document.readyState === 'complete') register();
	else window.addEventListener('load', register, { once: true });
}
