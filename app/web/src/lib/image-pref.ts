/** Global "show my photos instead of Scryfall art" preference. */
import { createSignal } from 'solid-js';

const KEY = 'mtg-show-user-images';

function load(): boolean {
	try {
		return localStorage.getItem(KEY) === 'true';
	} catch {
		return false;
	}
}

const [showUserImages, setShowUserImagesRaw] = createSignal(load());

export { showUserImages };

export function toggleUserImages(): void {
	const next = !showUserImages();
	setShowUserImagesRaw(next);
	try {
		localStorage.setItem(KEY, String(next));
	} catch {
		// private mode etc.; the in-memory value still applies
	}
}
