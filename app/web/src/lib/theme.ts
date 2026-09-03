/**
 * Appearance: follow the OS, or force light or dark. The choice is kept in
 * localStorage and mirrored as `data-theme` on the root element; index.html
 * applies the stored value in an inline script before first paint so a
 * forced theme never flashes the other palette.
 */
import { createSignal } from 'solid-js';

export type Theme = 'system' | 'light' | 'dark';
export const THEME_KEY = 'mtg-theme';

export const THEME_OPTIONS: { value: Theme; label: string }[] = [
	{ value: 'system', label: 'System' },
	{ value: 'light', label: 'Light' },
	{ value: 'dark', label: 'Dark' },
];

/** Page backgrounds from styles.css, for the browser chrome (`theme-color`). */
const CHROME = { light: '#f0f2f5', dark: '#1c2027' } as const;

export const isTheme = (v: unknown): v is Theme => v === 'system' || v === 'light' || v === 'dark';

/** The palette a preference resolves to, given whether the OS is dark. */
export function resolveTheme(pref: Theme, systemDark: boolean): 'light' | 'dark' {
	if (pref === 'system') return systemDark ? 'dark' : 'light';
	return pref;
}

function load(): Theme {
	try {
		const v = localStorage.getItem(THEME_KEY);
		return isTheme(v) ? v : 'system';
	} catch {
		return 'system';
	}
}

const media = typeof window === 'undefined' ? null : window.matchMedia('(prefers-color-scheme: dark)');

const [theme, setThemeRaw] = createSignal<Theme>(typeof window === 'undefined' ? 'system' : load());

export { theme };

function apply(pref: Theme): void {
	if (!media) return;
	const root = document.documentElement;
	if (pref === 'system') delete root.dataset.theme;
	else root.dataset.theme = pref;
	document
		.querySelector('meta[name="theme-color"]')
		?.setAttribute('content', CHROME[resolveTheme(pref, media.matches)]);
}

export function setTheme(pref: Theme): void {
	setThemeRaw(pref);
	try {
		if (pref === 'system') localStorage.removeItem(THEME_KEY);
		else localStorage.setItem(THEME_KEY, pref);
	} catch {
		// private mode etc.; the in-memory value still applies
	}
	apply(pref);
}

// Keep the chrome colour right when the OS flips while following it.
media?.addEventListener('change', () => apply(theme()));
apply(theme());
