/**
 * Client-generated card ids.
 *
 * `crypto.randomUUID()` only exists in secure contexts (https:// or
 * localhost). The app is normally opened over plain http on a LAN, so fall
 * back to a v4 UUID built from `getRandomValues`, which is available
 * everywhere. Both forms satisfy the server's id rule.
 */
export function newCardId(): string {
	if (typeof crypto.randomUUID === 'function') return crypto.randomUUID();
	return uuidFromRandomValues();
}

/** RFC 4122 v4 UUID without `crypto.randomUUID`. */
export function uuidFromRandomValues(): string {
	const bytes = crypto.getRandomValues(new Uint8Array(16));
	bytes[6] = (bytes[6]! & 0x0f) | 0x40; // version 4
	bytes[8] = (bytes[8]! & 0x3f) | 0x80; // RFC 4122 variant
	const hex = Array.from(bytes, (b) => b.toString(16).padStart(2, '0')).join('');
	return `${hex.slice(0, 8)}-${hex.slice(8, 12)}-${hex.slice(12, 16)}-${hex.slice(16, 20)}-${hex.slice(20)}`;
}
