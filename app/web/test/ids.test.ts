import assert from 'node:assert/strict';
import { test } from 'node:test';
import { newCardId, uuidFromRandomValues } from '../src/lib/ids.ts';

const V4 = /^[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$/;
/** Mirrors the server's card_id rule (ids double as file names). */
const SERVER_RULE = /^[A-Za-z0-9][A-Za-z0-9_-]{0,63}$/;

test('fallback produces well-formed, unique v4 UUIDs', () => {
	const ids = new Set(Array.from({ length: 500 }, () => uuidFromRandomValues()));
	assert.equal(ids.size, 500);
	for (const id of ids) {
		assert.match(id, V4);
		assert.match(id, SERVER_RULE);
	}
});

test('newCardId works whether or not crypto.randomUUID exists', () => {
	assert.match(newCardId(), V4);
	const original = crypto.randomUUID;
	// Simulate an insecure context (plain http on a LAN), where the API is absent.
	Object.defineProperty(crypto, 'randomUUID', { value: undefined, configurable: true });
	try {
		assert.match(newCardId(), V4);
	} finally {
		Object.defineProperty(crypto, 'randomUUID', { value: original, configurable: true });
	}
});
