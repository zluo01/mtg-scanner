import assert from 'node:assert/strict';
import { test } from 'node:test';
import type { CardEntry } from '../../shared/api.ts';
import { findOwned } from '../src/lib/duplicates.ts';

const card = (id: string, o: Partial<CardEntry> = {}): CardEntry => ({
	card_id: id,
	scryfall_id: 'sf-bolt',
	name: 'Lightning Bolt',
	set_code: 'm11',
	collector_number: '146',
	foil: false,
	count: 1,
	created_at: '2026-01-02T00:00:00Z',
	updated_at: '2026-01-02T00:00:00Z',
	has_photo: true,
	artist: null,
	type_line: null,
	rarity: null,
	set_name: null,
	colors: null,
	mana_value: null,
	released_at: null,
	...o,
});

test('findOwned matches printing + foil only', () => {
	const cards = [card('a', { count: 3 }), card('b', { foil: true }), card('c', { scryfall_id: 'sf-other' })];
	assert.equal(findOwned(cards, 'sf-bolt', false)?.card_id, 'a');
	assert.equal(findOwned(cards, 'sf-bolt', true)?.card_id, 'b');
	assert.equal(findOwned(cards, 'sf-none', false), null);
	assert.equal(findOwned(cards, null, false), null);
});
