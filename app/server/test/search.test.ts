import assert from 'node:assert/strict';
import { test } from 'node:test';
import { CardCatalog } from '../src/metadata.ts';
import { NameSearch, parseSetNumber } from '../src/search.ts';
import { meta } from './helpers.ts';

const cards = [
	meta({ scryfall_id: '1', name: 'Lightning Bolt', set_code: 'm11', collector_number: '146' }),
	meta({ scryfall_id: '2', name: 'Lightning Bolt', set_code: 'lea', collector_number: '161' }),
	meta({ scryfall_id: '3', name: 'Chain Lightning', set_code: 'leg', collector_number: '1' }),
	meta({ scryfall_id: '4', name: 'Lightning Helix', set_code: 'rav', collector_number: '10' }),
	meta({ scryfall_id: '5', name: 'Bolt Bend', set_code: 'war', collector_number: '2' }),
	meta({ scryfall_id: '6', name: 'Thunderbolt', set_code: 'wth', collector_number: '3' }),
	meta({ scryfall_id: '7', name: 'Lightning Bolt', set_code: 'm11', collector_number: '9' }),
	meta({ scryfall_id: '8', name: 'Fury Sliver', set_code: 'plst', collector_number: 'TSP-157' }),
	meta({
		scryfall_id: '9',
		name: "Azusa's Many Journeys",
		set_code: 'neo',
		collector_number: '172',
		lang: 'ja',
	}),
	meta({ scryfall_id: '10', name: "Azusa's Many Journeys", set_code: 'neo', collector_number: '172' }),
	meta({ scryfall_id: '11', name: 'Command Tower', set_code: 'pmei', collector_number: '2026-1' }),
];

test('ranks exact > prefix > word start > substring, then name/set/number', () => {
	const s = new NameSearch(cards);
	const ids = s.search('bolt').map((c) => c.scryfall_id);
	// prefix: Bolt Bend; word-start: Lightning Bolt x3 ordered by set (lea < m11)
	// then numeric collector number (#9 < #146); substring: Thunderbolt.
	assert.deepEqual(ids, ['5', '2', '7', '1', '6']);
	assert.equal(s.search('lightning bolt')[0]!.scryfall_id, '2');
});

test('is case-insensitive, trims, and enforces the minimum length', () => {
	const s = new NameSearch(cards);
	assert.equal(s.search('  LIGHTNING ').length, 5);
	assert.deepEqual(s.search('l'), []);
	assert.deepEqual(s.search('zzz'), []);
});

test('respects the result limit', () => {
	const s = new NameSearch(cards);
	assert.equal(s.search('lightning', 2).length, 2);
	assert.equal(s.size, cards.length);
});

test('parseSetNumber accepts the usual separators and awkward numbers', () => {
	assert.deepEqual(parseSetNumber('neo 172'), { setCode: 'neo', collectorNumber: '172' });
	assert.deepEqual(parseSetNumber('  NEO/172 '), { setCode: 'neo', collectorNumber: '172' });
	assert.deepEqual(parseSetNumber('neo#172'), { setCode: 'neo', collectorNumber: '172' });
	assert.deepEqual(parseSetNumber('sld-2179'), { setCode: 'sld', collectorNumber: '2179' });
	assert.deepEqual(parseSetNumber('plst TSP-157'), { setCode: 'plst', collectorNumber: 'TSP-157' });
	assert.deepEqual(parseSetNumber('pmei 2026-1'), { setCode: 'pmei', collectorNumber: '2026-1' });
	assert.deepEqual(parseSetNumber('war 213★'), { setCode: 'war', collectorNumber: '213★' });
	assert.deepEqual(parseSetNumber('unf 1012a'), { setCode: 'unf', collectorNumber: '1012a' });
	assert.equal(parseSetNumber('lightning bolt'), null);
	assert.equal(parseSetNumber('bolt'), null);
	assert.equal(parseSetNumber('m11'), null);
});

test('set + number lists that printing first, every language, then name matches', () => {
	const s = new NameSearch(cards, new CardCatalog(cards));
	assert.deepEqual(
		s.search('neo 172').map((c) => c.scryfall_id),
		['10', '9'],
		'English printing before the Japanese one',
	);
	assert.equal(s.search('PLST tsp-157')[0]?.scryfall_id, '8', 'number matched case-insensitively');
	assert.equal(s.search('m11/146')[0]?.scryfall_id, '1');
	assert.equal(s.search('pmei 2026-1')[0]?.scryfall_id, '11');
	assert.deepEqual(s.search('m11 999'), [], 'unknown number: nothing pretends to match');
	// Without a catalog the query is just a name search.
	assert.deepEqual(new NameSearch(cards).search('neo 172'), []);
});
