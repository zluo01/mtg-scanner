import assert from 'node:assert/strict';
import { test } from 'node:test';
import type { CardEntry } from '../../shared/api.ts';
import {
	activeChips,
	activeFilterCount,
	applyView,
	artistOptions,
	DEFAULT_VIEW,
	EMPTY_FILTERS,
	manaBucket,
	matchesFilters,
	matchesQuery,
	parseView,
	serializeView,
	setOptions,
} from '../src/lib/filters.ts';

let n = 0;
const card = (o: Partial<CardEntry>): CardEntry => ({
	card_id: `c${++n}`,
	scryfall_id: 'sf',
	name: 'Card',
	set_code: 'zen',
	collector_number: '1',
	foil: false,
	count: 1,
	created_at: `2026-01-${String(n).padStart(2, '0')}T00:00:00Z`,
	updated_at: '2026-01-01T00:00:00Z',
	has_photo: true,
	artist: 'Kieran Yanner',
	type_line: 'Creature — Kor Soldier',
	rarity: 'common',
	set_name: 'Zendikar',
	colors: 'W',
	mana_value: 2,
	released_at: '2009-10-02',
	...o,
});

const bolt = card({
	name: 'Lightning Bolt',
	set_code: 'm11',
	set_name: 'Magic 2011',
	type_line: 'Instant',
	colors: 'R',
	mana_value: 1,
	artist: 'Christopher Rush',
	released_at: '2010-07-16',
});
const kor = card({ name: 'Kor Outfitter', foil: true, count: 3 });
const gold = card({
	name: 'Gold Card',
	colors: 'WU',
	mana_value: 7,
	rarity: 'mythic',
	type_line: 'Legendary Artifact Creature — Golem',
});
const land = card({
	name: 'Forest',
	colors: '',
	mana_value: 0,
	type_line: 'Basic Land — Forest',
	rarity: 'common',
});
const unknown = card({
	name: 'Unknown',
	scryfall_id: null,
	set_code: null,
	artist: null,
	type_line: null,
	rarity: null,
	set_name: null,
	colors: null,
	mana_value: null,
	released_at: null,
});
const all = [bolt, kor, gold, land, unknown];

test('URL round trip', () => {
	const view = parseView(
		'?q=bolt&sort=mana&set=m11,zen&rarity=rare&type=creature,legendary&artist=Kieran+Yanner&color=W,M&mv=2,6%2B&foil=foil&copies=multi&attention=1',
	);
	assert.equal(view.query, 'bolt');
	assert.equal(view.sort, 'mana');
	assert.deepEqual(view.filters.sets, ['m11', 'zen']);
	assert.deepEqual(view.filters.types, ['creature', 'legendary']);
	assert.deepEqual(view.filters.artists, ['Kieran Yanner']);
	assert.deepEqual(view.filters.manaValues, ['2', '6+']);
	assert.equal(view.filters.foil, 'foil');
	assert.equal(view.filters.multiples, true);
	assert.equal(view.filters.attention, true);
	assert.equal(parseView(serializeView(view)).filters.colors.join(), 'W,M');
	assert.equal(serializeView(DEFAULT_VIEW), '');
	assert.equal(parseView('?sort=bogus&foil=maybe').sort, 'date');
	assert.equal(parseView('?foil=maybe').filters.foil, 'any');
});

test('facets combine with OR inside and AND across', () => {
	const f = { ...EMPTY_FILTERS, colors: ['W', 'R'] };
	assert.deepEqual(
		all.filter((c) => matchesFilters(c, f)).map((c) => c.name),
		['Lightning Bolt', 'Kor Outfitter', 'Gold Card'],
	);
	const both = { ...EMPTY_FILTERS, colors: ['W'], rarities: ['mythic'] };
	assert.deepEqual(
		all.filter((c) => matchesFilters(c, both)).map((c) => c.name),
		['Gold Card'],
	);
});

test('colorless and multicolor are their own options', () => {
	assert.deepEqual(
		all.filter((c) => matchesFilters(c, { ...EMPTY_FILTERS, colors: ['C'] })).map((c) => c.name),
		['Forest'],
	);
	assert.deepEqual(
		all.filter((c) => matchesFilters(c, { ...EMPTY_FILTERS, colors: ['M'] })).map((c) => c.name),
		['Gold Card'],
	);
});

test('type matches any word of the type line, legendary included', () => {
	const f = (types: string[]) =>
		all.filter((c) => matchesFilters(c, { ...EMPTY_FILTERS, types })).map((c) => c.name);
	assert.deepEqual(f(['creature']), ['Kor Outfitter', 'Gold Card']);
	assert.deepEqual(f(['legendary']), ['Gold Card']);
	assert.deepEqual(f(['instant', 'land']), ['Lightning Bolt', 'Forest']);
});

test('mana value buckets', () => {
	assert.equal(manaBucket(0), '0');
	assert.equal(manaBucket(2.5), '2');
	assert.equal(manaBucket(6), '6+');
	assert.equal(manaBucket(12), '6+');
	assert.equal(manaBucket(null), null);
	assert.deepEqual(
		all.filter((c) => matchesFilters(c, { ...EMPTY_FILTERS, manaValues: ['0', '6+'] })).map((c) => c.name),
		['Gold Card', 'Forest'],
	);
});

test('foil, multiples, attention', () => {
	assert.deepEqual(
		all.filter((c) => matchesFilters(c, { ...EMPTY_FILTERS, foil: 'foil' })).map((c) => c.name),
		['Kor Outfitter'],
	);
	assert.equal(all.filter((c) => matchesFilters(c, { ...EMPTY_FILTERS, foil: 'nonfoil' })).length, 4);
	assert.deepEqual(
		all.filter((c) => matchesFilters(c, { ...EMPTY_FILTERS, multiples: true })).map((c) => c.name),
		['Kor Outfitter'],
	);
	assert.deepEqual(
		all.filter((c) => matchesFilters(c, { ...EMPTY_FILTERS, attention: true })).map((c) => c.name),
		['Unknown'],
	);
});

test('attribute facets exclude unidentified cards', () => {
	for (const f of [
		{ sets: ['zen'] },
		{ rarities: ['common'] },
		{ types: ['creature'] },
		{ artists: ['Kieran Yanner'] },
		{ colors: ['C'] },
		{ manaValues: ['0'] },
	]) {
		assert.ok(!matchesFilters(unknown, { ...EMPTY_FILTERS, ...f }), JSON.stringify(f));
	}
});

test('search matches names, set codes, set names and set + number, inside the filters', () => {
	const names = (q: string, extra = {}) =>
		applyView(all, { ...DEFAULT_VIEW, query: q, filters: { ...EMPTY_FILTERS, ...extra } }).map((c) => c.name);
	assert.deepEqual(names('or'), ['Forest', 'Kor Outfitter']); // newest first
	assert.deepEqual(names('or', { types: ['land'] }), ['Forest']);
	assert.deepEqual(names('zen'), ['Forest', 'Gold Card', 'Kor Outfitter'], 'set code');
	assert.deepEqual(names('ZE'), ['Forest', 'Gold Card', 'Kor Outfitter'], 'start of a set code');
	assert.deepEqual(names('magic 20'), ['Lightning Bolt'], 'set name');
	assert.deepEqual(names('m11 1'), ['Lightning Bolt'], 'set + number');
	assert.deepEqual(names('M11/1'), ['Lightning Bolt'], 'any separator, any case');
	assert.deepEqual(names('m11 9'), [], 'number must start with it');
	assert.deepEqual(names('zen 1'), ['Forest', 'Gold Card', 'Kor Outfitter'], 'all three are zen #1');
	assert.deepEqual(names('zn'), [], 'a code must start with the query, not merely contain its letters');
	assert.equal(matchesQuery(unknown, 'zen'), false, 'unidentified cards have no set');
});

test('sorts', () => {
	const names = (sort: Parameters<typeof applyView>[1]['sort']) =>
		applyView(all, { ...DEFAULT_VIEW, sort }).map((c) => c.name);
	assert.deepEqual(names('date'), ['Unknown', 'Forest', 'Gold Card', 'Kor Outfitter', 'Lightning Bolt']);
	assert.deepEqual(names('name'), ['Forest', 'Gold Card', 'Kor Outfitter', 'Lightning Bolt', 'Unknown']);
	assert.deepEqual(names('release'), ['Lightning Bolt', 'Forest', 'Gold Card', 'Kor Outfitter', 'Unknown']);
	assert.deepEqual(names('mana'), ['Forest', 'Lightning Bolt', 'Kor Outfitter', 'Gold Card', 'Unknown']);
	assert.deepEqual(names('rarity'), ['Gold Card', 'Forest', 'Kor Outfitter', 'Lightning Bolt', 'Unknown']);
});

test('active chips remove one value each', () => {
	const f = {
		...EMPTY_FILTERS,
		colors: ['W', 'R'],
		sets: ['zen'],
		foil: 'foil' as const,
		manaValues: ['6+'],
	};
	const chips = activeChips(f, (code) => (code === 'zen' ? 'Zendikar' : code));
	assert.deepEqual(
		chips.map((c) => c.label),
		['Foil', 'White', 'Red', 'Mana value 6+', 'Zendikar'],
	);
	const red = chips.find((c) => c.key === 'colors:R');
	assert.deepEqual(red?.without, { ...f, colors: ['W'] });
	assert.equal(chips.find((c) => c.key === 'foil')?.without.foil, 'any');
	assert.deepEqual(
		activeChips(EMPTY_FILTERS, (c) => c),
		[],
	);
});

test('facet options and active count', () => {
	assert.deepEqual(setOptions(all), [
		{ value: 'm11', label: 'Magic 2011', count: 1 },
		{ value: 'zen', label: 'Zendikar', count: 3 },
	]);
	assert.deepEqual(
		artistOptions(all).map((o) => [o.label, o.count]),
		[
			['Kieran Yanner', 3],
			['Christopher Rush', 1],
		],
	);
	assert.equal(activeFilterCount(EMPTY_FILTERS), 0);
	assert.equal(activeFilterCount({ ...EMPTY_FILTERS, colors: ['W'], foil: 'foil', attention: true }), 3);
});
