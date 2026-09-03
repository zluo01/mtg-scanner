import assert from 'node:assert/strict';
import { test } from 'node:test';
import type { CardEntry } from '../../shared/api.ts';
import { csvField, libraryToCsv, libraryToMoxfieldCsv, moxfieldTimestamp, parseCsv } from '../src/csv.ts';

const card = (overrides: Partial<CardEntry>): CardEntry => ({
	card_id: 'id-1',
	scryfall_id: 'sf-1',
	name: 'Lightning Bolt',
	set_code: 'm11',
	collector_number: '146',
	foil: false,
	count: 1,
	created_at: '2026-01-01T00:00:00.000Z',
	updated_at: '2026-01-01T00:00:00.000Z',
	has_photo: true,
	artist: 'Christopher Rush',
	type_line: 'Instant',
	rarity: 'common',
	set_name: 'Magic 2011',
	colors: 'R',
	mana_value: 1,
	released_at: '2010-07-16',
	...overrides,
});

test('csvField quotes only when needed', () => {
	assert.equal(csvField('plain'), 'plain');
	assert.equal(csvField(null), '');
	assert.equal(csvField(3), '3');
	assert.equal(csvField(true), 'true');
	assert.equal(csvField('a,b'), '"a,b"');
	assert.equal(csvField('say "hi"'), '"say ""hi"""');
	assert.equal(csvField('line\nbreak'), '"line\nbreak"');
});

test('libraryToCsv writes a header and one row per card', () => {
	const csv = libraryToCsv([
		card({}),
		card({
			card_id: 'id-2',
			scryfall_id: null,
			name: 'Unknown',
			set_code: null,
			collector_number: null,
			artist: null,
			type_line: null,
			rarity: null,
			set_name: null,
			colors: null,
			mana_value: null,
			released_at: null,
			foil: true,
			count: 2,
		}),
	]);
	const lines = csv.trimEnd().split('\n');
	assert.equal(
		lines[0],
		'name,set_code,set_name,collector_number,rarity,artist,colors,mana_value,foil,count,scryfall_id,card_id,created_at',
	);
	assert.equal(
		lines[1],
		'Lightning Bolt,m11,Magic 2011,146,common,Christopher Rush,R,1,false,1,sf-1,id-1,2026-01-01T00:00:00.000Z',
	);
	assert.equal(lines[2], 'Unknown,,,,,,,,true,2,,id-2,2026-01-01T00:00:00.000Z');
});

test('libraryToCsv escapes names with commas', () => {
	const csv = libraryToCsv([card({ name: 'Ach! Hans, Run!' })]);
	assert.ok(csv.includes('"Ach! Hans, Run!",m11'));
});

test('parseCsv handles quotes, embedded commas and newlines, CRLF and a BOM', () => {
	assert.deepEqual(parseCsv('a,b\n1,2\n'), [
		['a', 'b'],
		['1', '2'],
	]);
	assert.deepEqual(parseCsv('﻿"Count","Name"\r\n"1","Ach! Hans, Run!"\r\n'), [
		['Count', 'Name'],
		['1', 'Ach! Hans, Run!'],
	]);
	assert.deepEqual(parseCsv('"say ""hi""","multi\nline",tail'), [['say "hi"', 'multi\nline', 'tail']]);
	assert.deepEqual(parseCsv('a,,c\n\n,\n'), [
		['a', '', 'c'],
		['', ''],
	]);
	assert.deepEqual(parseCsv(''), []);
	assert.deepEqual(parseCsv('x'), [['x']]);
});

test('moxfieldTimestamp converts ISO to Moxfield form', () => {
	assert.equal(moxfieldTimestamp('2026-09-01T00:24:01.653Z'), '2026-09-01 00:24:01.653000');
	assert.equal(moxfieldTimestamp('not a date'), '');
});

test('libraryToMoxfieldCsv writes Moxfield layout and skips nameless placeholders', () => {
	const csv = libraryToMoxfieldCsv([
		card({ count: 3, foil: true, updated_at: '2026-09-01T00:24:01.653Z' }),
		card({ card_id: 'p', scryfall_id: null, name: 'Unknown', set_code: null, collector_number: null }),
		card({ card_id: 'u', scryfall_id: null, name: 'Mystery Card', set_code: 'xyz', collector_number: '7' }),
		card({ card_id: 'dfc', name: 'Fell the Profane // Fell Mire', set_code: 'mh3', collector_number: '244' }),
	]);
	const lines = csv.trimEnd().split('\n');
	assert.equal(
		lines[0],
		'"Count","Tradelist Count","Name","Edition","Condition","Language","Foil","Tags","Last Modified","Collector Number","Alter","Proxy","Purchase Price"',
	);
	assert.equal(
		lines[1],
		'"3","3","Lightning Bolt","m11","Near Mint","English","foil","","2026-09-01 00:24:01.653000","146","False","False",""',
	);
	assert.equal(lines.length, 4);
	assert.ok(lines[2]?.startsWith('"1","1","Mystery Card","xyz",'));
	assert.ok(
		lines[3]?.startsWith('"1","1","Fell the Profane // Fell Mire","mh3",'),
		'names pass through as given',
	);
	// What we write, we can read back.
	const rows = parseCsv(csv);
	assert.equal(rows.length, 4);
	assert.equal(rows[1]?.[2], 'Lightning Bolt');
});
