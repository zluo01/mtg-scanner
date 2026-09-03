import assert from 'node:assert/strict';
import { test } from 'node:test';
import { HttpError } from '../src/errors.ts';
import { importMoxfield, parseMoxfieldCsv, parseMoxfieldTimestamp } from '../src/moxfield.ts';
import { meta, stores } from './helpers.ts';

const HEADER =
	'"Count","Tradelist Count","Name","Edition","Condition","Language","Foil","Tags","Last Modified","Collector Number","Alter","Proxy","Purchase Price"';
const row = (
	count: number,
	name: string,
	set: string,
	num: string,
	foil = '',
	lang = 'English',
	modified = '2026-09-01 00:24:01.653000',
) =>
	`"${count}","${count}","${name}","${set}","Near Mint","${lang}","${foil}","","${modified}","${num}","False","False",""`;
const file = (...rows: string[]) => `﻿${[HEADER, ...rows].join('\r\n')}\r\n`;

const PRINTINGS = [
	meta(),
	meta({ scryfall_id: 'sf-bolt-ja', lang: 'ja' }),
	meta({ scryfall_id: 'sf-chain', name: 'Chain Lightning', set_code: 'leg', collector_number: '1' }),
];

test('parseMoxfieldCsv reads the export layout', () => {
	const rows = parseMoxfieldCsv(
		file(row(2, 'Lightning Bolt', 'M11', '146'), row(1, 'Abrade', 'sld', '2179', 'etched', 'Japanese', '')),
	);
	assert.deepEqual(rows, [
		{
			count: 2,
			name: 'Lightning Bolt',
			set_code: 'm11',
			collector_number: '146',
			foil: false,
			lang: 'en',
			modified_at: '2026-09-01T00:24:01.653Z',
		},
		{
			count: 1,
			name: 'Abrade',
			set_code: 'sld',
			collector_number: '2179',
			foil: true,
			lang: 'ja',
			modified_at: null,
		},
	]);
	// Only the three matching columns are required; count defaults to 1.
	assert.deepEqual(parseMoxfieldCsv('Name,Edition,Collector Number\nBolt,m11,146\n')[0]?.count, 1);
});

test('parseMoxfieldCsv rejects other files and bad rows', () => {
	const bad = (text: string, pattern: RegExp) =>
		assert.throws(
			() => parseMoxfieldCsv(text),
			(e: unknown) => e instanceof HttpError && e.status === 400 && pattern.test(e.message),
		);
	bad('', /empty/);
	bad('name,set_code,count\nBolt,m11,1\n', /Moxfield/);
	bad(file(row(0, 'Bolt', 'm11', '146')), /Line 2: count/);
	bad(file(row(1, '', 'm11', '146')), /Line 2: missing card name/);
});

test('parseMoxfieldTimestamp', () => {
	assert.equal(parseMoxfieldTimestamp('2026-09-01 00:24:01.653000'), '2026-09-01T00:24:01.653Z');
	assert.equal(parseMoxfieldTimestamp('2026-09-01 00:24:01'), '2026-09-01T00:24:01.000Z');
	assert.equal(parseMoxfieldTimestamp(''), null);
	assert.equal(parseMoxfieldTimestamp('yesterday'), null);
});

test('import matches printings by set + number, keeps unmatched rows as unidentified', async () => {
	const s = await stores(PRINTINGS);
	try {
		const result = importMoxfield(
			s,
			parseMoxfieldCsv(
				file(
					row(2, 'lightning bolt', 'M11', '146', 'foil'),
					row(1, 'Lightning Bolt', 'm11', '146', '', 'Japanese'),
					row(1, 'Some Promo', 'pxyz', '9'),
				),
			),
			'set',
		);
		assert.deepEqual(result, {
			rows: 3,
			added: 3,
			updated: 0,
			unmatched: 1,
			unmatched_names: ['Some Promo'],
		});
		const lib = s.cards.list();
		assert.equal(lib.length, 3);
		const foil = lib.find((c) => c.foil);
		assert.equal(foil?.scryfall_id, 'sf-bolt');
		assert.equal(foil?.name, 'Lightning Bolt'); // canonical name from the index
		assert.equal(foil?.count, 2);
		assert.equal(foil?.created_at, '2026-09-01T00:24:01.653Z');
		assert.equal(lib.find((c) => !c.foil && c.scryfall_id)?.scryfall_id, 'sf-bolt-ja'); // language preferred
		const unknown = lib.find((c) => c.scryfall_id === null);
		assert.equal(unknown?.name, 'Some Promo');
		assert.equal(unknown?.set_code, 'pxyz');
		assert.equal(unknown?.collector_number, '9');
	} finally {
		await s.cleanup();
	}
});

test('a renumbered printing resolves by either number and stays one card', async () => {
	// Scryfall renumbered some promos (2026-01 -> 2026-1); the refreshed index carries both rows.
	const s = await stores([
		meta({ scryfall_id: 'sf-tower', name: 'Command Tower', set_code: 'pmei', collector_number: '2026-01' }),
		meta({ scryfall_id: 'sf-tower', name: 'Command Tower', set_code: 'pmei', collector_number: '2026-1' }),
	]);
	try {
		assert.equal(s.catalog.findPrinting('pmei', '2026-1')?.scryfall_id, 'sf-tower');
		assert.equal(s.catalog.findPrinting('pmei', '2026-01')?.scryfall_id, 'sf-tower');
		const r = importMoxfield(
			s,
			parseMoxfieldCsv(
				file(
					row(1, 'Command Tower', 'pmei', '2026-1', 'foil'),
					row(1, 'Command Tower', 'pmei', '2026-01', 'foil'),
				),
			),
			'set',
		);
		assert.deepEqual([r.added, r.unmatched], [1, 0]);
		assert.equal(s.cards.list()[0]?.count, 2);
	} finally {
		await s.cleanup();
	}
});

test('fullName joins the faces of a double-faced card and ignores duplicate rows', async () => {
	const s = await stores([
		meta({ scryfall_id: 'sf-dfc', name: 'Fell the Profane', set_code: 'mh3', collector_number: '244' }),
		meta({ scryfall_id: 'sf-dfc', name: 'Fell Mire', set_code: 'mh3', collector_number: '244' }),
		meta({ scryfall_id: 'sf-tower', name: 'Command Tower', set_code: 'pmei', collector_number: '2026-01' }),
		meta({ scryfall_id: 'sf-tower', name: 'Command Tower', set_code: 'pmei', collector_number: '2026-1' }),
	]);
	try {
		assert.equal(s.catalog.fullName('sf-dfc'), 'Fell the Profane // Fell Mire');
		assert.equal(s.catalog.fullName('sf-tower'), 'Command Tower');
		assert.equal(s.catalog.fullName('nope'), undefined);
		assert.equal(s.catalog.fullName(null), undefined);
		// The library row keeps the front face; the catalog's first row wins.
		importMoxfield(s, parseMoxfieldCsv(file(row(1, 'Fell the Profane // Fell Mire', 'mh3', '244'))), 'set');
		assert.equal(s.cards.list()[0]?.name, 'Fell the Profane');
	} finally {
		await s.cleanup();
	}
});

test('re-importing in set mode is idempotent; add mode stacks copies', async () => {
	const s = await stores(PRINTINGS);
	try {
		const text = file(row(2, 'Lightning Bolt', 'm11', '146'), row(1, 'Chain Lightning', 'leg', '1'));
		importMoxfield(s, parseMoxfieldCsv(text), 'set');
		const again = importMoxfield(s, parseMoxfieldCsv(text), 'set');
		assert.deepEqual(again, { rows: 2, added: 0, updated: 2, unmatched: 0, unmatched_names: [] });
		assert.deepEqual(
			s.cards.list().map((c) => c.count),
			[2, 1].sort(),
		);

		const more = importMoxfield(
			s,
			parseMoxfieldCsv(
				file(row(3, 'Lightning Bolt', 'm11', '146', '', 'English', '2026-09-02 10:00:00.000000')),
			),
			'add',
		);
		assert.equal(more.updated, 1);
		const grown = s.cards.list().find((c) => c.scryfall_id === 'sf-bolt');
		assert.equal(grown?.count, 5);
		assert.equal(grown?.created_at, '2026-09-02T10:00:00.000Z', 'more copies: newest added date');

		// Set mode with a lower count reduces to the file's count.
		importMoxfield(s, parseMoxfieldCsv(file(row(1, 'Lightning Bolt', 'm11', '146'))), 'set');
		assert.equal(s.cards.list().find((c) => c.scryfall_id === 'sf-bolt')?.count, 1);
		assert.equal(s.cards.count(), 2);
	} finally {
		await s.cleanup();
	}
});

test('two rows for the same card in one file add up, even in set mode', async () => {
	const s = await stores([meta()]);
	try {
		// Both rows resolve to the English printing (no Japanese row in this catalog).
		const text = file(
			row(1, 'Lightning Bolt', 'm11', '146'),
			row(2, 'Lightning Bolt', 'm11', '146', '', 'Japanese'),
		);
		const r = importMoxfield(s, parseMoxfieldCsv(text), 'set');
		assert.equal(r.added, 1);
		assert.equal(s.cards.list()[0]?.count, 3);
		importMoxfield(s, parseMoxfieldCsv(text), 'set');
		assert.equal(s.cards.list()[0]?.count, 3);
	} finally {
		await s.cleanup();
	}
});

test('an existing scanned card absorbs the file row', async () => {
	const s = await stores([meta()]);
	try {
		s.cards.insert({
			card_id: 'scan-1',
			scryfall_id: 'sf-bolt',
			name: 'Lightning Bolt',
			set_code: 'm11',
			collector_number: '146',
			foil: false,
		});
		const r = importMoxfield(s, parseMoxfieldCsv(file(row(4, 'Lightning Bolt', 'm11', '146'))), 'set');
		assert.deepEqual([r.added, r.updated], [0, 1]);
		assert.equal(s.cards.get('scan-1')?.count, 4);
		assert.equal(s.cards.count(), 1);
	} finally {
		await s.cleanup();
	}
});
