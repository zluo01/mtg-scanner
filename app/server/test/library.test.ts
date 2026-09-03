import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';
import { test } from 'node:test';
import type { NewCard } from '../src/db.ts';
import { HttpError } from '../src/errors.ts';
import { ImageStore } from '../src/images.ts';
import { Library } from '../src/library.ts';
import { stores } from './helpers.ts';

const bolt = (id: string, extra: Partial<NewCard> = {}): NewCard => ({
	card_id: id,
	scryfall_id: 'sf-bolt',
	name: 'Lightning Bolt',
	set_code: 'm11',
	collector_number: '146',
	foil: false,
	...extra,
});
const OLD = '2020-01-01T00:00:00.000Z';
const PHOTO_A = new Uint8Array([1, 2, 3]);
const PHOTO_B = new Uint8Array([9, 9, 9]);

test('add folds into the card that owns the printing + foil and bumps its added date', async () => {
	const s = await stores();
	try {
		const first = s.library.add(bolt('a', { created_at: OLD }));
		assert.equal(first.merged, false);
		const second = s.library.add(bolt('b', { count: 2 }));
		assert.equal(second.merged, true);
		assert.equal(second.card.card_id, 'a');
		assert.equal(second.card.count, 3);
		assert.ok(second.card.created_at > OLD, 'the survivor takes the newest added date');
		assert.equal(s.cards.count(), 1);
		assert.equal(s.cards.get('b'), null);

		// Foil is part of the key; placeholders never fold.
		assert.equal(s.library.add(bolt('c', { foil: true })).merged, false);
		const placeholder = (id: string): NewCard => ({
			card_id: id,
			scryfall_id: null,
			name: 'Unknown',
			set_code: null,
			collector_number: null,
			foil: false,
		});
		s.library.add(placeholder('p1'));
		s.library.add(placeholder('p2'));
		assert.equal(s.cards.count(), 4);
	} finally {
		await s.cleanup();
	}
});

test('addScan stores the photo, keeps an existing photo, and backs out on failure', async () => {
	const s = await stores();
	try {
		await s.library.addScan(bolt('a'), PHOTO_A);
		assert.deepEqual(new Uint8Array(await readFile(s.images.pathFor('a'))), PHOTO_A);
		const folded = await s.library.addScan(bolt('b'), PHOTO_B);
		assert.equal(folded.card.card_id, 'a');
		assert.equal(folded.card.count, 2);
		assert.deepEqual(new Uint8Array(await readFile(s.images.pathFor('a'))), PHOTO_A, 'existing photo kept');
		assert.equal(s.images.has('b'), false);

		// A card without a photo (an import) gains the scan's photo.
		s.library.add(bolt('imp', { foil: true }));
		const gained = await s.library.addScan(bolt('scan', { foil: true }), PHOTO_B);
		assert.equal(gained.card.card_id, 'imp');
		assert.deepEqual(new Uint8Array(await readFile(s.images.pathFor('imp'))), PHOTO_B);

		// Write failure: the copy is taken back out, whether it folded or was new.
		const failing = new (class extends ImageStore {
			override async write(): Promise<void> {
				throw new Error('disk full');
			}
		})(s.images.dir);
		const lib = new Library(s.cards, failing);
		s.library.add(bolt('nophoto', { scryfall_id: 'sf-np' }));
		await assert.rejects(lib.addScan(bolt('x', { scryfall_id: 'sf-np' }), PHOTO_A), /disk full/);
		assert.equal(s.cards.get('nophoto')?.count, 1, 'the folded copy was removed again');
		await assert.rejects(lib.addScan(bolt('y', { scryfall_id: 'sf-new' }), PHOTO_A), /disk full/);
		assert.equal(s.cards.get('y'), null, 'the fresh row was removed again');
	} finally {
		await s.cleanup();
	}
});

test('change folds into a twin and moves the photo when the twin has none', async () => {
	const s = await stores();
	try {
		s.library.add(bolt('nonfoil', { created_at: OLD }));
		await s.library.addScan(bolt('foil', { foil: true }), PHOTO_A);
		const survivor = await s.library.change('foil', { foil: false });
		assert.equal(survivor.card_id, 'nonfoil');
		assert.equal(survivor.count, 2);
		assert.ok(survivor.created_at > OLD);
		assert.equal(s.cards.get('foil'), null);
		assert.equal(s.images.has('foil'), false);
		assert.deepEqual(new Uint8Array(await readFile(s.images.pathFor('nonfoil'))), PHOTO_A);

		// No twin: an ordinary update. A twin with a photo keeps its own.
		const alone = await s.library.change('nonfoil', { count: 5 });
		assert.equal(alone.card_id, 'nonfoil');
		assert.equal(alone.count, 5);
		await s.library.addScan(bolt('other', { foil: true }), PHOTO_B);
		const kept = await s.library.change('other', { foil: false });
		assert.equal(kept.card_id, 'nonfoil');
		assert.deepEqual(new Uint8Array(await readFile(s.images.pathFor('nonfoil'))), PHOTO_A);
		assert.equal(s.images.has('other'), false);
		await assert.rejects(
			s.library.change('missing', { count: 1 }),
			(e: unknown) => e instanceof HttpError && e.status === 404,
		);
	} finally {
		await s.cleanup();
	}
});

test('remove deletes the row with its photo', async () => {
	const s = await stores();
	try {
		await s.library.addScan(bolt('a', { count: 2 }), PHOTO_A);
		assert.equal(await s.library.remove('a'), true);
		assert.equal(s.cards.get('a'), null);
		assert.equal(s.images.has('a'), false);
		assert.equal(await s.library.remove('a'), false);
	} finally {
		await s.cleanup();
	}
});

test('dedupeAll folds rows written before the rule, oldest id surviving with the newest date', async () => {
	const s = await stores();
	try {
		s.cards.insert(bolt('old', { created_at: OLD }));
		s.cards.insert(bolt('mid', { count: 2, created_at: '2023-01-01T00:00:00.000Z' }));
		await s.images.write('mid', PHOTO_A);
		s.cards.insert(bolt('new', { created_at: '2026-01-01T00:00:00.000Z' }));
		s.cards.insert(bolt('foil', { foil: true }));
		assert.equal(await s.library.dedupeAll(), 2);
		const kept = s.cards.get('old');
		assert.equal(kept?.count, 4);
		assert.equal(kept?.created_at, '2026-01-01T00:00:00.000Z');
		assert.equal(s.cards.count(), 2);
		assert.equal(s.images.has('old'), true);
		assert.equal(s.images.has('mid'), false);
		assert.equal(await s.library.dedupeAll(), 0);
	} finally {
		await s.cleanup();
	}
});
