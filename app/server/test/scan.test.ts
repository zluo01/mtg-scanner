import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';
import { test } from 'node:test';
import { HttpError } from '../src/errors.ts';
import { ImageStore } from '../src/images.ts';
import { Library } from '../src/library.ts';
import {
	AMBIGUOUS_THRESHOLD,
	addScannedCard,
	CONFIDENT_THRESHOLD,
	classifyConfidence,
	identifyScan,
} from '../src/scan.ts';
import { FAKE_JPEG, fixedIdentifier, meta, stores } from './helpers.ts';

test('classifyConfidence thresholds are inclusive', () => {
	assert.equal(classifyConfidence(0.95), 'CONFIDENT');
	assert.equal(classifyConfidence(CONFIDENT_THRESHOLD), 'CONFIDENT');
	assert.equal(classifyConfidence(0.5), 'AMBIGUOUS');
	assert.equal(classifyConfidence(AMBIGUOUS_THRESHOLD), 'AMBIGUOUS');
	assert.equal(classifyConfidence(0.39), 'NO_MATCH');
	assert.equal(classifyConfidence(0), 'NO_MATCH');
});

test('identifyScan reports the top hit and every candidate without storing anything', async () => {
	const s = await stores();
	try {
		const confident = await identifyScan(fixedIdentifier([{ card: meta(), similarity: 0.95 }]), FAKE_JPEG);
		assert.deepEqual(
			{ confidence: confident.confidence, similarity: confident.similarity, n: confident.candidates.length },
			{ confidence: 'CONFIDENT', similarity: 0.95, n: 1 },
		);
		assert.equal(confident.candidates[0]?.name, 'Lightning Bolt');

		const ambiguous = await identifyScan(
			fixedIdentifier([
				{ card: meta({ scryfall_id: 'a', name: 'Card A' }), similarity: 0.55 },
				{ card: meta({ scryfall_id: 'b', name: 'Card B' }), similarity: 0.5 },
				{ card: meta({ scryfall_id: 'c', name: 'Card C' }), similarity: 0.45 },
			]),
			FAKE_JPEG,
		);
		assert.equal(ambiguous.confidence, 'AMBIGUOUS');
		assert.deepEqual(
			ambiguous.candidates.map((c) => c.scryfall_id),
			['a', 'b', 'c'],
		);

		const weak = await identifyScan(
			fixedIdentifier([{ card: meta({ name: 'Low' }), similarity: 0.3 }]),
			FAKE_JPEG,
		);
		assert.equal(weak.confidence, 'NO_MATCH');
		assert.equal(weak.candidates[0]?.name, 'Low', 'weak hits are still offered');

		const nothing = await identifyScan(fixedIdentifier([]), FAKE_JPEG);
		assert.deepEqual(nothing, { confidence: 'NO_MATCH', similarity: 0, candidates: [] });

		assert.equal(s.cards.count(), 0);
	} finally {
		await s.cleanup();
	}
});

test('addScannedCard writes the confirmed printing with its photo, from the index', async () => {
	const s = await stores();
	try {
		const res = await addScannedCard(s, {
			cardId: 'new-1',
			scryfallId: 'sf-bolt',
			foil: true,
			image: FAKE_JPEG,
		});
		assert.equal(res.merged, false);
		assert.equal(res.card.card_id, 'new-1');
		assert.equal(res.card.name, 'Lightning Bolt');
		assert.equal(res.card.set_code, 'm11');
		assert.equal(res.card.foil, true);
		assert.equal(res.card.artist, 'Christopher Rush');
		assert.equal(res.card.has_photo, true);
		assert.deepEqual(new Uint8Array(await readFile(s.images.pathFor('new-1'))), FAKE_JPEG);

		const placeholder = await addScannedCard(s, {
			cardId: 'nm-1',
			scryfallId: null,
			foil: false,
			image: FAKE_JPEG,
		});
		assert.equal(placeholder.card.scryfall_id, null);
		assert.equal(placeholder.card.name, 'Unknown');
		assert.ok(await s.images.exists('nm-1'));

		await assert.rejects(
			addScannedCard(s, { cardId: 'x', scryfallId: 'sf-nope', foil: false, image: FAKE_JPEG }),
			(e: unknown) => e instanceof HttpError && e.status === 400,
		);
		await assert.rejects(
			addScannedCard(s, { cardId: 'new-1', scryfallId: 'sf-bolt', foil: false, image: FAKE_JPEG }),
			(e: unknown) => e instanceof HttpError && e.status === 409,
		);
		assert.equal(s.cards.count(), 2);
	} finally {
		await s.cleanup();
	}
});

test('a scan of a printing already owned folds into it; foil is a different card', async () => {
	const s = await stores();
	try {
		s.cards.insert({
			card_id: 'existing',
			scryfall_id: 'sf-bolt',
			name: 'Lightning Bolt',
			set_code: 'm11',
			collector_number: '146',
			foil: false,
			count: 3,
			created_at: '2020-01-01T00:00:00.000Z',
		});
		const dup = await addScannedCard(s, {
			cardId: 'dup',
			scryfallId: 'sf-bolt',
			foil: false,
			image: FAKE_JPEG,
		});
		assert.equal(dup.merged, true);
		assert.equal(dup.card.card_id, 'existing');
		assert.equal(dup.card.count, 4);
		assert.ok(dup.card.created_at > '2020-01-01T00:00:00.000Z', 'newest added date');
		assert.equal(dup.card.has_photo, true, 'the imported card gained the scan photo');
		assert.equal(s.cards.get('dup'), null);
		assert.equal(s.cards.count(), 1);

		const foil = await addScannedCard(s, {
			cardId: 'foil',
			scryfallId: 'sf-bolt',
			foil: true,
			image: FAKE_JPEG,
		});
		assert.equal(foil.merged, false);
		assert.equal(foil.card.card_id, 'foil');
		assert.equal(s.cards.count(), 2);
	} finally {
		await s.cleanup();
	}
});

test('rolls back the row when the photo cannot be written', async () => {
	const s = await stores();
	try {
		const failing = new (class extends ImageStore {
			override async write(): Promise<void> {
				throw new Error('disk full');
			}
		})(s.images.dir);
		await assert.rejects(
			addScannedCard(
				{ library: new Library(s.cards, failing), catalog: s.catalog },
				{ cardId: 'rb', scryfallId: 'sf-bolt', foil: false, image: FAKE_JPEG },
			),
			/disk full/,
		);
		assert.equal(s.cards.get('rb'), null);
	} finally {
		await s.cleanup();
	}
});
