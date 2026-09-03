import assert from 'node:assert/strict';
import { test } from 'node:test';
import { CardStore } from '../src/db.ts';
import { HttpError } from '../src/errors.ts';

const newCard = (id: string, extra: Partial<Parameters<CardStore['insert']>[0]> = {}) => ({
	card_id: id,
	scryfall_id: `sf-${id}`,
	name: `Card ${id}`,
	set_code: 'm11',
	collector_number: '1',
	foil: false,
	...extra,
});

test('insert + get + list', () => {
	const db = CardStore.open(':memory:');
	const a = db.insert(newCard('a'));
	assert.equal(a.count, 1);
	assert.equal(a.foil, false);
	assert.equal(a.created_at, a.updated_at);
	db.insert(newCard('b'));
	assert.equal(db.get('a')?.name, 'Card a');
	assert.equal(db.get('nope'), null);
	assert.deepEqual(
		db.list().map((c) => c.card_id),
		['b', 'a'],
	);
	assert.equal(db.count(), 2);
	db.close();
});

test('insert accepts a count and a created_at, and validates the count', () => {
	const db = CardStore.open(':memory:');
	const a = db.insert(newCard('a', { count: 4, created_at: '2020-01-01T00:00:00.000Z' }));
	assert.equal(a.count, 4);
	assert.equal(a.created_at, '2020-01-01T00:00:00.000Z');
	assert.notEqual(a.updated_at, a.created_at);
	assert.throws(
		() => db.insert(newCard('b', { count: 0 })),
		(e: unknown) => e instanceof HttpError && e.status === 400,
	);
	assert.throws(() => db.insert(newCard('c', { count: 1.5 })));
	db.close();
});

test('transaction commits on return and rolls back on throw', () => {
	const db = CardStore.open(':memory:');
	assert.equal(
		db.transaction(() => {
			db.insert(newCard('a'));
			return 'done';
		}),
		'done',
	);
	assert.throws(() =>
		db.transaction(() => {
			db.insert(newCard('b'));
			throw new Error('boom');
		}),
	);
	assert.deepEqual(
		db.list().map((c) => c.card_id),
		['a'],
	);
	db.close();
});

test('the database itself refuses a second row for a printing + foil', () => {
	const db = CardStore.open(':memory:');
	assert.equal(db.printingRuleEnforced(), true);
	db.insert(newCard('a', { scryfall_id: 'sf-x' }));
	assert.throws(() => db.insert(newCard('b', { scryfall_id: 'sf-x' })), /UNIQUE/);
	db.insert(newCard('c', { scryfall_id: 'sf-x', foil: true })); // foil is a different card
	db.insert(newCard('p1', { scryfall_id: null }));
	db.insert(newCard('p2', { scryfall_id: null })); // placeholders are exempt
	assert.equal(db.count(), 4);
	db.close();
});

test('upsert folds into the owning row in one statement', () => {
	const db = CardStore.open(':memory:');
	const bolt = (id: string, extra: Partial<Parameters<CardStore['upsert']>[0]> = {}) =>
		newCard(id, { scryfall_id: 'sf-x', ...extra });
	const first = db.upsert(bolt('a', { created_at: '2020-01-01T00:00:00.000Z' }));
	assert.equal(first.merged, false);
	assert.equal(first.card.card_id, 'a');
	const second = db.upsert(bolt('b', { count: 2, created_at: '2025-01-01T00:00:00.000Z' }));
	assert.equal(second.merged, true);
	assert.equal(second.card.card_id, 'a');
	assert.equal(second.card.count, 3);
	assert.equal(second.card.created_at, '2025-01-01T00:00:00.000Z', 'newest added date');
	assert.equal(db.get('b'), null);
	const older = db.upsert(bolt('c', { created_at: '2010-01-01T00:00:00.000Z' }));
	assert.equal(older.card.created_at, '2025-01-01T00:00:00.000Z', 'an older addition keeps the newer date');
	assert.equal(older.card.count, 4);
	assert.equal(db.upsert(bolt('f', { foil: true })).merged, false);
	assert.equal(db.upsert(bolt('p', { scryfall_id: null })).merged, false);
	assert.equal(db.upsert(bolt('q', { scryfall_id: null })).merged, false);
	assert.equal(db.count(), 4);
	assert.throws(() => db.upsert(bolt('z', { count: 0 })));
	db.close();
});

test('a database from before the rule is opened without the index until it is folded', () => {
	const db = CardStore.open(':memory:', { enforcePrintingRule: false });
	assert.equal(db.printingRuleEnforced(), false);
	db.insert(newCard('a', { scryfall_id: 'sf-x' }));
	db.insert(newCard('b', { scryfall_id: 'sf-x' }));
	assert.equal(db.hasDuplicatePrintings(), true);
	assert.throws(() => db.enforcePrintingRule(), /UNIQUE/);
	db.merge('a', 'b');
	assert.equal(db.hasDuplicatePrintings(), false);
	db.enforcePrintingRule();
	assert.equal(db.printingRuleEnforced(), true);
	assert.throws(() => db.insert(newCard('c', { scryfall_id: 'sf-x' })), /UNIQUE/);
	db.close();
});

test('insert rejects duplicate ids', () => {
	const db = CardStore.open(':memory:');
	db.insert(newCard('a'));
	assert.throws(() => db.insert(newCard('a')));
	db.close();
});

test('placeholder rows allow null identification', () => {
	const db = CardStore.open(':memory:');
	const p = db.insert({
		card_id: 'p',
		scryfall_id: null,
		name: 'Unknown',
		set_code: null,
		collector_number: null,
		foil: true,
	});
	assert.equal(p.scryfall_id, null);
	assert.equal(p.foil, true);
	db.close();
});

test('findDuplicates matches printing + foil', () => {
	const db = CardStore.open(':memory:', { enforcePrintingRule: false });
	db.insert(newCard('a', { scryfall_id: 'sf-x' }));
	db.insert(newCard('b', { scryfall_id: 'sf-x' }));
	db.insert(newCard('c', { scryfall_id: 'sf-x', foil: true }));
	db.insert(newCard('d', { scryfall_id: 'sf-y' }));
	assert.deepEqual(
		db.findDuplicates('sf-x', false).map((c) => c.card_id),
		['a', 'b'],
	);
	assert.deepEqual(
		db.findDuplicates('sf-x', true).map((c) => c.card_id),
		['c'],
	);
	assert.deepEqual(db.findDuplicates('sf-z', false), []);
	db.close();
});

test('update is partial and bumps updated_at', async () => {
	const db = CardStore.open(':memory:');
	const before = db.insert(newCard('a'));
	await new Promise((r) => setTimeout(r, 2));
	const after = db.update('a', { count: 4 });
	assert.equal(after.count, 4);
	assert.equal(after.name, 'Card a');
	assert.equal(after.scryfall_id, 'sf-a');
	assert.notEqual(after.updated_at, before.updated_at);
	assert.equal(after.created_at, before.created_at);

	const fixed = db.update('a', {
		scryfall_id: 'sf-new',
		name: 'New',
		set_code: 'leg',
		collector_number: '9',
		foil: true,
	});
	assert.equal(fixed.scryfall_id, 'sf-new');
	assert.equal(fixed.collector_number, '9');
	assert.equal(fixed.foil, true);
	assert.equal(fixed.count, 4);

	assert.equal(db.update('a', {}).count, 4);
	db.close();
});

test('update of missing card throws 404', () => {
	const db = CardStore.open(':memory:');
	assert.throws(
		() => db.update('nope', { count: 2 }),
		(e: unknown) => e instanceof HttpError && e.status === 404,
	);
	db.close();
});

test('delete reports whether a row existed', () => {
	const db = CardStore.open(':memory:');
	db.insert(newCard('a'));
	assert.equal(db.delete('a'), true);
	assert.equal(db.delete('a'), false);
	assert.equal(db.count(), 0);
	db.close();
});

test('merge adds counts, takes the newer added date and deletes source', () => {
	const db = CardStore.open(':memory:', { enforcePrintingRule: false });
	db.insert(newCard('t', { count: 3, created_at: '2020-01-01T00:00:00.000Z' }));
	db.insert(newCard('s', { count: 5, created_at: '2025-06-01T00:00:00.000Z' }));
	db.insert(newCard('other'));

	const merged = db.merge('t', 's');
	assert.equal(merged.card_id, 't');
	assert.equal(merged.count, 8);
	assert.equal(merged.created_at, '2025-06-01T00:00:00.000Z');
	assert.equal(db.get('s'), null);
	assert.equal(db.get('other')?.count, 1);

	// An older source leaves the target's date alone.
	db.insert(newCard('older', { created_at: '2010-01-01T00:00:00.000Z' }));
	assert.equal(db.merge('t', 'older').created_at, '2025-06-01T00:00:00.000Z');
	db.close();
});

test('merge validates both sides and rolls back', () => {
	const db = CardStore.open(':memory:');
	db.insert(newCard('t'));
	db.insert(newCard('s', { scryfall_id: 'sf-other' }));
	assert.throws(
		() => db.merge('t', 't'),
		(e: unknown) => e instanceof HttpError && e.status === 400,
	);
	assert.throws(
		() => db.merge('t', 'missing'),
		(e: unknown) => e instanceof HttpError && e.status === 404,
	);
	assert.throws(
		() => db.merge('missing', 's'),
		(e: unknown) => e instanceof HttpError && e.status === 404,
	);
	// Nothing changed.
	assert.equal(db.get('t')?.count, 1);
	assert.equal(db.get('s')?.count, 1);
	db.close();
});
