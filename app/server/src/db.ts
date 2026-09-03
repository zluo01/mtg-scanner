/**
 * SQLite-backed card library using Node's built-in `node:sqlite`.
 *
 * A row holds only what identifies a card and what the user chose: the
 * printing id, a readable name/set/number fallback, foil, and count.
 * Everything else about the printing (artist, type, colours, ...) is
 * attached at read time from the index metadata (see `metadata.ts`).
 *
 * The statements live in `sql/*.sql` (loaded by `sql.ts`); this class binds
 * parameters and maps rows. All methods are synchronous: the library is
 * tiny (thousands of rows at most) and single-user, so a blocking call
 * costs microseconds and keeps the call sites simple.
 */
import { DatabaseSync } from 'node:sqlite';
import type { CardEntry, UpdateCardRequest } from '../../shared/api.ts';
import { badRequest, notFound } from './errors.ts';
import { SQL } from './sql.ts';

/** The stored fields of a card (`CardEntry` minus the printing attributes). */
export type StoredCard = Pick<
	CardEntry,
	| 'card_id'
	| 'scryfall_id'
	| 'name'
	| 'set_code'
	| 'collector_number'
	| 'foil'
	| 'count'
	| 'created_at'
	| 'updated_at'
>;

/** Fields the caller supplies when creating a row. */
export type NewCard = Pick<
	StoredCard,
	'card_id' | 'scryfall_id' | 'name' | 'set_code' | 'collector_number' | 'foil'
> & {
	/** Defaults to 1. */
	count?: number;
	/** Defaults to now; imports pass the source's timestamp so "recently added" order means something. */
	created_at?: string;
};

/** What the store can change on a row. `created_at` is for folds and imports, never from clients. */
export type StorePatch = UpdateCardRequest & { created_at?: string };

interface Row {
	card_id: string;
	scryfall_id: string | null;
	name: string;
	set_code: string | null;
	collector_number: string | null;
	foil: number;
	count: number;
	created_at: string;
	updated_at: string;
}

function rowToCard(row: Row): StoredCard {
	return {
		card_id: row.card_id,
		scryfall_id: row.scryfall_id,
		name: row.name,
		set_code: row.set_code,
		collector_number: row.collector_number,
		foil: row.foil !== 0,
		count: row.count,
		created_at: row.created_at,
		updated_at: row.updated_at,
	};
}

const now = (): string => new Date().toISOString();

function checkCount(count: number): number {
	if (!Number.isInteger(count) || count < 1) throw badRequest('count must be a positive whole number');
	return count;
}

/** The parameters of insert.sql and upsert.sql. */
function newRowParams(card: NewCard, ts: string) {
	return {
		card_id: card.card_id,
		scryfall_id: card.scryfall_id,
		name: card.name,
		set_code: card.set_code,
		collector_number: card.collector_number,
		foil: card.foil ? 1 : 0,
		count: checkCount(card.count ?? 1),
		created_at: card.created_at ?? ts,
		updated_at: ts,
	};
}

export class CardStore {
	#db: DatabaseSync;

	private constructor(db: DatabaseSync) {
		this.#db = db;
	}

	/**
	 * Open (or create) the database at `file`. Use `':memory:'` for tests.
	 * The printing rule is put in place unless the data already breaks it
	 * (rows from before the rule), in which case the caller folds those
	 * first and then calls `enforcePrintingRule`. Tests that need a
	 * pre-rule database pass `enforcePrintingRule: false`.
	 */
	static open(file: string, options: { enforcePrintingRule?: boolean } = {}): CardStore {
		const db = new DatabaseSync(file);
		if (file !== ':memory:') db.exec('PRAGMA journal_mode = WAL');
		db.exec(SQL.schema);
		const store = new CardStore(db);
		if (options.enforcePrintingRule !== false && !store.hasDuplicatePrintings()) store.enforcePrintingRule();
		return store;
	}

	close(): void {
		this.#db.close();
	}

	/** Whether the unique printing + foil index exists. */
	printingRuleEnforced(): boolean {
		return this.#db.prepare(SQL['printing-rule-enforced']).get() !== undefined;
	}

	/** Rows that share a printing + foil exist (only possible before the rule was enforced). */
	hasDuplicatePrintings(): boolean {
		const row = this.#db.prepare(SQL['has-duplicate-printings']).get() as { n: number };
		return row.n > 0;
	}

	/** Create the unique index. Throws if duplicate printings still exist. */
	enforcePrintingRule(): void {
		this.#db.exec(SQL['printing-rule']);
	}

	insert(card: NewCard): StoredCard {
		this.#db.prepare(SQL.insert).run(newRowParams(card, now()));
		return this.#mustGet(card.card_id);
	}

	/**
	 * Insert, or fold into the row that already holds the printing + foil:
	 * `add` adds the copies (upsert.sql), `set` makes the count the given
	 * one (upsert-set.sql, for imports). Either way the survivor takes the
	 * newer added date when it gained copies. Requires the printing rule.
	 */
	upsert(card: NewCard, mode: 'add' | 'set' = 'add'): { card: StoredCard; merged: boolean } {
		const statement = mode === 'set' ? SQL['upsert-set'] : SQL.upsert;
		const row = this.#db.prepare(statement).get(newRowParams(card, now())) as unknown as Row;
		return { card: rowToCard(row), merged: row.card_id !== card.card_id };
	}

	get(cardId: string): StoredCard | null {
		const row = this.#db.prepare(SQL.get).get({ card_id: cardId }) as Row | undefined;
		return row ? rowToCard(row) : null;
	}

	/** Every card, newest first. */
	list(): StoredCard[] {
		const rows = this.#db.prepare(SQL.list).all() as unknown as Row[];
		return rows.map(rowToCard);
	}

	count(): number {
		const row = this.#db.prepare(SQL.count).get() as { n: number };
		return row.n;
	}

	/** Identified cards with the same printing and foil status. */
	findDuplicates(scryfallId: string, foil: boolean): StoredCard[] {
		const rows = this.#db
			.prepare(SQL['find-duplicates'])
			.all({ scryfall_id: scryfallId, foil: foil ? 1 : 0 }) as unknown as Row[];
		return rows.map(rowToCard);
	}

	/**
	 * Apply a partial update. Only keys present in `patch` are written (see
	 * update.sql). Returns the updated row, or throws 404 if the card does
	 * not exist.
	 */
	update(cardId: string, patch: StorePatch): StoredCard {
		const fields = [
			'scryfall_id',
			'name',
			'set_code',
			'collector_number',
			'foil',
			'count',
			'created_at',
		] as const;
		if (fields.every((f) => patch[f] === undefined)) return this.#mustGet(cardId);
		if (patch.count !== undefined) checkCount(patch.count);
		const params: Record<string, string | number | null> = { card_id: cardId, updated_at: now() };
		for (const f of fields) {
			const v = patch[f];
			params[`set_${f}`] = v === undefined ? 0 : 1;
			params[f] = v === undefined ? null : typeof v === 'boolean' ? (v ? 1 : 0) : v;
		}
		const result = this.#db.prepare(SQL.update).run(params);
		if (result.changes === 0) throw notFound(`Card ${cardId} not found`);
		return this.#mustGet(cardId);
	}

	/** Returns `true` if a row was removed. */
	delete(cardId: string): boolean {
		return this.#db.prepare(SQL.delete).run({ card_id: cardId }).changes > 0;
	}

	/**
	 * Fold `source` into `target`: counts add, the target's `created_at`
	 * becomes the newer of the two (it now holds the newest addition), and
	 * `source` is deleted. Atomic; may run inside a `transaction`.
	 */
	merge(targetId: string, sourceId: string): StoredCard {
		if (targetId === sourceId) throw badRequest('Cannot merge a card into itself');
		const source = this.get(sourceId);
		if (!source) throw notFound(`Card ${sourceId} not found`);
		const updated = this.#db
			.prepare(SQL.merge)
			.run({ card_id: targetId, count: source.count, created_at: source.created_at, updated_at: now() });
		if (updated.changes === 0) throw notFound(`Card ${targetId} not found`);
		this.delete(sourceId);
		return this.#mustGet(targetId);
	}

	/** Run `fn` inside one write transaction; any throw rolls everything back. */
	transaction<T>(fn: () => T): T {
		const db = this.#db;
		db.exec('BEGIN IMMEDIATE');
		try {
			const result = fn();
			db.exec('COMMIT');
			return result;
		} catch (err) {
			db.exec('ROLLBACK');
			throw err;
		}
	}

	#mustGet(cardId: string): StoredCard {
		const card = this.get(cardId);
		if (!card) throw notFound(`Card ${cardId} not found`);
		return card;
	}
}
