/**
 * SQLite-backed card library using Node's built-in `node:sqlite`.
 *
 * A row holds only what identifies a card and what the user chose: the
 * printing id, a readable name/set/number fallback, foil, and count.
 * Everything else about the printing (artist, type, colours, ...) is
 * attached at read time from the index metadata (see `metadata.ts`).
 *
 * All methods are synchronous: the library is tiny (thousands of rows at
 * most) and single-user, so a blocking call costs microseconds and keeps
 * the call sites simple.
 */
import { DatabaseSync } from 'node:sqlite';
import type { CardEntry, UpdateCardRequest } from '../../shared/api.ts';
import { badRequest, notFound } from './errors.ts';

const SCHEMA = `
CREATE TABLE IF NOT EXISTS cards (
	card_id          TEXT PRIMARY KEY,
	scryfall_id      TEXT,
	name             TEXT NOT NULL,
	set_code         TEXT,
	collector_number TEXT,
	foil             INTEGER NOT NULL DEFAULT 0,
	count            INTEGER NOT NULL DEFAULT 1,
	created_at       TEXT NOT NULL,
	updated_at       TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_cards_scryfall_id ON cards (scryfall_id);
`;

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

export class CardStore {
	#db: DatabaseSync;

	private constructor(db: DatabaseSync) {
		this.#db = db;
	}

	/** Open (or create) the database at `file`. Use `':memory:'` for tests. */
	static open(file: string): CardStore {
		const db = new DatabaseSync(file);
		if (file !== ':memory:') db.exec('PRAGMA journal_mode = WAL');
		db.exec(SCHEMA);
		return new CardStore(db);
	}

	close(): void {
		this.#db.close();
	}

	insert(card: NewCard): StoredCard {
		const count = card.count ?? 1;
		if (!Number.isInteger(count) || count < 1) throw badRequest('count must be a positive whole number');
		const ts = now();
		this.#db
			.prepare(
				`INSERT INTO cards (card_id, scryfall_id, name, set_code, collector_number, foil, count, created_at, updated_at)
				 VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)`,
			)
			.run(
				card.card_id,
				card.scryfall_id,
				card.name,
				card.set_code,
				card.collector_number,
				card.foil ? 1 : 0,
				count,
				card.created_at ?? ts,
				ts,
			);
		return this.#mustGet(card.card_id);
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

	get(cardId: string): StoredCard | null {
		const row = this.#db.prepare('SELECT * FROM cards WHERE card_id = ?').get(cardId) as Row | undefined;
		return row ? rowToCard(row) : null;
	}

	/** Every card, newest first. */
	list(): StoredCard[] {
		const rows = this.#db
			.prepare('SELECT * FROM cards ORDER BY created_at DESC, rowid DESC')
			.all() as unknown as Row[];
		return rows.map(rowToCard);
	}

	count(): number {
		const row = this.#db.prepare('SELECT COUNT(*) AS n FROM cards').get() as { n: number };
		return row.n;
	}

	/** Identified cards with the same printing and foil status. */
	findDuplicates(scryfallId: string, foil: boolean): StoredCard[] {
		const rows = this.#db
			.prepare('SELECT * FROM cards WHERE scryfall_id = ? AND foil = ? ORDER BY created_at')
			.all(scryfallId, foil ? 1 : 0) as unknown as Row[];
		return rows.map(rowToCard);
	}

	/**
	 * Apply a partial update. Only keys present in `patch` are written
	 * (`copies` is not a column and is ignored here; see `Library.change`).
	 * Returns the updated row, or throws 404 if the card does not exist.
	 */
	update(cardId: string, patch: StorePatch): StoredCard {
		const sets: string[] = [];
		const values: (string | number | null)[] = [];
		const put = (column: string, value: string | number | null) => {
			sets.push(`${column} = ?`);
			values.push(value);
		};
		if (patch.scryfall_id !== undefined) put('scryfall_id', patch.scryfall_id);
		if (patch.name !== undefined) put('name', patch.name);
		if (patch.set_code !== undefined) put('set_code', patch.set_code);
		if (patch.collector_number !== undefined) put('collector_number', patch.collector_number);
		if (patch.foil !== undefined) put('foil', patch.foil ? 1 : 0);
		if (patch.count !== undefined) put('count', patch.count);
		if (patch.created_at !== undefined) put('created_at', patch.created_at);
		if (sets.length === 0) return this.#mustGet(cardId);

		put('updated_at', now());
		values.push(cardId);
		const result = this.#db.prepare(`UPDATE cards SET ${sets.join(', ')} WHERE card_id = ?`).run(...values);
		if (result.changes === 0) throw notFound(`Card ${cardId} not found`);
		return this.#mustGet(cardId);
	}

	/** Returns `true` if a row was removed. */
	delete(cardId: string): boolean {
		const result = this.#db.prepare('DELETE FROM cards WHERE card_id = ?').run(cardId);
		return result.changes > 0;
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
			.prepare(
				'UPDATE cards SET count = count + ?, created_at = MAX(created_at, ?), updated_at = ? WHERE card_id = ?',
			)
			.run(source.count, source.created_at, now(), targetId);
		if (updated.changes === 0) throw notFound(`Card ${targetId} not found`);
		this.#db.prepare('DELETE FROM cards WHERE card_id = ?').run(sourceId);
		return this.#mustGet(targetId);
	}

	#mustGet(cardId: string): StoredCard {
		const card = this.get(cardId);
		if (!card) throw notFound(`Card ${cardId} not found`);
		return card;
	}
}
