/**
 * The library's one rule: a printing + foil lives in exactly one row.
 *
 * The database enforces it (a unique index over printing + foil, see
 * `db.ts`), so adding is a single upsert: counts add and the survivor takes
 * the newest added date. What the database cannot do is move a photo, so
 * this service sits above the store for the cases that involve one: a
 * scan's photo when it folds into a card without one, a photo following a
 * row that folds after an edit, and the one-time fold of rows written
 * before the rule existed. Placeholders (no printing) never fold. Imports
 * fold on the same key in `moxfield.ts`.
 */
import type { UpdateCardRequest } from '../../shared/api.ts';
import type { CardStore, NewCard, StoredCard } from './db.ts';
import { notFound } from './errors.ts';
import type { ImageStore } from './images.ts';

export interface Added {
	card: StoredCard;
	/** The copy was added to a card already in the library. */
	merged: boolean;
}

export class Library {
	readonly #cards: CardStore;
	readonly #images: ImageStore;

	constructor(cards: CardStore, images: ImageStore) {
		this.#cards = cards;
		this.#images = images;
	}

	get(cardId: string): StoredCard | null {
		return this.#cards.get(cardId);
	}

	hasPhoto(cardId: string): boolean {
		return this.#images.has(cardId);
	}

	/** The card holding this printing + foil, other than `except`. */
	find(scryfallId: string | null, foil: boolean, except?: string): StoredCard | undefined {
		if (!scryfallId) return undefined;
		return this.#cards.findDuplicates(scryfallId, foil).find((c) => c.card_id !== except);
	}

	/** Insert, or fold into the card that already holds this printing + foil. */
	add(card: NewCard): Added {
		return this.#cards.upsert(card);
	}

	/**
	 * A scan: add the row (or fold) and store its photo, unless the card it
	 * folded into already has one. If the photo cannot be written the copy
	 * is taken back out, so a row never claims a photo that is not there.
	 */
	async addScan(card: NewCard, photo: Uint8Array): Promise<Added> {
		const added = this.add(card);
		const id = added.card.card_id;
		if (added.merged && this.#images.has(id)) return added;
		try {
			await this.#images.write(id, photo);
		} catch (err) {
			if (added.merged) this.#cards.update(id, { count: added.card.count - 1 });
			else await this.remove(id);
			throw err;
		}
		return added;
	}

	/**
	 * Apply a change. If it would move the card onto a printing + foil
	 * another card already holds, the card folds into that one instead
	 * (the index would refuse the plain update). Returns the row now
	 * holding the copies, which may not be `cardId`.
	 */
	async change(cardId: string, patch: UpdateCardRequest): Promise<StoredCard> {
		const { survivor, folded } = this.#cards.transaction(() => {
			const current = this.#cards.get(cardId);
			if (!current) throw notFound(`Card ${cardId} not found`);
			const next = { ...current, ...patch };
			const twin = this.find(next.scryfall_id, next.foil, cardId);
			if (!twin) return { survivor: this.#cards.update(cardId, patch), folded: false };
			if (patch.count !== undefined) this.#cards.update(cardId, { count: patch.count });
			return { survivor: this.#cards.merge(twin.card_id, cardId), folded: true };
		});
		if (folded) await this.#movePhoto(cardId, survivor.card_id);
		return survivor;
	}

	/** Remove a card and its photo. `false` if it does not exist. */
	async remove(cardId: string): Promise<boolean> {
		if (!this.#cards.delete(cardId)) return false;
		await this.#images.remove(cardId);
		return true;
	}

	/**
	 * One-time fold of rows written before the database enforced the rule.
	 * The oldest row survives with the newest added date. Returns how many
	 * rows were folded away; the caller then enforces the rule.
	 */
	async dedupeAll(): Promise<number> {
		const groups = new Map<string, StoredCard[]>();
		for (const c of this.#cards.list()) {
			if (!c.scryfall_id) continue;
			const key = `${c.scryfall_id}|${c.foil ? 1 : 0}`;
			const list = groups.get(key);
			if (list) list.push(c);
			else groups.set(key, [c]);
		}
		let folded = 0;
		for (const rows of groups.values()) {
			if (rows.length < 2) continue;
			const keep = rows[rows.length - 1]!; // list() is newest first
			for (const extra of rows.slice(0, -1)) {
				this.#cards.transaction(() => this.#cards.merge(keep.card_id, extra.card_id));
				await this.#movePhoto(extra.card_id, keep.card_id);
				folded++;
			}
		}
		return folded;
	}

	/** After a fold: the survivor keeps its photo, or inherits the folded row's. */
	async #movePhoto(from: string, to: string): Promise<void> {
		if (!this.#images.has(from)) return;
		if (!this.#images.has(to)) await this.#images.copy(from, to);
		await this.#images.remove(from);
	}
}
