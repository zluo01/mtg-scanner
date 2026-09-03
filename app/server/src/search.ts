/**
 * Search over the Scryfall reference metadata: case-insensitive name
 * substrings, and "set code + collector number" (`neo 172`, `NEO/172`,
 * `plst tsp-157`), which finds a printing in any language without knowing
 * its English name.
 */
import type { ScryfallCard } from '../../shared/api.ts';
import type { CardCatalog, CardMetadata } from './metadata.ts';

export const MAX_RESULTS = 50;
export const MIN_QUERY_LENGTH = 2;

interface Entry {
	lower: string;
	card: CardMetadata;
}

/** Rank buckets: lower is better. */
function rank(lower: string, q: string): number {
	if (lower === q) return 0;
	if (lower.startsWith(q)) return 1;
	if (lower.includes(` ${q}`)) return 2;
	if (lower.includes(q)) return 3;
	return -1;
}

const collator = new Intl.Collator('en', { numeric: true, sensitivity: 'base' });

/** Only the public fields leave the server. */
const toResult = (c: CardMetadata): ScryfallCard => ({
	scryfall_id: c.scryfall_id,
	name: c.name,
	set_code: c.set_code,
	collector_number: c.collector_number,
	artist: c.artist,
});

/**
 * `neo 172`, `neo/172`, `neo#172`, `neo:172`, `neo-172` (hyphen only when a
 * digit follows, so `plst tsp-157` keeps its hyphenated number). Set codes
 * are 2-6 letters/digits; numbers may carry letters, hyphens, ★ and †.
 */
const SET_NUMBER = /^([a-z0-9]{2,6})(?:[\s/#:]+|-(?=\d))([a-z0-9★†-]+)$/i;

export function parseSetNumber(query: string): { setCode: string; collectorNumber: string } | null {
	const m = SET_NUMBER.exec(query.trim());
	return m ? { setCode: m[1]!.toLowerCase(), collectorNumber: m[2]! } : null;
}

export class NameSearch {
	#entries: Entry[];
	#catalog: CardCatalog | undefined;

	/** Holds references to the shared metadata rows; nothing is copied. */
	constructor(cards: Iterable<CardMetadata>, catalog?: CardCatalog) {
		this.#entries = Array.from(cards, (card) => ({ lower: card.name.toLowerCase(), card }));
		this.#catalog = catalog;
	}

	get size(): number {
		return this.#entries.length;
	}

	/**
	 * A set + number query lists those printings first (English first, then
	 * other languages). Then name matches: exact and prefix rank first, then
	 * word-start, then any substring; ties break by name, set, collector
	 * number (numeric-aware).
	 */
	search(query: string, limit = MAX_RESULTS): ScryfallCard[] {
		const q = query.trim().toLowerCase();
		if (q.length < MIN_QUERY_LENGTH) return [];

		const byNumber = this.#byNumber(query);
		const seen = new Set(byNumber.map((c) => c.scryfall_id));
		const hits: { r: number; card: CardMetadata }[] = [];
		for (const e of this.#entries) {
			if (seen.has(e.card.scryfall_id)) continue;
			const r = rank(e.lower, q);
			if (r >= 0) hits.push({ r, card: e.card });
		}
		hits.sort(
			(a, b) =>
				a.r - b.r ||
				collator.compare(a.card.name, b.card.name) ||
				collator.compare(a.card.set_code, b.card.set_code) ||
				collator.compare(a.card.collector_number, b.card.collector_number),
		);
		return [...byNumber, ...hits.map((h) => h.card)].slice(0, limit).map(toResult);
	}

	#byNumber(query: string): CardMetadata[] {
		const parsed = this.#catalog && parseSetNumber(query);
		return parsed ? this.#catalog!.findPrintings(parsed.setCode, parsed.collectorNumber) : [];
	}
}
