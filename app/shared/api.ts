/**
 * HTTP API contract shared by `server/` and `web/`.
 *
 * Types only. Both packages import this file with `import type`, so it never
 * reaches a runtime bundle. Keep it free of runtime code.
 */

/**
 * A card in the user's library.
 *
 * The first nine fields are what SQLite stores. The rest describe the
 * printing and are attached by the server from the index metadata on every
 * response, keyed by `scryfall_id`; they are all `null` for unidentified
 * placeholders. Nothing about the printing is stored twice.
 *
 * One row holds every copy of a printing + foil. The server keeps that
 * true on every write: a scan of a card already owned, a foil flip, a
 * corrected printing or an import that lands on an existing card folds
 * into it (counts add, `created_at` becomes the newest addition), so a
 * response may come back with a different `card_id` than was sent.
 */
export interface CardEntry {
	/** Client-generated UUID. Primary key. */
	card_id: string;
	/** Scryfall printing id. `null` for unidentified placeholders. */
	scryfall_id: string | null;
	/** Card name. `"Unknown"` for placeholders. */
	name: string;
	set_code: string | null;
	collector_number: string | null;
	foil: boolean;
	/** Copy count, always >= 1. */
	count: number;
	/** ISO 8601 UTC timestamps. */
	created_at: string;
	updated_at: string;

	/** Whether a scan photo exists at `/scans/{card_id}.jpg`. Imported cards have none. */
	has_photo: boolean;

	artist: string | null;
	type_line: string | null;
	/** common | uncommon | rare | mythic | special | bonus */
	rarity: string | null;
	set_name: string | null;
	/** Letters in WUBRG order, `""` for colorless. */
	colors: string | null;
	mana_value: number | null;
	/** Set release date, `YYYY-MM-DD`. */
	released_at: string | null;
}

export type Confidence = 'CONFIDENT' | 'AMBIGUOUS' | 'NO_MATCH';

/** A Scryfall printing from the reference database. */
export interface ScryfallCard {
	scryfall_id: string;
	name: string;
	set_code: string;
	collector_number: string;
	artist: string | null;
}

/** A nearest-neighbour hit from the visual index. */
export interface ScanCandidate extends ScryfallCard {
	/** Cosine similarity in [-1, 1]; higher is closer. */
	similarity: number;
}

/**
 * `POST /api/identify` with a JPEG/PNG body: what the index makes of the
 * photo. Nothing is stored; the client reviews this and then adds the card.
 */
export interface IdentifyResponse {
	confidence: Confidence;
	/** Similarity of the best hit; 0 when the index returned nothing. */
	similarity: number;
	/** Top-K hits, best first. Present for every confidence level. */
	candidates: ScanCandidate[];
}

/**
 * `POST /api/cards?card_id=&scryfall_id=&foil=` with the photo as body:
 * add the scanned card as the user confirmed it. `scryfall_id` empty means
 * unidentified. The printing's name, set and number come from the index.
 */
export interface AddCardResponse {
	/**
	 * The library row now holding this scan: a new row with the requested
	 * `card_id`, or the card already owning this printing + foil (its count
	 * went up by one and `merged` is true).
	 */
	card: CardEntry;
	merged: boolean;
}

/**
 * `PUT /api/cards/:id` — partial update; omitted fields are left unchanged.
 * Printing attributes are never sent: they follow `scryfall_id`. A change
 * that lands on a printing + foil already held folds into that card, and
 * the response is that card.
 */
export interface UpdateCardRequest {
	scryfall_id?: string | null;
	name?: string;
	set_code?: string | null;
	collector_number?: string | null;
	foil?: boolean;
	count?: number;
}

export interface LibraryResponse {
	cards: CardEntry[];
}

/**
 * `POST /api/import?mode=` with a Moxfield collection CSV body. Rows are
 * matched to printings by set code + collector number. `set` makes an
 * existing card's count equal the file's (re-importing is idempotent);
 * `add` adds the file's copies on top.
 */
export type ImportMode = 'set' | 'add';

export interface ImportResponse {
	/** Rows read from the file. */
	rows: number;
	/** Library cards created. */
	added: number;
	/** Existing library cards whose count the file changed or confirmed. */
	updated: number;
	/** Rows whose printing is not in the index; kept as unidentified cards. */
	unmatched: number;
	/** Distinct names of unmatched rows, first 50. */
	unmatched_names: string[];
}

/** `GET /api/export?format=`: everything the app knows, or Moxfield's CSV layout. */
export type ExportFormat = 'full' | 'moxfield';

export interface CardResponse {
	card: CardEntry;
}

export interface SearchResponse {
	cards: ScryfallCard[];
}

export interface DeleteResponse {
	success: true;
}

export interface HealthResponse {
	ok: true;
	/** Number of printings in the visual index. */
	cards_indexed: number;
	/** Number of rows in the user's library. */
	library_size: number;
}

/** Every non-2xx response from `/api/*` has this shape. */
export interface ErrorResponse {
	error: string;
	status: number;
}
