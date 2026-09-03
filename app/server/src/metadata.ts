/**
 * Card metadata that accompanies the visual index. Row `i` of the parquet
 * file describes vector `i` of the FAISS index.
 *
 * Written by the Python training pipeline (`build_embedding_index.py`).
 * Column names from an older Rust-written variant (`id`, `set`) are also
 * accepted, and the attribute columns are optional so older files still
 * load (their attributes come back as `null`).
 */
import { asyncBufferFromFile, parquetMetadataAsync, parquetReadObjects, parquetSchema } from 'hyparquet';
import type { CardEntry, ScryfallCard } from '../../shared/api.ts';

export interface CardMetadata extends ScryfallCard {
	type_line: string | null;
	rarity: string | null;
	set_name: string | null;
	/** Letters in WUBRG order, `""` for colorless; `null` if the column is absent. */
	colors: string | null;
	mana_value: number | null;
	released_at: string | null;
	/** Scryfall language code (`en`, `ja`, ...); `null` if the column is absent. */
	lang: string | null;
	/** Image file name in the training corpus, when the parquet records it. */
	filename: string | null;
}

/** The printing attributes a library card inherits from its `scryfall_id`. */
export type PrintingAttributes = Pick<
	CardEntry,
	'artist' | 'type_line' | 'rarity' | 'set_name' | 'colors' | 'mana_value' | 'released_at'
>;

const NO_ATTRIBUTES: PrintingAttributes = {
	artist: null,
	type_line: null,
	rarity: null,
	set_name: null,
	colors: null,
	mana_value: null,
	released_at: null,
};

/** Set codes are lower-case already; collector numbers are matched case-insensitively (`TSP-157`). */
const numberKey = (setCode: string, collectorNumber: string) =>
	`${setCode.toLowerCase()}|${collectorNumber.toLowerCase()}`;

const ID_COLUMNS = ['scryfall_id', 'id'];
const SET_COLUMNS = ['set_code', 'set'];

function pick(available: Set<string>, candidates: string[], required: true): string;
function pick(available: Set<string>, candidates: string[], required: false): string | null;
function pick(available: Set<string>, candidates: string[], required: boolean): string | null {
	const found = candidates.find((c) => available.has(c));
	if (!found && required) {
		throw new Error(`card_metadata.parquet is missing a required column (tried ${candidates.join(', ')})`);
	}
	return found ?? null;
}

const str = (value: unknown): string => (value == null ? '' : String(value));
const strOrNull = (value: unknown): string | null => (value == null ? null : String(value));
const numOrNull = (value: unknown): number | null => {
	if (value == null) return null;
	const n = Number(value);
	return Number.isFinite(n) ? n : null;
};

export async function loadCardMetadata(file: string): Promise<CardMetadata[]> {
	const buffer = await asyncBufferFromFile(file);
	const meta = await parquetMetadataAsync(buffer);
	const available = new Set(parquetSchema(meta).children.map((c) => c.element.name));

	const idCol = pick(available, ID_COLUMNS, true);
	const setCol = pick(available, SET_COLUMNS, true);
	const nameCol = pick(available, ['name'], true);
	const cnCol = pick(available, ['collector_number'], true);
	const optional = {
		artist: pick(available, ['artist'], false),
		type_line: pick(available, ['type_line'], false),
		rarity: pick(available, ['rarity'], false),
		set_name: pick(available, ['set_name'], false),
		colors: pick(available, ['colors'], false),
		mana_value: pick(available, ['mana_value', 'cmc'], false),
		released_at: pick(available, ['released_at'], false),
		lang: pick(available, ['lang'], false),
		filename: pick(available, ['filename'], false),
	};

	const columns = [idCol, nameCol, setCol, cnCol, ...Object.values(optional)].filter(
		(c): c is string => c !== null,
	);
	const rows = await parquetReadObjects({ file: buffer, columns });

	return rows.map((row) => {
		const get = (col: string | null) => (col ? row[col] : undefined);
		const artist = str(get(optional.artist));
		return {
			scryfall_id: str(row[idCol]),
			name: str(row[nameCol]),
			set_code: str(row[setCol]),
			collector_number: str(row[cnCol]),
			artist: artist === '' ? null : artist,
			type_line: strOrNull(get(optional.type_line)),
			rarity: strOrNull(get(optional.rarity)),
			set_name: strOrNull(get(optional.set_name)),
			colors: strOrNull(get(optional.colors)),
			mana_value: numOrNull(get(optional.mana_value)),
			released_at: strOrNull(get(optional.released_at)),
			lang: strOrNull(get(optional.lang)),
			filename: strOrNull(get(optional.filename)) || null,
		};
	});
}

/**
 * The metadata rows plus lookups by id and by set + collector number.
 * Double-faced cards contribute one row per face with the same
 * `scryfall_id`; the id lookup keeps the first (front) face. A printing
 * Scryfall has renumbered can appear under two numbers (the stale row is
 * kept by the index refresh), so the number lookup registers every
 * distinct set + number a printing has had.
 */
export class CardCatalog {
	readonly rows: CardMetadata[];
	readonly #byId = new Map<string, CardMetadata>();
	/** `set|number` -> the printings with that number (one per language). */
	readonly #byNumber = new Map<string, CardMetadata[]>();
	/** id -> distinct face names in order (one for most cards, two for double-faced). */
	readonly #faceNames = new Map<string, string[]>();

	constructor(rows: CardMetadata[]) {
		this.rows = rows;
		for (const row of rows) {
			if (!this.#byId.has(row.scryfall_id)) this.#byId.set(row.scryfall_id, row);
			const key = numberKey(row.set_code, row.collector_number);
			const list = this.#byNumber.get(key);
			if (!list) this.#byNumber.set(key, [row]);
			else if (!list.some((m) => m.scryfall_id === row.scryfall_id)) list.push(row);
			const names = this.#faceNames.get(row.scryfall_id);
			if (!names) this.#faceNames.set(row.scryfall_id, [row.name]);
			else if (!names.includes(row.name)) names.push(row.name);
		}
	}

	/** Every printing (one per language) with this set code and collector number, English first. */
	findPrintings(setCode: string, collectorNumber: string): CardMetadata[] {
		const list = this.#byNumber.get(numberKey(setCode, collectorNumber)) ?? [];
		return [...list].sort(
			(a, b) => Number(b.lang === 'en' || b.lang === null) - Number(a.lang === 'en' || a.lang === null),
		);
	}

	/**
	 * The whole card's name, `Front // Back` for double-faced cards (the
	 * rows name each face). Moxfield and Scryfall both use this form.
	 */
	fullName(scryfallId: string | null): string | undefined {
		return scryfallId ? this.#faceNames.get(scryfallId)?.join(' // ') : undefined;
	}

	get size(): number {
		return this.rows.length;
	}

	get(scryfallId: string): CardMetadata | undefined {
		return this.#byId.get(scryfallId);
	}

	/**
	 * The printing with this set code and collector number, preferring the
	 * given language, then English, then whatever is there.
	 */
	findPrinting(setCode: string, collectorNumber: string, lang = 'en'): CardMetadata | undefined {
		const list = this.findPrintings(setCode, collectorNumber);
		return list.find((m) => m.lang === lang) ?? list[0];
	}

	/** Attributes for a printing, or all-null when unknown or unidentified. */
	attributes(scryfallId: string | null): PrintingAttributes {
		const m = scryfallId ? this.#byId.get(scryfallId) : undefined;
		if (!m) return NO_ATTRIBUTES;
		return {
			artist: m.artist,
			type_line: m.type_line,
			rarity: m.rarity,
			set_name: m.set_name,
			colors: m.colors,
			mana_value: m.mana_value,
			released_at: m.released_at,
		};
	}
}
