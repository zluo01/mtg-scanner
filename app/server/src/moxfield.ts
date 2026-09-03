/**
 * Import a Moxfield collection export into the library.
 *
 * Each row names a printing by set code + collector number; that pair is
 * looked up in the index metadata (preferring the row's language) and the
 * library row takes the printing's canonical name, so imported cards look
 * exactly like scanned ones. Rows the index does not know are kept as
 * unidentified cards with the file's name, set and number, ready for the
 * "identify" flow, and reported back.
 *
 * Cards are keyed by printing + foil (as scans are), so a file row lands on
 * an existing card when there is one. `mode` decides what happens then.
 */
import { randomUUID } from 'node:crypto';
import type { ImportMode, ImportResponse } from '../../shared/api.ts';
import { parseCsv } from './csv.ts';
import type { CardStore, StoredCard } from './db.ts';
import { badRequest } from './errors.ts';
import type { CardCatalog } from './metadata.ts';

export interface MoxfieldRow {
	count: number;
	name: string;
	/** Lower-cased set code (Moxfield's "Edition"). */
	set_code: string;
	collector_number: string;
	/** Moxfield distinguishes foil and etched; both are foil here. */
	foil: boolean;
	/** Scryfall language code, `en` when unknown. */
	lang: string;
	/** "Last Modified" as ISO 8601, when present and parseable. */
	modified_at: string | null;
}

const LANGUAGES: Record<string, string> = {
	english: 'en',
	spanish: 'es',
	french: 'fr',
	german: 'de',
	italian: 'it',
	portuguese: 'pt',
	japanese: 'ja',
	korean: 'ko',
	russian: 'ru',
	'chinese simplified': 'zhs',
	'chinese traditional': 'zht',
	phyrexian: 'ph',
};

/** Moxfield writes `2026-09-01 00:24:01.653000` (UTC, microseconds). */
export function parseMoxfieldTimestamp(value: string): string | null {
	const m = /^(\d{4}-\d{2}-\d{2})[ T](\d{2}:\d{2}:\d{2})(?:\.(\d+))?/.exec(value.trim());
	if (!m) return null;
	const millis = (m[3] ?? '').padEnd(3, '0').slice(0, 3);
	const d = new Date(`${m[1]}T${m[2]}.${millis}Z`);
	return Number.isNaN(d.getTime()) ? null : d.toISOString();
}

export function parseMoxfieldCsv(text: string): MoxfieldRow[] {
	const rows = parseCsv(text);
	const head = rows[0];
	if (!head) throw badRequest('The file is empty');
	const columns = head.map((c) => c.trim().toLowerCase());
	const col = (name: string) => columns.indexOf(name);
	const iCount = col('count');
	const iName = col('name');
	const iSet = col('edition');
	const iNumber = col('collector number');
	const iFoil = col('foil');
	const iLang = col('language');
	const iModified = col('last modified');
	if (iName < 0 || iSet < 0 || iNumber < 0) {
		throw badRequest('Not a Moxfield collection CSV: it needs Name, Edition and Collector Number columns');
	}

	return rows.slice(1).map((cells, i) => {
		const get = (index: number) => (index >= 0 ? (cells[index] ?? '').trim() : '');
		const line = i + 2;
		const name = get(iName);
		if (!name) throw badRequest(`Line ${line}: missing card name`);
		const count = iCount >= 0 ? Number(get(iCount)) : 1;
		if (!Number.isInteger(count) || count < 1 || count > 9999) {
			throw badRequest(`Line ${line}: count must be a whole number between 1 and 9999`);
		}
		return {
			count,
			name,
			set_code: get(iSet).toLowerCase(),
			collector_number: get(iNumber),
			foil: get(iFoil) !== '',
			lang: LANGUAGES[get(iLang).toLowerCase()] ?? 'en',
			modified_at: parseMoxfieldTimestamp(get(iModified)),
		};
	});
}

export interface ImportDeps {
	cards: CardStore;
	catalog: CardCatalog;
}

/** Printing + foil for identified cards; name + set + number + foil otherwise. */
const keyOf = (c: Pick<StoredCard, 'scryfall_id' | 'name' | 'set_code' | 'collector_number' | 'foil'>) =>
	c.scryfall_id
		? `id:${c.scryfall_id}:${c.foil ? 1 : 0}`
		: `u:${c.name.toLowerCase()}:${c.set_code ?? ''}:${c.collector_number ?? ''}:${c.foil ? 1 : 0}`;

export function importMoxfield(
	{ cards, catalog }: ImportDeps,
	rows: MoxfieldRow[],
	mode: ImportMode,
): ImportResponse {
	const byKey = new Map<string, StoredCard>();
	for (const c of cards.list()) byKey.set(keyOf(c), c);
	/** Keys this file has already written, so a second row for the same card adds to the first. */
	const seen = new Set<string>();
	let added = 0;
	let updated = 0;
	const unmatched: string[] = [];

	cards.transaction(() => {
		for (const row of rows) {
			const printing = catalog.findPrinting(row.set_code, row.collector_number, row.lang);
			const identity = printing
				? {
						scryfall_id: printing.scryfall_id,
						name: printing.name,
						set_code: printing.set_code,
						collector_number: printing.collector_number,
					}
				: {
						scryfall_id: null,
						name: row.name,
						set_code: row.set_code || null,
						collector_number: row.collector_number || null,
					};
			if (!printing) unmatched.push(row.name);

			const key = keyOf({ ...identity, foil: row.foil });
			const existing = byKey.get(key);
			if (existing) {
				const count = mode === 'add' || seen.has(key) ? existing.count + row.count : row.count;
				if (count !== existing.count) {
					// More copies than before: the card counts as newly added.
					const created_at =
						count > existing.count ? (row.modified_at ?? new Date().toISOString()) : undefined;
					byKey.set(key, cards.update(existing.card_id, { count, created_at }));
				}
				if (!seen.has(key)) updated++;
			} else {
				const created = cards.insert({
					card_id: randomUUID(),
					...identity,
					foil: row.foil,
					count: row.count,
					created_at: row.modified_at ?? undefined,
				});
				byKey.set(key, created);
				added++;
			}
			seen.add(key);
		}
	});

	return {
		rows: rows.length,
		added,
		updated,
		unmatched: unmatched.length,
		unmatched_names: [...new Set(unmatched)].slice(0, 50),
	};
}
