/** CSV in and out: the app's own export, Moxfield's layout, and an RFC 4180 parser. */
import type { CardEntry } from '../../shared/api.ts';
import { PLACEHOLDER_NAME } from './scan.ts';

const COLUMNS = [
	'name',
	'set_code',
	'set_name',
	'collector_number',
	'rarity',
	'artist',
	'colors',
	'mana_value',
	'foil',
	'count',
	'scryfall_id',
	'card_id',
	'created_at',
] as const;

/** RFC 4180 field quoting: wrap in quotes when needed, double embedded quotes. */
export function csvField(value: string | number | boolean | null): string {
	if (value === null) return '';
	const s = String(value);
	return /[",\r\n]/.test(s) ? `"${s.replaceAll('"', '""')}"` : s;
}

/** Render the library as CSV with a header row. */
export function libraryToCsv(cards: CardEntry[]): string {
	const lines = [COLUMNS.join(',')];
	for (const card of cards) {
		lines.push(COLUMNS.map((col) => csvField(card[col])).join(','));
	}
	return `${lines.join('\n')}\n`;
}

/**
 * Parse RFC 4180 CSV: quoted fields, doubled quotes, LF or CRLF line ends,
 * an optional byte-order mark. Blank lines are skipped. Returns rows of
 * raw string fields; the caller interprets the header.
 */
export function parseCsv(text: string): string[][] {
	const rows: string[][] = [];
	let row: string[] = [];
	let field = '';
	let quoted = false;
	const endRow = () => {
		row.push(field);
		field = '';
		if (row.length > 1 || row[0] !== '') rows.push(row);
		row = [];
	};
	for (let i = text.charCodeAt(0) === 0xfeff ? 1 : 0; i < text.length; i++) {
		const ch = text[i];
		if (quoted) {
			if (ch !== '"') field += ch;
			else if (text[i + 1] === '"') {
				field += '"';
				i++;
			} else quoted = false;
		} else if (ch === '"') quoted = true;
		else if (ch === ',') {
			row.push(field);
			field = '';
		} else if (ch === '\n') endRow();
		else if (ch !== '\r') field += ch;
	}
	if (field !== '' || row.length > 0) endRow();
	return rows;
}

// ------------------------------------------------------------ Moxfield

/** Moxfield's collection export columns, in its order. */
export const MOXFIELD_COLUMNS = [
	'Count',
	'Tradelist Count',
	'Name',
	'Edition',
	'Condition',
	'Language',
	'Foil',
	'Tags',
	'Last Modified',
	'Collector Number',
	'Alter',
	'Proxy',
	'Purchase Price',
] as const;

/** Moxfield quotes every field. */
const quoted = (value: string | number) => `"${String(value).replaceAll('"', '""')}"`;

/** `2026-09-01T00:24:01.653Z` -> `2026-09-01 00:24:01.653000` (Moxfield's timestamp form). */
export function moxfieldTimestamp(iso: string): string {
	const d = new Date(iso);
	if (Number.isNaN(d.getTime())) return '';
	const s = d.toISOString();
	return `${s.slice(0, 10)} ${s.slice(11, 23)}000`;
}

/**
 * Render the library in Moxfield's own export layout so it imports back
 * without mapping. Unidentified cards with no name are left out; those
 * that came from Moxfield keep their name, set and number and round-trip.
 * The caller supplies the whole card name (`Front // Back` for double-faced
 * cards, which Moxfield expects; library rows hold the front face).
 * Condition, language and tradelist count are not tracked here, so they
 * are written as Near Mint, English, and the full count.
 */
export function libraryToMoxfieldCsv(cards: CardEntry[]): string {
	const lines = [MOXFIELD_COLUMNS.map(quoted).join(',')];
	for (const c of cards) {
		if (c.scryfall_id === null && c.name === PLACEHOLDER_NAME) continue;
		lines.push(
			[
				c.count,
				c.count,
				c.name,
				c.set_code ?? '',
				'Near Mint',
				'English',
				c.foil ? 'foil' : '',
				'',
				moxfieldTimestamp(c.updated_at),
				c.collector_number ?? '',
				'False',
				'False',
				'',
			]
				.map(quoted)
				.join(','),
		);
	}
	return `${lines.join('\n')}\n`;
}
