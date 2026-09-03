/**
 * Library view model: filters, sort, and name search, plus their URL form.
 *
 * Pure functions only (no DOM), so this is unit-tested under Node.
 * Semantics: within one facet the chosen options combine with OR, across
 * facets with AND, and the text search is one more AND on top. Attribute
 * facets (set, rarity, type, artist, colour, mana value) exclude
 * unidentified cards, which have no attributes.
 */
import type { CardEntry } from '@shared/api';

export type SortKey = 'date' | 'name' | 'set' | 'release' | 'mana' | 'rarity';
export const SORT_OPTIONS: { value: SortKey; label: string }[] = [
	{ value: 'date', label: 'Recently added' },
	{ value: 'name', label: 'Name' },
	{ value: 'set', label: 'Set' },
	{ value: 'release', label: 'Newest set' },
	{ value: 'mana', label: 'Mana value' },
	{ value: 'rarity', label: 'Rarity' },
];

export const COLORS = [
	{ value: 'W', label: 'White' },
	{ value: 'U', label: 'Blue' },
	{ value: 'B', label: 'Black' },
	{ value: 'R', label: 'Red' },
	{ value: 'G', label: 'Green' },
	{ value: 'C', label: 'Colorless' },
	{ value: 'M', label: 'Multicolor' },
] as const;
export type ColorKey = (typeof COLORS)[number]['value'];

export const RARITIES = ['common', 'uncommon', 'rare', 'mythic'] as const;
export const TYPES = [
	'creature',
	'instant',
	'sorcery',
	'artifact',
	'enchantment',
	'planeswalker',
	'land',
	'battle',
	'legendary',
] as const;
export const MANA_VALUES = ['0', '1', '2', '3', '4', '5', '6+'] as const;

export type FoilFilter = 'any' | 'foil' | 'nonfoil';

export interface Filters {
	sets: string[];
	rarities: string[];
	types: string[];
	artists: string[];
	colors: string[];
	manaValues: string[];
	foil: FoilFilter;
	/** Only cards with more than one copy. */
	multiples: boolean;
	/** Only unidentified cards. */
	attention: boolean;
}

export const EMPTY_FILTERS: Filters = {
	sets: [],
	rarities: [],
	types: [],
	artists: [],
	colors: [],
	manaValues: [],
	foil: 'any',
	multiples: false,
	attention: false,
};

export interface View {
	query: string;
	sort: SortKey;
	filters: Filters;
}

export const DEFAULT_VIEW: View = { query: '', sort: 'date', filters: EMPTY_FILTERS };

// ------------------------------------------------------------------ URL

const LIST_KEYS: [keyof Filters & string, string][] = [
	['sets', 'set'],
	['rarities', 'rarity'],
	['types', 'type'],
	['artists', 'artist'],
	['colors', 'color'],
	['manaValues', 'mv'],
];

const isSortKey = (v: string | null): v is SortKey => SORT_OPTIONS.some((o) => o.value === v);

export function parseView(search: string): View {
	const p = new URLSearchParams(search);
	const filters: Filters = { ...EMPTY_FILTERS };
	for (const [field, key] of LIST_KEYS) {
		const raw = p.get(key);
		(filters[field] as string[]) = raw ? raw.split(',').filter(Boolean) : [];
	}
	const foil = p.get('foil');
	filters.foil = foil === 'foil' || foil === 'nonfoil' ? foil : 'any';
	filters.multiples = p.get('copies') === 'multi';
	filters.attention = p.get('attention') === '1';
	const sort = p.get('sort');
	return { query: p.get('q') ?? '', sort: isSortKey(sort) ? sort : 'date', filters };
}

export function serializeView(view: View): string {
	const p = new URLSearchParams();
	if (view.query) p.set('q', view.query);
	if (view.sort !== 'date') p.set('sort', view.sort);
	for (const [field, key] of LIST_KEYS) {
		const list = view.filters[field] as string[];
		if (list.length > 0) p.set(key, list.join(','));
	}
	if (view.filters.foil !== 'any') p.set('foil', view.filters.foil);
	if (view.filters.multiples) p.set('copies', 'multi');
	if (view.filters.attention) p.set('attention', '1');
	const s = p.toString();
	return s ? `?${s}` : '';
}

/** Number of active filter facets (what the toolbar badge shows). */
export function activeFilterCount(f: Filters): number {
	let n = 0;
	for (const [field] of LIST_KEYS) if ((f[field] as string[]).length > 0) n++;
	if (f.foil !== 'any') n++;
	if (f.multiples) n++;
	if (f.attention) n++;
	return n;
}

/** One removable chip per active filter value, in display order. */
export interface ActiveChip {
	key: string;
	label: string;
	/** The filters with this one value removed. */
	without: Filters;
}

const capitalize = (s: string) => s.charAt(0).toUpperCase() + s.slice(1);

export function activeChips(f: Filters, setName: (code: string) => string): ActiveChip[] {
	const chips: ActiveChip[] = [];
	if (f.attention)
		chips.push({ key: 'attention', label: 'Needs identifying', without: { ...f, attention: false } });
	if (f.multiples)
		chips.push({ key: 'multiples', label: 'More than one copy', without: { ...f, multiples: false } });
	if (f.foil !== 'any')
		chips.push({
			key: 'foil',
			label: f.foil === 'foil' ? 'Foil' : 'Not foil',
			without: { ...f, foil: 'any' },
		});
	const list = (field: keyof Filters & string, label: (v: string) => string) => {
		for (const v of f[field] as string[]) {
			chips.push({
				key: `${field}:${v}`,
				label: label(v),
				without: { ...f, [field]: (f[field] as string[]).filter((x) => x !== v) },
			});
		}
	};
	list('colors', (v) => COLORS.find((c) => c.value === v)?.label ?? v);
	list('manaValues', (v) => `Mana value ${v}`);
	list('rarities', capitalize);
	list('types', capitalize);
	list('sets', setName);
	list('artists', (v) => v);
	return chips;
}

// ------------------------------------------------------------- matching

export function manaBucket(value: number | null): string | null {
	if (value === null) return null;
	return value >= 6 ? '6+' : String(Math.floor(value));
}

function matchesColor(colors: string | null, wanted: string[]): boolean {
	if (colors === null) return false;
	return wanted.some((c) => {
		if (c === 'C') return colors === '';
		if (c === 'M') return colors.length >= 2;
		return colors.includes(c);
	});
}

function matchesType(typeLine: string | null, wanted: string[]): boolean {
	if (typeLine === null) return false;
	const lower = typeLine.toLowerCase();
	return wanted.some((t) => lower.includes(t));
}

export function matchesFilters(card: CardEntry, f: Filters): boolean {
	if (f.attention && card.scryfall_id !== null) return false;
	if (f.multiples && card.count < 2) return false;
	if (f.foil === 'foil' && !card.foil) return false;
	if (f.foil === 'nonfoil' && card.foil) return false;
	if (f.sets.length > 0 && !(card.set_code && f.sets.includes(card.set_code))) return false;
	if (f.rarities.length > 0 && !(card.rarity && f.rarities.includes(card.rarity))) return false;
	if (f.types.length > 0 && !matchesType(card.type_line, f.types)) return false;
	if (f.artists.length > 0 && !(card.artist && f.artists.includes(card.artist))) return false;
	if (f.colors.length > 0 && !matchesColor(card.colors, f.colors)) return false;
	if (f.manaValues.length > 0) {
		const b = manaBucket(card.mana_value);
		if (b === null || !f.manaValues.includes(b)) return false;
	}
	return true;
}

// ------------------------------------------------------------------ sort

const RARITY_RANK: Record<string, number> = { mythic: 0, rare: 1, uncommon: 2, common: 3 };
const byName = (a: CardEntry, b: CardEntry) => a.name.localeCompare(b.name);

export function compareCards(sort: SortKey): (a: CardEntry, b: CardEntry) => number {
	switch (sort) {
		case 'name':
			return byName;
		case 'set':
			return (a, b) => (a.set_code ?? '').localeCompare(b.set_code ?? '') || byName(a, b);
		case 'release':
			// Newest set first; unknown release dates last.
			return (a, b) => (b.released_at ?? '').localeCompare(a.released_at ?? '') || byName(a, b);
		case 'mana':
			return (a, b) =>
				(a.mana_value ?? Number.POSITIVE_INFINITY) - (b.mana_value ?? Number.POSITIVE_INFINITY) ||
				byName(a, b);
		case 'rarity':
			return (a, b) =>
				(RARITY_RANK[a.rarity ?? ''] ?? 9) - (RARITY_RANK[b.rarity ?? ''] ?? 9) || byName(a, b);
		default:
			return (a, b) => b.created_at.localeCompare(a.created_at);
	}
}

/** `zen 21`, `zen/21`, `zen#21`: a set code and (the start of) a collector number. */
const SET_NUMBER = /^([a-z0-9]{2,6})[\s/#:]+(\S+)$/;

/**
 * The text search: part of the card name, a set code (whole or its start,
 * since codes are three or four letters), part of the set name, or a set
 * code followed by a collector number. `q` is trimmed and lower-cased.
 */
export function matchesQuery(card: CardEntry, q: string): boolean {
	if (!q) return true;
	if (card.name.toLowerCase().includes(q)) return true;
	const code = card.set_code?.toLowerCase() ?? '';
	if (code.startsWith(q)) return true;
	if (card.set_name?.toLowerCase().includes(q)) return true;
	const m = SET_NUMBER.exec(q);
	return m !== null && code === m[1] && (card.collector_number?.toLowerCase().startsWith(m[2]!) ?? false);
}

/** Filters, then text search, then sort. Returns a new array. */
export function applyView(cards: CardEntry[], view: View): CardEntry[] {
	const q = view.query.trim().toLowerCase();
	const out = cards.filter((c) => matchesFilters(c, view.filters) && matchesQuery(c, q));
	return out.sort(compareCards(view.sort));
}

// ---------------------------------------------------------------- facets

export interface FacetOption {
	value: string;
	label: string;
	count: number;
}

const countBy = (
	cards: CardEntry[],
	key: (c: CardEntry) => string | null,
	label: (v: string, c: CardEntry) => string,
) => {
	const map = new Map<string, FacetOption>();
	for (const c of cards) {
		const v = key(c);
		if (!v) continue;
		const cur = map.get(v);
		if (cur) cur.count++;
		else map.set(v, { value: v, label: label(v, c), count: 1 });
	}
	return [...map.values()];
};

/** Sets present in the library, by set name, with counts. */
export function setOptions(cards: CardEntry[]): FacetOption[] {
	return countBy(
		cards,
		(c) => c.set_code,
		(v, c) => c.set_name ?? v.toUpperCase(),
	).sort((a, b) => a.label.localeCompare(b.label));
}

/** Artists present in the library, most frequent first. */
export function artistOptions(cards: CardEntry[]): FacetOption[] {
	return countBy(
		cards,
		(c) => c.artist,
		(v) => v,
	).sort((a, b) => b.count - a.count || a.label.localeCompare(b.label));
}
