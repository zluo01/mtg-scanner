import type { CardEntry } from '@shared/api';

export type ScryfallSize = 'small' | 'normal';

/** Scryfall CDN URL for a printing's front face. */
export function scryfallImageUrl(scryfallId: string, size: ScryfallSize = 'normal'): string {
	return `https://cards.scryfall.io/${size}/front/${scryfallId[0]}/${scryfallId[1]}/${scryfallId}.jpg`;
}

/**
 * The user's own scan photo. `updated_at` is appended so the browser
 * refetches after a merge replaces the file under the same name.
 */
export function userImageUrl(card: Pick<CardEntry, 'card_id' | 'updated_at'>): string {
	return `/scans/${card.card_id}.jpg?v=${encodeURIComponent(card.updated_at)}`;
}
