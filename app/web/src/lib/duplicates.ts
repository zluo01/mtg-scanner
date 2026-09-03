/** The library card that already holds a printing + foil, for "you already have this" previews. */
import type { CardEntry } from '@shared/api';

export function findOwned(cards: CardEntry[], scryfallId: string | null, foil: boolean): CardEntry | null {
	if (!scryfallId) return null;
	return cards.find((c) => c.scryfall_id === scryfallId && c.foil === foil) ?? null;
}
