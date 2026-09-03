/**
 * The user's card library as reactive state. One resource holds the list;
 * mutations patch it locally for instant feedback and the server response
 * (which is authoritative) is merged back in.
 */
import type { CardEntry } from '@shared/api';
import { type Accessor, createContext, createResource, useContext } from 'solid-js';
import { api } from './api.ts';

export interface Library {
	cards: Accessor<CardEntry[]>;
	loading: Accessor<boolean>;
	error: Accessor<unknown>;
	refetch: () => void;
	/** Insert or replace a card in place. */
	upsert: (card: CardEntry) => void;
	remove: (cardId: string) => void;
}

export function createLibrary(): Library {
	const [cards, { refetch, mutate }] = createResource<CardEntry[]>(async () => (await api.library()).cards, {
		initialValue: [],
	});
	return {
		cards: () => cards() ?? [],
		loading: () => cards.loading,
		error: () => cards.error,
		refetch: () => void refetch(),
		upsert: (card) =>
			mutate((list = []) => {
				const i = list.findIndex((c) => c.card_id === card.card_id);
				if (i < 0) return [card, ...list];
				const next = list.slice();
				next[i] = card;
				return next;
			}),
		remove: (cardId) => mutate((list = []) => list.filter((c) => c.card_id !== cardId)),
	};
}

export const LibraryContext = createContext<Library>();

export function useLibrary(): Library {
	const lib = useContext(LibraryContext);
	if (!lib) throw new Error('useLibrary must be used inside LibraryContext');
	return lib;
}
