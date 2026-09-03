import type { CardEntry } from '@shared/api';
import { Camera } from 'lucide-solid';
import { For, Show } from 'solid-js';
import { Badge } from '~/components/ui/Badge';
import { Button } from '~/components/ui/Button';
import { CardImage } from '~/components/ui/CardImage';
import { showUserImages } from '~/lib/image-pref';
import { scryfallImageUrl, userImageUrl } from '~/lib/images';

export interface CardGridProps {
	cards: CardEntry[];
	total: number;
	query: string;
	/** Whether any filter is active (for the empty-result message). */
	filtered: boolean;
	onSelect: (card: CardEntry) => void;
	onScan: () => void;
	onImport: () => void;
	onClearSearch: () => void;
	onClearFilters: () => void;
}

export function CardGrid(props: CardGridProps) {
	return (
		<Show when={props.total > 0} fallback={<EmptyLibrary onScan={props.onScan} onImport={props.onImport} />}>
			<Show
				when={props.cards.length > 0}
				fallback={
					<NoMatches
						query={props.query}
						filtered={props.filtered}
						onClearSearch={props.onClearSearch}
						onClearFilters={props.onClearFilters}
					/>
				}
			>
				<div class="grid grid-cols-2 gap-x-3 gap-y-5 min-[400px]:grid-cols-3 sm:grid-cols-4 lg:grid-cols-5 xl:grid-cols-6">
					<For each={props.cards}>
						{(card) => <CardTile card={card} onClick={() => props.onSelect(card)} />}
					</For>
				</div>
			</Show>
		</Show>
	);
}

function CardTile(props: { card: CardEntry; onClick: () => void }) {
	const placeholder = () => props.card.scryfall_id === null;
	/** The user's photo when asked for and present; otherwise the printing's art; otherwise nothing. */
	const src = (): string | null => {
		const c = props.card;
		if (c.has_photo && (showUserImages() || !c.scryfall_id)) return userImageUrl(c);
		return c.scryfall_id ? scryfallImageUrl(c.scryfall_id) : null;
	};

	return (
		<button
			type="button"
			onClick={props.onClick}
			class="group text-left focus-visible:outline-none"
			aria-label={`${props.card.name}${props.card.count > 1 ? `, ${props.card.count} copies` : ''}`}
		>
			<div class="relative">
				<CardImage
					src={src()}
					alt=""
					class="transition-transform duration-150 group-active:scale-[0.98] group-focus-visible:ring-2 group-focus-visible:ring-accent"
					onError={(e) => {
						const img = e.currentTarget as HTMLImageElement;
						if (props.card.scryfall_id && !img.src.includes('scryfall.io'))
							img.src = scryfallImageUrl(props.card.scryfall_id);
					}}
				/>
				<Show when={props.card.count > 1}>
					<Badge variant="count" class="absolute top-1.5 right-1.5">
						×{props.card.count}
					</Badge>
				</Show>
				<Show when={props.card.foil}>
					<Badge variant="foil" class="absolute top-1.5 left-1.5">
						Foil
					</Badge>
				</Show>
				<Show when={placeholder()}>
					<Badge variant="warn" class="absolute bottom-1.5 left-1.5">
						Unidentified
					</Badge>
				</Show>
			</div>
			<p class="mt-2 truncate text-[13px] font-medium leading-tight text-ink">{props.card.name}</p>
			<p class="truncate text-[12px] text-muted">
				<Show when={props.card.set_code} fallback="Tap to identify">
					{props.card.set_code?.toUpperCase()} {props.card.collector_number}
				</Show>
			</p>
		</button>
	);
}

function EmptyLibrary(props: { onScan: () => void; onImport: () => void }) {
	return (
		<div class="mx-auto flex max-w-xs flex-col items-center pt-16 text-center">
			<div class="card-slot flex w-40 items-center justify-center text-muted">
				<Camera class="h-9 w-9" />
			</div>
			<h2 class="mt-7 text-[20px] font-semibold">Your collection starts here</h2>
			<p class="mt-2 text-[15px] text-muted">
				Point the camera at a card, or at a whole binder page, and it's identified and saved. Already keep a
				collection on Moxfield? Bring it over.
			</p>
			<Button variant="primary" size="lg" class="mt-7" onClick={props.onScan}>
				<Camera class="h-5 w-5" />
				Scan a card
			</Button>
			<Button variant="ghost" class="mt-2" onClick={props.onImport}>
				Import from Moxfield
			</Button>
		</div>
	);
}

function NoMatches(props: {
	query: string;
	filtered: boolean;
	onClearSearch: () => void;
	onClearFilters: () => void;
}) {
	const title = () =>
		props.query.trim() ? `Nothing matches “${props.query.trim()}”` : 'Nothing matches these filters';
	const hint = () => {
		if (props.query.trim() && props.filtered)
			return 'Search looks at card names and sets, within the filters you set.';
		if (props.query.trim()) return 'Search looks at card names, set codes and set names.';
		return 'Loosen a filter, or clear them all.';
	};
	return (
		<div class="mx-auto flex max-w-xs flex-col items-center pt-16 text-center">
			<h2 class="text-[17px] font-semibold">{title()}</h2>
			<p class="mt-2 text-[15px] text-muted">{hint()}</p>
			<div class="mt-6 flex gap-2">
				<Show when={props.query.trim()}>
					<Button onClick={props.onClearSearch}>Clear search</Button>
				</Show>
				<Show when={props.filtered}>
					<Button onClick={props.onClearFilters}>Clear filters</Button>
				</Show>
			</div>
		</div>
	);
}
