import { ArrowDownUp, Camera, Image, Search, Settings, SlidersHorizontal, X } from 'lucide-solid';
import { createSignal, For, type JSX, Show } from 'solid-js';
import { Chip } from '~/components/ui/Chip';
import {
	activeChips,
	activeFilterCount,
	EMPTY_FILTERS,
	type Filters,
	SORT_OPTIONS,
	type View,
} from '~/lib/filters';
import { showUserImages, toggleUserImages } from '~/lib/image-pref';

export interface HeaderProps {
	view: View;
	onQuery: (q: string) => void;
	onFilters: (filters: Filters) => void;
	onOpenSort: () => void;
	onOpenFilters: () => void;
	onOpenSettings: () => void;
	/** Display name for a set code, for the active-filter chips. */
	setName: (code: string) => string;
	total: number;
	visible: number;
}

const IconButton = (p: { label: string; onClick: () => void; pressed?: boolean; children: JSX.Element }) => (
	<button
		type="button"
		onClick={p.onClick}
		aria-label={p.label}
		title={p.label}
		aria-pressed={p.pressed}
		class="flex h-11 w-11 items-center justify-center rounded-control text-ink hover:bg-raised aria-pressed:text-accent"
	>
		{p.children}
	</button>
);

/** Toolbar control: icon, short label, optional count. */
const ToolButton = (p: { label: string; onClick: () => void; children: JSX.Element }) => (
	<button
		type="button"
		onClick={p.onClick}
		aria-label={p.label}
		class="flex h-9 items-center gap-1.5 rounded-control px-2.5 text-[13px] font-medium text-ink hover:bg-raised aria-expanded:bg-raised"
	>
		{p.children}
	</button>
);

export function Header(props: HeaderProps) {
	const [searchOpen, setSearchOpen] = createSignal(false);
	let mobileInput: HTMLInputElement | undefined;

	const query = () => props.view.query;
	const filters = () => props.view.filters;
	const filterCount = () => activeFilterCount(filters());
	const chips = () => activeChips(filters(), props.setName);
	const sortLabel = () => SORT_OPTIONS.find((o) => o.value === props.view.sort)?.label ?? 'Sort';

	const count = () => {
		if (props.total === 0) return 'No cards yet';
		const all = props.total === 1 ? '1 card' : `${props.total} cards`;
		return props.visible === props.total ? all : `${props.visible} of ${props.total}`;
	};

	const openSearch = () => {
		setSearchOpen(true);
		queueMicrotask(() => mobileInput?.focus());
	};
	const closeSearch = () => {
		setSearchOpen(false);
		props.onQuery('');
	};

	const SearchField = (p: { ref?: (el: HTMLInputElement) => void; class?: string }) => (
		<div class={`relative ${p.class ?? ''}`}>
			<Search class="pointer-events-none absolute top-1/2 left-3 h-4 w-4 -translate-y-1/2 text-muted" />
			<input
				ref={p.ref}
				type="search"
				placeholder="Search names and sets"
				value={query()}
				onInput={(e) => props.onQuery(e.currentTarget.value)}
				class="h-9 w-full rounded-control bg-raised pr-8 pl-9 text-[14px] text-ink outline-none placeholder:text-muted focus:ring-2 focus:ring-accent"
			/>
			<Show when={query()}>
				<button
					type="button"
					aria-label="Clear search"
					onClick={() => props.onQuery('')}
					class="absolute top-1/2 right-1 flex h-7 w-7 -translate-y-1/2 items-center justify-center rounded-md text-muted hover:text-ink"
				>
					<X class="h-4 w-4" />
				</button>
			</Show>
		</div>
	);

	return (
		<header class="sticky top-0 z-40 border-b border-line bg-bg/95 backdrop-blur pt-[env(safe-area-inset-top)]">
			<div class="mx-auto flex h-14 max-w-6xl items-center gap-1 pr-1 pl-4">
				<h1 class="text-[22px] font-semibold tracking-tight">MTG Scanner</h1>
				<div class="flex-1" />
				<Show when={!searchOpen()}>
					<div class="sm:hidden">
						<IconButton label="Search" onClick={openSearch}>
							<Search class="h-5 w-5" />
						</IconButton>
					</div>
				</Show>
				<IconButton
					label={
						showUserImages()
							? 'Showing your photos. Switch to card art'
							: 'Showing card art. Switch to your photos'
					}
					onClick={toggleUserImages}
					pressed={showUserImages()}
				>
					<Show when={showUserImages()} fallback={<Image class="h-5 w-5" />}>
						<Camera class="h-5 w-5" />
					</Show>
				</IconButton>
				<IconButton label="Settings" onClick={props.onOpenSettings}>
					<Settings class="h-5 w-5" />
				</IconButton>
			</div>

			<div class="mx-auto flex h-12 max-w-6xl items-center gap-2 px-4">
				<Show
					when={!searchOpen()}
					fallback={
						<>
							<SearchField
								class="flex-1"
								ref={(el) => {
									mobileInput = el;
								}}
							/>
							<button type="button" onClick={closeSearch} class="text-[14px] font-medium text-ink">
								Cancel
							</button>
						</>
					}
				>
					<SearchField class="hidden w-64 sm:block" />
					<span class="text-[13px] text-muted">{count()}</span>
					<div class="flex-1" />
					<ToolButton label={`Sort: ${sortLabel()}`} onClick={props.onOpenSort}>
						<ArrowDownUp class="h-4 w-4 text-muted" />
						<span class="max-w-32 truncate">{sortLabel()}</span>
					</ToolButton>
					<ToolButton
						label={filterCount() > 0 ? `Filters, ${filterCount()} active` : 'Filters'}
						onClick={props.onOpenFilters}
					>
						<SlidersHorizontal class="h-4 w-4 text-muted" />
						Filters
						<Show when={filterCount() > 0}>
							<span class="flex h-5 min-w-5 items-center justify-center rounded-full bg-accent px-1 text-[11px] font-semibold text-on-accent">
								{filterCount()}
							</span>
						</Show>
					</ToolButton>
				</Show>
			</div>

			<Show when={chips().length > 0}>
				<div class="mx-auto flex max-w-6xl items-center gap-2 overflow-x-auto px-4 pb-2.5 [scrollbar-width:none]">
					<For each={chips()}>
						{(chip) => (
							<Chip
								size="sm"
								pressed
								aria-label={`Remove filter ${chip.label}`}
								onClick={() => props.onFilters(chip.without)}
							>
								{chip.label}
								<X class="h-3.5 w-3.5" />
							</Chip>
						)}
					</For>
					<button
						type="button"
						onClick={() => props.onFilters(EMPTY_FILTERS)}
						class="shrink-0 px-2 text-[13px] font-medium text-muted hover:text-ink"
					>
						Clear all
					</button>
				</div>
			</Show>
		</header>
	);
}
