import type { CardEntry } from '@shared/api';
import { Camera } from 'lucide-solid';
import { createMemo, createSignal, Show } from 'solid-js';
import { CardDetail } from '~/components/CardDetail';
import { CardGrid } from '~/components/CardGrid';
import { FilterSheet } from '~/components/FilterSheet';
import { Header } from '~/components/Header';
import { Scanner } from '~/components/Scanner';
import { SettingsSheet } from '~/components/SettingsSheet';
import { SortSheet } from '~/components/SortSheet';
import { Toasts } from '~/components/Toasts';
import { Button } from '~/components/ui/Button';
import { Spinner } from '~/components/ui/Spinner';
import { errorMessage } from '~/lib/api';
import { activeFilterCount, applyView, EMPTY_FILTERS, setOptions } from '~/lib/filters';
import { createLibrary, LibraryContext } from '~/lib/library';
import { createUrlState } from '~/lib/url-state';

export function App() {
	const library = createLibrary();
	const url = createUrlState();
	const [selectedId, setSelectedId] = createSignal<string | null>(null);
	const [scannerOpen, setScannerOpen] = createSignal(false);
	const [sortOpen, setSortOpen] = createSignal(false);
	const [filtersOpen, setFiltersOpen] = createSignal(false);
	const [settingsOpen, setSettingsOpen] = createSignal(false);

	const visible = createMemo(() => applyView(library.cards(), url.view()));
	const setNames = createMemo(() => new Map(setOptions(library.cards()).map((o) => [o.value, o.label])));
	const selected = createMemo<CardEntry | null>(
		() => library.cards().find((c) => c.card_id === selectedId()) ?? null,
	);
	const empty = () => !library.loading() && library.cards().length === 0;
	const clearFilters = () => url.setFilters(EMPTY_FILTERS);

	return (
		<LibraryContext.Provider value={library}>
			<div class="flex min-h-dvh flex-col bg-bg text-ink">
				<Header
					view={url.view()}
					onQuery={url.setQuery}
					onFilters={url.setFilters}
					onOpenSort={() => setSortOpen(true)}
					onOpenFilters={() => setFiltersOpen(true)}
					onOpenSettings={() => setSettingsOpen(true)}
					setName={(code) => setNames().get(code) ?? code.toUpperCase()}
					total={library.cards().length}
					visible={visible().length}
				/>

				<main class="mx-auto w-full max-w-6xl flex-1 px-4 pt-4 pb-[calc(6.5rem+env(safe-area-inset-bottom))]">
					<Show
						when={!library.error()}
						fallback={
							<div class="mx-auto flex max-w-xs flex-col items-center pt-16 text-center">
								<h2 class="text-[17px] font-semibold">Can't reach the server</h2>
								<p class="mt-2 text-[15px] text-muted">{errorMessage(library.error())}</p>
								<Button class="mt-6" onClick={library.refetch}>
									Try again
								</Button>
							</div>
						}
					>
						<Show
							when={!library.loading() || library.cards().length > 0}
							fallback={
								<div class="flex justify-center pt-24">
									<Spinner />
								</div>
							}
						>
							<CardGrid
								cards={visible()}
								total={library.cards().length}
								query={url.query()}
								filtered={activeFilterCount(url.filters()) > 0}
								onSelect={(card) => setSelectedId(card.card_id)}
								onScan={() => setScannerOpen(true)}
								onImport={() => setSettingsOpen(true)}
								onClearSearch={() => url.setQuery('')}
								onClearFilters={clearFilters}
							/>
						</Show>
					</Show>
				</main>

				<Show when={!empty()}>
					<button
						type="button"
						onClick={() => setScannerOpen(true)}
						aria-label="Scan a card"
						class="fixed right-5 bottom-[calc(1.25rem+env(safe-area-inset-bottom))] z-30 flex h-16 w-16 items-center justify-center rounded-full bg-accent text-on-accent shadow-[0_8px_24px_rgba(0,0,0,0.35)] transition-transform active:scale-95 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ink"
					>
						<Camera class="h-7 w-7" />
					</button>
				</Show>

				<Scanner open={scannerOpen()} onClose={() => setScannerOpen(false)} />
				<CardDetail card={selected()} onClose={() => setSelectedId(null)} onSelect={setSelectedId} />
				<SortSheet
					open={sortOpen()}
					onClose={() => setSortOpen(false)}
					value={url.sort()}
					onChange={url.setSort}
				/>
				<SettingsSheet open={settingsOpen()} onClose={() => setSettingsOpen(false)} />
				<FilterSheet
					open={filtersOpen()}
					onClose={() => setFiltersOpen(false)}
					cards={library.cards()}
					filters={url.filters()}
					onChange={url.setFilters}
					visible={visible().length}
				/>
				<Toasts />
			</div>
		</LibraryContext.Provider>
	);
}
