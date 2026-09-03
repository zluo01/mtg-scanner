/** Type-ahead against the Scryfall reference database. Renders the body of a sheet. */
import type { ScryfallCard } from '@shared/api';
import { Search } from 'lucide-solid';
import { createSignal, For, onCleanup, onMount, Show } from 'solid-js';
import { CardImage } from '~/components/ui/CardImage';
import { Spinner } from '~/components/ui/Spinner';
import { api, errorMessage } from '~/lib/api';
import { scryfallImageUrl } from '~/lib/images';

const DEBOUNCE_MS = 300;
const MIN_CHARS = 2;

export function CardSearch(props: { initialQuery?: string; onSelect: (card: ScryfallCard) => void }) {
	const [query, setQuery] = createSignal(props.initialQuery ?? '');
	const [results, setResults] = createSignal<ScryfallCard[]>([]);
	const [loading, setLoading] = createSignal(false);
	const [error, setError] = createSignal<string | null>(null);
	let input: HTMLInputElement | undefined;
	let timer: ReturnType<typeof setTimeout> | undefined;
	let inflight: AbortController | undefined;

	const run = async (value: string) => {
		inflight?.abort();
		const controller = new AbortController();
		inflight = controller;
		setLoading(true);
		setError(null);
		try {
			const res = await api.search(value, controller.signal);
			if (!controller.signal.aborted) setResults(res.cards);
		} catch (err) {
			if (!controller.signal.aborted) {
				setResults([]);
				setError(errorMessage(err));
			}
		} finally {
			if (!controller.signal.aborted) setLoading(false);
		}
	};

	const onInput = (value: string) => {
		setQuery(value);
		clearTimeout(timer);
		if (value.trim().length < MIN_CHARS) {
			inflight?.abort();
			setResults([]);
			setLoading(false);
			return;
		}
		timer = setTimeout(() => run(value.trim()), DEBOUNCE_MS);
	};

	onMount(() => {
		input?.focus();
		input?.select();
		if (query().trim().length >= MIN_CHARS) run(query().trim());
	});
	onCleanup(() => {
		clearTimeout(timer);
		inflight?.abort();
	});

	return (
		<div class="flex min-h-0 flex-1 flex-col">
			<div class="relative shrink-0 px-4 pt-3 pb-2">
				<Search class="pointer-events-none absolute top-1/2 left-7 h-4 w-4 -translate-y-1/2 text-muted" />
				<input
					ref={input}
					type="search"
					placeholder="Card name, or set and number"
					value={query()}
					onInput={(e) => onInput(e.currentTarget.value)}
					class="h-11 w-full rounded-control bg-raised pl-9 pr-3 text-[15px] text-ink outline-none placeholder:text-muted focus:ring-2 focus:ring-accent"
				/>
			</div>

			<div class="min-h-0 flex-1 overflow-y-auto">
				<Show when={loading()}>
					<div class="flex justify-center py-6">
						<Spinner class="h-5 w-5" />
					</div>
				</Show>
				<Show when={error()}>
					<p class="px-4 py-6 text-center text-[14px] text-bad">{error()}</p>
				</Show>
				<Show when={!loading() && !error() && query().trim().length < MIN_CHARS}>
					<p class="px-4 py-6 text-center text-[14px] text-muted">
						Type part of the card name, or the set code and collector number printed on the card, like{' '}
						<span class="text-ink">neo 172</span>. That works in any language.
					</p>
				</Show>
				<Show when={!loading() && !error() && query().trim().length >= MIN_CHARS && results().length === 0}>
					<p class="px-4 py-6 text-center text-[14px] text-muted">
						No printings found for “{query().trim()}”.
					</p>
				</Show>
				<ul>
					<For each={results()}>
						{(card) => (
							<li>
								<button
									type="button"
									onClick={() => props.onSelect(card)}
									class="flex w-full items-center gap-3 px-4 py-2 text-left hover:bg-raised"
								>
									<CardImage src={scryfallImageUrl(card.scryfall_id, 'small')} class="w-11 shrink-0" />
									<div class="min-w-0 flex-1">
										<p class="truncate text-[15px] font-medium">{card.name}</p>
										<p class="truncate text-[13px] text-muted">
											{card.set_code.toUpperCase()} {card.collector_number}
											<Show when={card.artist}> — {card.artist}</Show>
										</p>
									</div>
								</button>
							</li>
						)}
					</For>
				</ul>
			</div>
		</div>
	);
}
