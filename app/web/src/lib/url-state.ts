/** The library view (search, sort, filters) mirrored into the URL so it survives reloads. */
import { createEffect, createSignal, onCleanup } from 'solid-js';
import { type Filters, parseView, type SortKey, serializeView, type View } from './filters.ts';

export function createUrlState() {
	const [view, setView] = createSignal<View>(parseView(window.location.search));

	createEffect(() => {
		const next = `${window.location.pathname}${serializeView(view())}`;
		if (next !== `${window.location.pathname}${window.location.search}`) {
			window.history.replaceState(null, '', next);
		}
	});

	const onPop = () => setView(parseView(window.location.search));
	window.addEventListener('popstate', onPop);
	onCleanup(() => window.removeEventListener('popstate', onPop));

	return {
		view,
		query: () => view().query,
		sort: () => view().sort,
		filters: () => view().filters,
		setQuery: (query: string) => setView((v) => ({ ...v, query })),
		setSort: (sort: SortKey) => setView((v) => ({ ...v, sort })),
		setFilters: (filters: Filters) => setView((v) => ({ ...v, filters })),
	};
}

export type UrlState = ReturnType<typeof createUrlState>;
