/**
 * Library filters. Changes apply immediately (the grid updates behind the
 * sheet); the footer button only closes. Set and artist lists are built from
 * the library itself, so they only ever offer values that exist.
 */
import type { CardEntry } from '@shared/api';
import { createMemo, createSignal, For, type JSX, Show } from 'solid-js';
import { Button } from '~/components/ui/Button';
import { Chip, ColorDot } from '~/components/ui/Chip';
import { Dialog, SheetFooter, SheetHeader } from '~/components/ui/Dialog';
import { Segmented } from '~/components/ui/Segmented';
import {
	activeFilterCount,
	artistOptions,
	COLORS,
	EMPTY_FILTERS,
	type FacetOption,
	type Filters,
	type FoilFilter,
	MANA_VALUES,
	RARITIES,
	setOptions,
	TYPES,
} from '~/lib/filters';

export interface FilterSheetProps {
	open: boolean;
	onClose: () => void;
	/** The whole library, for the set and artist lists. */
	cards: CardEntry[];
	filters: Filters;
	onChange: (filters: Filters) => void;
	/** How many cards the current view shows. */
	visible: number;
}

type ListField = 'sets' | 'rarities' | 'types' | 'artists' | 'colors' | 'manaValues';

const FOIL_OPTIONS: { value: FoilFilter; label: string }[] = [
	{ value: 'any', label: 'Any' },
	{ value: 'foil', label: 'Foil' },
	{ value: 'nonfoil', label: 'Not foil' },
];

const capitalize = (s: string) => s.charAt(0).toUpperCase() + s.slice(1);

export function FilterSheet(props: FilterSheetProps) {
	const sets = createMemo(() => setOptions(props.cards));
	const artists = createMemo(() => artistOptions(props.cards));
	const f = () => props.filters;
	const has = (field: ListField, v: string) => f()[field].includes(v);
	const toggle = (field: ListField, v: string) => {
		const list = f()[field];
		props.onChange({ ...f(), [field]: has(field, v) ? list.filter((x) => x !== v) : [...list, v] });
	};
	const summary = () => {
		if (props.visible === 0) return 'Nothing matches';
		return props.visible === 1 ? 'Show 1 card' : `Show ${props.visible} cards`;
	};

	return (
		<Dialog open={props.open} onClose={props.onClose} label="Filters">
			<SheetHeader title="Filters" onClose={props.onClose} />
			<div class="min-h-0 flex-1 overflow-y-auto px-5 py-4">
				<Section title="Color">
					<For each={COLORS}>
						{(c) => (
							<Chip pressed={has('colors', c.value)} onClick={() => toggle('colors', c.value)}>
								<ColorDot color={c.value} />
								{c.label}
							</Chip>
						)}
					</For>
				</Section>

				<Section title="Mana value">
					<For each={MANA_VALUES}>
						{(v) => (
							<Chip
								pressed={has('manaValues', v)}
								onClick={() => toggle('manaValues', v)}
								class="min-w-11 justify-center"
							>
								{v}
							</Chip>
						)}
					</For>
				</Section>

				<Section title="Rarity">
					<For each={RARITIES}>
						{(r) => (
							<Chip pressed={has('rarities', r)} onClick={() => toggle('rarities', r)}>
								{capitalize(r)}
							</Chip>
						)}
					</For>
				</Section>

				<Section title="Card type">
					<For each={TYPES}>
						{(t) => (
							<Chip pressed={has('types', t)} onClick={() => toggle('types', t)}>
								{capitalize(t)}
							</Chip>
						)}
					</For>
				</Section>

				<Section title="Printing">
					<Segmented
						label="Foil"
						options={FOIL_OPTIONS}
						value={f().foil}
						onChange={(foil) => props.onChange({ ...f(), foil })}
					/>
				</Section>

				<Section title="Only show">
					<Chip pressed={f().multiples} onClick={() => props.onChange({ ...f(), multiples: !f().multiples })}>
						More than one copy
					</Chip>
					<Chip pressed={f().attention} onClick={() => props.onChange({ ...f(), attention: !f().attention })}>
						Needs identifying
					</Chip>
				</Section>

				<Show when={sets().length > 0}>
					<Section title="Set" block>
						<CheckList
							options={sets()}
							selected={f().sets}
							onToggle={(v) => toggle('sets', v)}
							searchLabel="Find a set"
							showCode
						/>
					</Section>
				</Show>

				<Show when={artists().length > 0}>
					<Section title="Artist" block>
						<CheckList
							options={artists()}
							selected={f().artists}
							onToggle={(v) => toggle('artists', v)}
							searchLabel="Find an artist"
						/>
					</Section>
				</Show>
			</div>
			<SheetFooter>
				<Button
					variant="ghost"
					onClick={() => props.onChange(EMPTY_FILTERS)}
					disabled={activeFilterCount(f()) === 0}
				>
					Reset
				</Button>
				<div class="flex-1" />
				<Button variant="primary" onClick={props.onClose}>
					{summary()}
				</Button>
			</SheetFooter>
		</Dialog>
	);
}

function Section(props: { title: string; block?: boolean; children: JSX.Element }) {
	return (
		<section class="mt-6 first:mt-0">
			<h3 class="text-[14px] font-semibold">{props.title}</h3>
			<div class={props.block ? 'mt-2' : 'mt-2.5 flex flex-wrap gap-2'}>{props.children}</div>
		</section>
	);
}

/** Checkbox list with counts; gets a find box once it is long enough to need one. */
function CheckList(props: {
	options: FacetOption[];
	selected: string[];
	onToggle: (value: string) => void;
	searchLabel: string;
	showCode?: boolean;
}) {
	const [q, setQ] = createSignal('');
	const shown = () => {
		const s = q().trim().toLowerCase();
		if (!s) return props.options;
		return props.options.filter(
			(o) => o.label.toLowerCase().includes(s) || o.value.toLowerCase().includes(s),
		);
	};
	return (
		<div class="overflow-hidden rounded-control bg-raised">
			<Show when={props.options.length > 6}>
				<input
					type="search"
					placeholder={props.searchLabel}
					value={q()}
					onInput={(e) => setQ(e.currentTarget.value)}
					class="h-10 w-full border-b border-line bg-transparent px-3 text-[14px] text-ink outline-none placeholder:text-muted"
				/>
			</Show>
			<ul class="max-h-60 overflow-y-auto py-1">
				<For each={shown()}>
					{(o) => (
						<li>
							<label class="flex h-10 cursor-pointer items-center gap-3 px-3 text-[14px]">
								<input
									type="checkbox"
									checked={props.selected.includes(o.value)}
									onChange={() => props.onToggle(o.value)}
									class="h-4 w-4 accent-accent"
								/>
								<span class="flex-1 truncate">{o.label}</span>
								<Show when={props.showCode}>
									<span class="text-[12px] text-muted">{o.value.toUpperCase()}</span>
								</Show>
								<span class="w-6 text-right text-[12px] text-muted">{o.count}</span>
							</label>
						</li>
					)}
				</For>
				<Show when={shown().length === 0}>
					<li class="px-3 py-2 text-[14px] text-muted">Nothing matches</li>
				</Show>
			</ul>
		</div>
	);
}
