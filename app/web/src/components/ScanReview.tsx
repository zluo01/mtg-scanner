/**
 * Review one scan before it goes into the library. Nothing has been stored
 * yet: the server only identified the photo. The user confirms the
 * printing, corrects it (candidates or search), sets foil, and Add writes
 * the card with its photo in one call; Discard or closing the sheet leaves
 * nothing behind. A card already owned in that printing + foil simply gains
 * a copy (the server folds it), and the review says so beforehand.
 */
import type { CardEntry, Confidence, IdentifyResponse, ScanCandidate, ScryfallCard } from '@shared/api';
import { createMemo, createSignal, For, type JSX, Match, Show, Switch } from 'solid-js';
import { CardSearch } from '~/components/CardSearch';
import { Badge } from '~/components/ui/Badge';
import { Button } from '~/components/ui/Button';
import { CardImage } from '~/components/ui/CardImage';
import { SheetFooter, SheetHeader } from '~/components/ui/Dialog';
import { FoilToggle } from '~/components/ui/FoilToggle';
import { api, errorMessage } from '~/lib/api';
import { findOwned } from '~/lib/duplicates';
import { scryfallImageUrl } from '~/lib/images';
import { useLibrary } from '~/lib/library';
import { notify } from '~/lib/toast';

export interface ScanReviewProps {
	result: IdentifyResponse;
	/** Client-generated id the card will get. */
	cardId: string;
	/** The rectified card photo to store, and its object URL for display. */
	image: Blob;
	previewUrl: string;
	/** Foil as set in the scanner; editable here. */
	foil: boolean;
	/** The card was added (or folded into) `card`. */
	onAdded: (card: CardEntry) => void;
	/** The scan was dropped; nothing was stored. */
	onDiscarded: () => void;
	onClose: () => void;
	/** Inside a batch: the parent owns the sheet header. */
	embedded?: boolean;
}

type View = 'main' | 'candidates' | 'search';

/** The printing the user has settled on; `null` = add as unidentified. */
type Draft = Pick<ScryfallCard, 'scryfall_id' | 'name' | 'set_code' | 'collector_number' | 'artist'> | null;

export function ScanReview(props: ScanReviewProps) {
	const library = useLibrary();
	const top = props.result.candidates[0];
	const [draft, setDraft] = createSignal<Draft>(props.result.confidence === 'NO_MATCH' || !top ? null : top);
	const [foil, setFoil] = createSignal(props.foil);
	const [view, setView] = createSignal<View>('main');
	const [busy, setBusy] = createSignal(false);
	/** Becomes CONFIDENT once the user has picked the printing themselves. */
	const [confidence, setConfidence] = createSignal<Confidence>(props.result.confidence);
	const [corrected, setCorrected] = createSignal(false);

	const pct = () => Math.round(props.result.similarity * 100);
	const alternatives = () => props.result.candidates.filter((c) => c.scryfall_id !== draft()?.scryfall_id);
	/** The card this scan will join, if the printing + foil is already owned. */
	const owned = createMemo(() => findOwned(library.cards(), draft()?.scryfall_id ?? null, foil()));

	/** Pick a different printing; back to the main view so foil can still be set. */
	const correct = (pick: ScryfallCard | ScanCandidate) => {
		setDraft({
			scryfall_id: pick.scryfall_id,
			name: pick.name,
			set_code: pick.set_code,
			collector_number: pick.collector_number,
			artist: pick.artist,
		});
		setCorrected(true);
		setConfidence('CONFIDENT');
		setView('main');
	};

	const add = async () => {
		setBusy(true);
		try {
			const { card } = await api.addScan(props.cardId, props.image, {
				scryfall_id: draft()?.scryfall_id ?? null,
				foil: foil(),
			});
			library.upsert(card);
			props.onAdded(card);
		} catch (err) {
			notify(errorMessage(err), 'error');
		} finally {
			setBusy(false);
		}
	};

	const Title = (p: { title: string; trailing?: JSX.Element }) => (
		<Show
			when={!props.embedded}
			fallback={<h3 class="shrink-0 px-5 pt-4 text-[17px] font-semibold">{p.title}</h3>}
		>
			<SheetHeader title={p.title} onClose={props.onClose} trailing={p.trailing} />
		</Show>
	);

	const CandidateTile = (p: { candidate: ScanCandidate }) => (
		<button
			type="button"
			onClick={() => correct(p.candidate)}
			disabled={busy()}
			class="group text-left disabled:opacity-45"
		>
			<CardImage
				src={scryfallImageUrl(p.candidate.scryfall_id)}
				class="group-hover:ring-2 group-hover:ring-accent"
			/>
			<p class="mt-1.5 truncate text-[13px] font-medium">{p.candidate.name}</p>
			<p class="text-[12px] text-muted">
				{p.candidate.set_code.toUpperCase()} {p.candidate.collector_number}
			</p>
			<p class="text-[12px] text-muted">{Math.round(p.candidate.similarity * 100)}% similar</p>
		</button>
	);

	const mainTitle = () => {
		if (corrected()) return 'Card chosen';
		return confidence() === 'CONFIDENT' ? 'Match found' : 'Best guess';
	};

	return (
		<Switch>
			<Match when={view() === 'search'}>
				<Title title="Find the card" />
				<CardSearch initialQuery={draft()?.name ?? ''} onSelect={correct} />
				<SheetFooter>
					<Button onClick={() => setView('main')} disabled={busy()}>
						Back
					</Button>
				</SheetFooter>
			</Match>

			<Match when={view() === 'candidates'}>
				<Title title="Pick the printing" />
				<div class="min-h-0 flex-1 overflow-y-auto px-5 py-4">
					<p class="mb-4 text-[14px] text-muted">The closest printings in the index. Tap the right one.</p>
					<div class="grid grid-cols-3 gap-3">
						<For each={alternatives()}>{(c) => <CandidateTile candidate={c} />}</For>
					</div>
				</div>
				<SheetFooter>
					<Button onClick={() => setView('main')} disabled={busy()}>
						Back
					</Button>
					<div class="flex-1" />
					<Button onClick={() => setView('search')} disabled={busy()}>
						Search instead
					</Button>
				</SheetFooter>
			</Match>

			<Match when={confidence() === 'NO_MATCH'}>
				<Title title="Not recognised" />
				<div class="flex min-h-0 flex-1 flex-col items-center overflow-y-auto px-5 py-4 text-center">
					<CardImage src={props.previewUrl} class="w-36" />
					<p class="mt-5 max-w-xs text-[15px] text-muted">
						No printing in the index looks like this photo. Search for it now, or add it as unidentified and
						fix it later from your library.
					</p>
					<Show when={alternatives().length > 0}>
						<Button
							variant="ghost"
							size="sm"
							class="mt-3"
							onClick={() => setView('candidates')}
							disabled={busy()}
						>
							Show the closest guesses
						</Button>
					</Show>
				</div>
				<SheetFooter>
					<Button variant="danger" onClick={props.onDiscarded} disabled={busy()}>
						Discard
					</Button>
					<div class="flex-1" />
					<Button onClick={add} disabled={busy()}>
						{busy() ? 'Adding' : 'Add as unidentified'}
					</Button>
					<Button variant="primary" onClick={() => setView('search')} disabled={busy()}>
						Search
					</Button>
				</SheetFooter>
			</Match>

			<Match when={true}>
				<Title title={mainTitle()} />
				<div class="min-h-0 flex-1 overflow-y-auto px-5 py-4">
					<div class="grid grid-cols-2 gap-4">
						<figure>
							<CardImage src={props.previewUrl} />
							<figcaption class="mt-1.5 text-center text-[12px] text-muted">Your photo</figcaption>
						</figure>
						<Show when={draft()}>
							{(d) => (
								<figure>
									<CardImage src={scryfallImageUrl(d().scryfall_id)} />
									<figcaption class="mt-1.5 text-center text-[12px] text-muted">
										<Show when={!corrected()} fallback="Your pick">
											<Badge variant={confidence() === 'CONFIDENT' ? 'good' : 'muted'}>{pct()}% match</Badge>
										</Show>
									</figcaption>
								</figure>
							)}
						</Show>
					</div>

					<h3 class="mt-5 text-[20px] font-semibold leading-tight">{draft()?.name ?? 'Unknown'}</h3>
					<p class="mt-0.5 text-[14px] text-muted">
						{draft()?.set_code.toUpperCase()} {draft()?.collector_number}
					</p>
					<Show when={draft()?.artist}>
						<p class="text-[14px] text-muted">{draft()?.artist}</p>
					</Show>
					<Show when={owned()}>
						{(o) => (
							<p class="mt-2 text-[14px] text-accent">
								You already have this card{foil() ? ' in foil' : ''} ×{o().count}. Adding makes it ×
								{o().count + 1}.
							</p>
						)}
					</Show>

					<div class="mt-4 flex items-center justify-between rounded-control bg-raised py-1.5 pr-1.5 pl-3">
						<span class="text-[14px]">Foil printing</span>
						<FoilToggle on={foil()} onToggle={() => setFoil((f) => !f)} disabled={busy()} />
					</div>

					<Show when={confidence() === 'AMBIGUOUS'}>
						<p class="mt-4 text-[14px] text-muted">
							This match isn't certain. Compare the two images, or pick a different printing.
						</p>
					</Show>
				</div>
				<SheetFooter>
					<Button variant="danger" onClick={props.onDiscarded} disabled={busy()}>
						Discard
					</Button>
					<div class="flex-1" />
					<Button
						onClick={() => setView(alternatives().length > 0 ? 'candidates' : 'search')}
						disabled={busy()}
					>
						Wrong card
					</Button>
					<Button variant="primary" onClick={add} disabled={busy()}>
						{busy() ? 'Adding' : 'Add'}
					</Button>
				</SheetFooter>
			</Match>
		</Switch>
	);
}
