/** A library card: view, adjust copies and foil, re-identify, delete. */
import type { CardEntry, ScryfallCard } from '@shared/api';
import { Minus, Plus } from 'lucide-solid';
import { createEffect, createSignal, Match, on, Show, Switch } from 'solid-js';
import { CardSearch } from '~/components/CardSearch';
import { Badge } from '~/components/ui/Badge';
import { Button } from '~/components/ui/Button';
import { CardImage } from '~/components/ui/CardImage';
import { Dialog, SheetFooter, SheetHeader } from '~/components/ui/Dialog';
import { FoilToggle } from '~/components/ui/FoilToggle';
import { Segmented } from '~/components/ui/Segmented';
import { api, errorMessage } from '~/lib/api';
import { showUserImages } from '~/lib/image-pref';
import { scryfallImageUrl, userImageUrl } from '~/lib/images';
import { useLibrary } from '~/lib/library';
import { notify } from '~/lib/toast';

type ImageView = 'art' | 'photo';
const IMAGE_OPTIONS: { value: ImageView; label: string }[] = [
	{ value: 'art', label: 'Card art' },
	{ value: 'photo', label: 'Your photo' },
];

const capitalize = (s: string) => s.charAt(0).toUpperCase() + s.slice(1);
/** Mana values are whole numbers except for a few half-mana cards. */
const formatMana = (v: number | null) => (v === null ? '' : Number.isInteger(v) ? String(v) : v.toFixed(1));

export function CardDetail(props: {
	card: CardEntry | null;
	onClose: () => void;
	/** The card folded into another (a foil flip or new printing it already had): show that one. */
	onSelect: (cardId: string) => void;
}) {
	const library = useLibrary();
	const [view, setView] = createSignal<'main' | 'search' | 'confirm-delete'>('main');
	const [busy, setBusy] = createSignal(false);
	const initialView = (): ImageView => (showUserImages() && props.card?.has_photo ? 'photo' : 'art');
	const [imageView, setImageView] = createSignal<ImageView>(initialView());

	const card = () => props.card;
	// Reset transient state whenever a different card is opened.
	createEffect(
		on(
			() => props.card?.card_id,
			() => {
				setView('main');
				setImageView(initialView());
			},
		),
	);

	/** The photo when chosen and present, else the printing's art, else the photo, else nothing. */
	const imageSrc = (): string | null => {
		const c = card();
		if (!c) return null;
		if (imageView() === 'photo' && c.has_photo) return userImageUrl(c);
		if (c.scryfall_id) return scryfallImageUrl(c.scryfall_id);
		return c.has_photo ? userImageUrl(c) : null;
	};

	const update = async (patch: Parameters<typeof api.update>[1]) => {
		const c = card();
		if (!c) return;
		setBusy(true);
		library.upsert({ ...c, ...patch } as CardEntry); // optimistic
		try {
			const { card: updated } = await api.update(c.card_id, patch);
			library.upsert(updated);
			if (updated.card_id !== c.card_id) {
				// Folded into the card that already had this printing + foil.
				library.remove(c.card_id);
				notify(`Joined the ${updated.name} you already had: now ×${updated.count}`);
				props.onSelect(updated.card_id);
			}
		} catch (err) {
			library.upsert(c);
			notify(errorMessage(err), 'error');
		} finally {
			setBusy(false);
		}
	};

	const identify = async (pick: ScryfallCard) => {
		await update({
			scryfall_id: pick.scryfall_id,
			name: pick.name,
			set_code: pick.set_code,
			collector_number: pick.collector_number,
		});
		setView('main');
	};

	const remove = async () => {
		const c = card();
		if (!c) return;
		setBusy(true);
		try {
			await api.remove(c.card_id);
			library.remove(c.card_id);
			notify(`Removed ${c.name}`);
			props.onClose();
		} catch (err) {
			notify(errorMessage(err), 'error');
		} finally {
			setBusy(false);
		}
	};

	return (
		<Dialog open={card() !== null} onClose={props.onClose} label="Card details">
			<Show when={card()}>
				{(c) => (
					<Switch>
						<Match when={view() === 'search'}>
							<SheetHeader
								title={c().scryfall_id ? 'Change printing' : 'Identify this card'}
								onClose={props.onClose}
							/>
							<CardSearch initialQuery={c().name === 'Unknown' ? '' : c().name} onSelect={identify} />
							<SheetFooter>
								<Button onClick={() => setView('main')} disabled={busy()}>
									Back
								</Button>
							</SheetFooter>
						</Match>
						<Match when={true}>
							<SheetHeader title={c().name} onClose={props.onClose} />
							<div class="min-h-0 flex-1 overflow-y-auto px-5 py-4">
								<CardImage src={imageSrc()} alt={c().name} class="mx-auto w-[min(62%,15rem)]" />
								<Show when={c().scryfall_id && c().has_photo}>
									<div class="mt-3 flex justify-center">
										<Segmented
											label="Image"
											options={IMAGE_OPTIONS}
											value={imageView()}
											onChange={setImageView}
										/>
									</div>
								</Show>

								<div class="mt-5 flex items-start justify-between gap-3">
									<div class="min-w-0">
										<h3 class="text-[20px] font-semibold leading-tight">{c().name}</h3>
										<Show when={c().set_code}>
											<p class="mt-0.5 text-[14px] text-muted">
												<Show when={c().set_name}>{c().set_name} </Show>
												<span class="text-ink/70">
													{c().set_code?.toUpperCase()} {c().collector_number}
												</span>
											</p>
										</Show>
										<Show when={c().type_line}>
											<p class="mt-2 text-[14px]">{c().type_line}</p>
										</Show>
										<Show when={c().rarity}>
											<p class="text-[14px] text-muted">
												{capitalize(c().rarity ?? '')}
												<Show when={c().mana_value !== null}>, mana value {formatMana(c().mana_value)}</Show>
											</p>
										</Show>
										<Show when={c().artist}>
											<p class="text-[14px] text-muted">Illustrated by {c().artist}</p>
										</Show>
									</div>
									<Show when={!c().scryfall_id}>
										<Badge variant="warn">Unidentified</Badge>
									</Show>
								</div>

								<div class="mt-5 flex items-center justify-between rounded-control bg-raised py-1.5 pr-1.5 pl-3">
									<span class="text-[14px]">Copies</span>
									<div class="flex items-center">
										<Button
											variant="ghost"
											size="icon"
											class="h-9 w-9"
											onClick={() => update({ count: c().count - 1 })}
											disabled={busy() || c().count <= 1}
											aria-label="One fewer copy"
										>
											<Minus class="h-4 w-4" />
										</Button>
										<span class="w-8 text-center text-[17px] font-semibold">{c().count}</span>
										<Button
											variant="ghost"
											size="icon"
											class="h-9 w-9"
											onClick={() => update({ count: c().count + 1 })}
											disabled={busy()}
											aria-label="One more copy"
										>
											<Plus class="h-4 w-4" />
										</Button>
									</div>
								</div>

								<div class="mt-3 flex items-center justify-between rounded-control bg-raised py-1.5 pr-1.5 pl-3">
									<span class="text-[14px]">Foil printing</span>
									<FoilToggle on={c().foil} onToggle={() => update({ foil: !c().foil })} disabled={busy()} />
								</div>

								<Button class="mt-5 w-full" onClick={() => setView('search')} disabled={busy()}>
									{c().scryfall_id ? 'Wrong card?' : 'Identify this card'}
								</Button>

								<Show
									when={view() === 'confirm-delete'}
									fallback={
										<Button
											variant="danger"
											class="mt-2 w-full"
											onClick={() => setView('confirm-delete')}
											disabled={busy()}
										>
											Delete from library
										</Button>
									}
								>
									<div class="mt-2 flex items-center gap-2 rounded-control border border-bad/40 py-2 pr-2 pl-3">
										<span class="flex-1 text-[14px]">Delete this card and its photo?</span>
										<Button size="sm" onClick={() => setView('main')} disabled={busy()}>
											Cancel
										</Button>
										<Button
											size="sm"
											variant="primary"
											class="bg-bad text-white"
											onClick={remove}
											disabled={busy()}
										>
											Delete
										</Button>
									</div>
								</Show>
							</div>
						</Match>
					</Switch>
				)}
			</Show>
		</Dialog>
	);
}
