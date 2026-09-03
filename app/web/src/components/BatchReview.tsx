/**
 * Several cards detected in one photo (a binder page). Every crop is
 * identified immediately so results are ready while the user reviews them
 * one at a time; each card enters the library only when the user adds it.
 */
import type { CardEntry, IdentifyResponse } from '@shared/api';
import { createSignal, For, Match, onCleanup, onMount, Show, Switch } from 'solid-js';
import { ScanReview } from '~/components/ScanReview';
import { Button } from '~/components/ui/Button';
import { CardImage } from '~/components/ui/CardImage';
import { SheetFooter, SheetHeader } from '~/components/ui/Dialog';
import { Spinner } from '~/components/ui/Spinner';
import { api, errorMessage } from '~/lib/api';
import { cn } from '~/lib/cn';
import type { DetectedCard } from '~/lib/scan-pipeline';

type Slot =
	| { status: 'pending' }
	| { status: 'ready'; result: IdentifyResponse }
	| { status: 'error'; message: string }
	| { status: 'done'; card: CardEntry | null };

export function BatchReview(props: {
	detections: DetectedCard[];
	foil: boolean;
	onFinished: () => void;
	onClose: () => void;
}) {
	const [slots, setSlots] = createSignal<Slot[]>(props.detections.map(() => ({ status: 'pending' })));
	const [index, setIndex] = createSignal(0);
	let cancelled = false;

	const setSlot = (i: number, slot: Slot) => setSlots((list) => list.map((s, j) => (j === i ? slot : s)));

	const submit = async (i: number) => {
		const det = props.detections[i]!;
		setSlot(i, { status: 'pending' });
		try {
			const result = await api.identify(det.jpeg);
			if (cancelled) return;
			setSlot(i, { status: 'ready', result });
		} catch (err) {
			if (!cancelled) setSlot(i, { status: 'error', message: errorMessage(err) });
		}
	};

	onMount(() => {
		for (let i = 0; i < props.detections.length; i++) void submit(i);
	});
	onCleanup(() => {
		cancelled = true;
	});

	const current = () => slots()[index()];
	const total = () => props.detections.length;
	const finished = () => slots().filter((s) => s.status === 'done').length;
	const allDone = () => finished() === total();
	const advance = (card: CardEntry | null) => {
		setSlot(index(), { status: 'done', card });
		if (index() + 1 < total()) setIndex(index() + 1);
	};

	const progress = (
		<fieldset class="m-0 flex items-center gap-1 border-0 p-0 pr-2">
			<legend class="sr-only">{`${finished()} of ${total()} reviewed`}</legend>
			<For each={slots()}>
				{(s, i) => (
					<span
						class={cn(
							'h-1.5 w-3.5 rounded-full',
							s.status === 'done'
								? 'bg-accent'
								: s.status === 'error'
									? 'bg-bad'
									: i() === index()
										? 'bg-accent/40'
										: 'bg-line',
						)}
					/>
				)}
			</For>
		</fieldset>
	);

	return (
		<>
			<SheetHeader
				title={allDone() ? 'Binder page' : `Card ${index() + 1} of ${total()}`}
				onClose={props.onClose}
				trailing={progress}
			/>
			<Switch>
				<Match when={allDone()}>
					<div class="flex flex-1 flex-col items-center justify-center px-5 py-8 text-center">
						<h3 class="text-[20px] font-semibold">All {total()} reviewed</h3>
						<p class="mt-2 text-[15px] text-muted">
							{slots().filter((s) => s.status === 'done' && s.card).length} added to your library.
						</p>
					</div>
					<SheetFooter>
						<div class="flex-1" />
						<Button variant="primary" onClick={props.onFinished}>
							Scan more
						</Button>
					</SheetFooter>
				</Match>
				<Match when={current()?.status === 'pending'}>
					<div class="flex flex-1 flex-col items-center justify-center gap-5 px-5 py-8">
						<CardImage src={props.detections[index()]!.previewUrl} class="w-32" />
						<div class="flex items-center gap-2 text-[14px] text-muted">
							<Spinner class="h-4 w-4" />
							Identifying
						</div>
					</div>
				</Match>
				<Match when={current()?.status === 'error'}>
					<div class="flex flex-1 flex-col items-center justify-center gap-4 px-5 py-8 text-center">
						<CardImage src={props.detections[index()]!.previewUrl} class="w-32" />
						<p class="text-[14px] text-bad">{(current() as { message: string }).message}</p>
					</div>
					<SheetFooter>
						<Button onClick={() => advance(null)}>Skip</Button>
						<div class="flex-1" />
						<Button variant="primary" onClick={() => submit(index())}>
							Try again
						</Button>
					</SheetFooter>
				</Match>
				<Match when={current()?.status === 'ready'}>
					<Show when={current()} keyed>
						{(slot) =>
							slot.status === 'ready' ? (
								<ScanReview
									embedded
									result={slot.result}
									cardId={props.detections[index()]!.cardId}
									image={props.detections[index()]!.jpeg}
									previewUrl={props.detections[index()]!.previewUrl}
									foil={props.foil}
									onClose={props.onClose}
									onAdded={(card) => advance(card)}
									onDiscarded={() => advance(null)}
								/>
							) : null
						}
					</Show>
				</Match>
			</Switch>
		</>
	);
}
