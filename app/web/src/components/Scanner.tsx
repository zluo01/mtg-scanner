/**
 * The scanner: pick a photo (the system picker offers the camera on
 * phones), detect and rectify the cards on-device, identify them on the
 * server. One card goes straight to review; several become a batch.
 *
 * Over plain http (the usual LAN setup) `crypto.randomUUID` does not exist,
 * hence `lib/ids.ts`. Browsers cannot decode HEIC; with HEIC absent from the
 * accept list iOS transcodes to JPEG on selection.
 */
import type { CardEntry, IdentifyResponse } from '@shared/api';
import { Upload } from 'lucide-solid';
import { createEffect, createSignal, Match, onCleanup, Show, Switch } from 'solid-js';
import { BatchReview } from '~/components/BatchReview';
import { ScanReview } from '~/components/ScanReview';
import { Button } from '~/components/ui/Button';
import { Dialog, SheetHeader } from '~/components/ui/Dialog';
import { FoilToggle } from '~/components/ui/FoilToggle';
import { Spinner } from '~/components/ui/Spinner';
import { api, errorMessage } from '~/lib/api';
import { type DetectedCard, detectAndRectify, fileToBitmap, releaseDetections } from '~/lib/scan-pipeline';
import { notify } from '~/lib/toast';
import { loadDetector } from '~/lib/yolo';

type Phase =
	| { kind: 'pick' }
	| { kind: 'processing'; message: string }
	| { kind: 'single'; result: IdentifyResponse; detection: DetectedCard }
	| { kind: 'batch'; detections: DetectedCard[] };

/** Formats every browser can decode; deliberately not `image/*` (see header). */
const ACCEPT = 'image/jpeg,image/png,image/webp';
const isHeic = (file: File) => /^image\/hei[cf]$/i.test(file.type) || /\.hei[cf]$/i.test(file.name);
const isTouch = () => typeof window !== 'undefined' && window.matchMedia('(pointer: coarse)').matches;

export function Scanner(props: { open: boolean; onClose: () => void }) {
	const [foil, setFoil] = createSignal(false);
	const [phase, setPhase] = createSignal<Phase>({ kind: 'pick' });
	/** Message shown under the drop zone until dismissed (no card, or a failure). */
	const [notice, setNotice] = createSignal<string | null>(null);
	/** Object URL of the chosen photo, shown while it is processed. */
	const [preview, setPreview] = createSignal<string | null>(null);
	const [dragging, setDragging] = createSignal(false);

	let fileInput: HTMLInputElement | undefined;
	const processing = () => phase().kind === 'processing';

	// Warm the detector as soon as the sheet opens.
	createEffect(() => {
		if (props.open)
			loadDetector().catch((err) => setNotice(`The card detector did not load: ${errorMessage(err)}`));
	});

	const clearPreview = () => {
		const url = preview();
		if (url) URL.revokeObjectURL(url);
		setPreview(null);
	};
	onCleanup(clearPreview);

	const process = async (file: File) => {
		if (isHeic(file)) {
			setNotice(
				'This browser cannot read HEIC photos. On iPhone, set Settings > Camera > Formats to Most Compatible, or share the photo as JPEG.',
			);
			return;
		}
		clearPreview();
		setPreview(URL.createObjectURL(file));
		setNotice(null);
		setPhase({ kind: 'processing', message: 'Finding cards' });
		let detections: DetectedCard[] = [];
		try {
			const bitmap = await fileToBitmap(file);
			detections = await detectAndRectify(bitmap);
			bitmap.close();
			if (detections.length === 0) {
				setNotice('No card found. Fill the frame and keep the card flat.');
				setPhase({ kind: 'pick' });
				return;
			}
			if (detections.length === 1) {
				const only = detections[0]!;
				setPhase({ kind: 'processing', message: 'Identifying' });
				// Nothing is stored until the review's Add.
				const result = await api.identify(only.jpeg);
				setPhase({ kind: 'single', result, detection: only });
				return;
			}
			setPhase({ kind: 'batch', detections });
		} catch (err) {
			console.error('Scan failed:', err);
			releaseDetections(detections);
			setNotice(`Scan failed: ${errorMessage(err)}`);
			setPhase({ kind: 'pick' });
		}
	};

	const onPicked = (input: HTMLInputElement) => {
		const file = input.files?.[0];
		input.value = '';
		if (file) void process(file);
	};

	const backToPick = () => {
		const p = phase();
		if (p.kind === 'single') releaseDetections([p.detection]);
		if (p.kind === 'batch') releaseDetections(p.detections);
		clearPreview();
		setPhase({ kind: 'pick' });
	};

	const close = () => {
		backToPick();
		setNotice(null);
		props.onClose();
	};

	const added = (card: CardEntry) => {
		notify(`Added ${card.name}${card.foil ? ' (foil)' : ''}${card.count > 1 ? `, now ×${card.count}` : ''}`);
		backToPick();
	};

	return (
		<Dialog open={props.open} onClose={close} label="Scan cards">
			<Switch>
				<Match when={phase().kind === 'single'}>
					{(() => {
						const p = phase() as Extract<Phase, { kind: 'single' }>;
						return (
							<ScanReview
								result={p.result}
								cardId={p.detection.cardId}
								image={p.detection.jpeg}
								previewUrl={p.detection.previewUrl}
								foil={foil()}
								onClose={close}
								onAdded={added}
								onDiscarded={backToPick}
							/>
						);
					})()}
				</Match>

				<Match when={phase().kind === 'batch'}>
					<BatchReview
						detections={(phase() as Extract<Phase, { kind: 'batch' }>).detections}
						foil={foil()}
						onClose={close}
						onFinished={backToPick}
					/>
				</Match>

				<Match when={true}>
					<SheetHeader
						title="Scan"
						onClose={close}
						trailing={<FoilToggle on={foil()} onToggle={() => setFoil((f) => !f)} class="mr-2" />}
					/>
					<div class="px-5 pt-4 pb-[max(1.25rem,env(safe-area-inset-bottom))]">
						{/* biome-ignore lint/a11y/noStaticElementInteractions: drag-and-drop target only; the button inside is the pointer/keyboard control */}
						<div
							class={`relative flex min-h-[19rem] flex-col items-center justify-center gap-5 rounded-card border-2 border-dashed p-6 text-center transition-colors ${
								dragging() ? 'border-accent bg-accent/5' : 'border-line'
							}`}
							onDragOver={(e) => {
								e.preventDefault();
								setDragging(true);
							}}
							onDragLeave={() => setDragging(false)}
							onDrop={(e) => {
								e.preventDefault();
								setDragging(false);
								const file = e.dataTransfer?.files?.[0];
								if (file) void process(file);
							}}
						>
							<Show
								when={preview()}
								fallback={
									<div class="card-slot flex w-24 items-center justify-center text-muted">
										<Upload class="h-7 w-7" />
									</div>
								}
							>
								{(url) => (
									<img src={url()} alt="Selected for scanning" class="max-h-52 rounded-card object-contain" />
								)}
							</Show>
							<div>
								<p class="text-[17px] font-semibold">
									<Show when={preview()} fallback="One card, or a whole binder page">
										{processing() ? 'Working on it' : 'Choose another photo?'}
									</Show>
								</p>
								<p class="mt-1 text-[14px] text-muted">
									<Show when={isTouch()} fallback="Drop a photo here, or choose one from your files.">
										Take a new photo or pick one from your library.
									</Show>
								</p>
							</div>
							<Button variant="primary" size="lg" onClick={() => fileInput?.click()} disabled={processing()}>
								<Upload class="h-5 w-5" />
								Choose a photo
							</Button>

							<Show when={processing()}>
								<div class="absolute inset-0 flex flex-col items-center justify-center gap-3 rounded-card bg-surface/85">
									<Spinner class="h-8 w-8" />
									<p class="text-[15px] font-medium">
										{(phase() as Extract<Phase, { kind: 'processing' }>).message}
									</p>
								</div>
							</Show>
						</div>

						<Show when={notice()}>
							{(text) => (
								<div class="mt-3 flex items-center gap-3 rounded-control bg-raised py-2 pr-2 pl-4">
									<p class="flex-1 text-[14px]">{text()}</p>
									<Button size="sm" onClick={() => setNotice(null)}>
										OK
									</Button>
								</div>
							)}
						</Show>
					</div>
					<input
						ref={fileInput}
						type="file"
						accept={ACCEPT}
						class="hidden"
						onChange={(e) => onPicked(e.currentTarget)}
					/>
				</Match>
			</Switch>
		</Dialog>
	);
}
