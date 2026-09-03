/**
 * Settings: appearance, and moving the whole library in or out as a file.
 * Export offers Moxfield's own CSV layout (imports straight into a Moxfield
 * collection) and the app's full CSV; import reads a Moxfield collection
 * export.
 */
import type { ImportMode, ImportResponse } from '@shared/api';
import { FileDown, FileUp } from 'lucide-solid';
import { createSignal, Show } from 'solid-js';
import { Button, buttonClass } from '~/components/ui/Button';
import { Dialog, SheetHeader } from '~/components/ui/Dialog';
import { Segmented } from '~/components/ui/Segmented';
import { api, errorMessage } from '~/lib/api';
import { useLibrary } from '~/lib/library';
import { setTheme, THEME_OPTIONS, theme } from '~/lib/theme';
import { notify } from '~/lib/toast';

const MODES: { value: ImportMode; label: string }[] = [
	{ value: 'set', label: "Use the file's count" },
	{ value: 'add', label: 'Add its copies' },
];

const plural = (n: number, word: string) => `${n} ${n === 1 ? word : `${word}s`}`;

export function SettingsSheet(props: { open: boolean; onClose: () => void }) {
	const library = useLibrary();
	const [mode, setMode] = createSignal<ImportMode>('set');
	const [busy, setBusy] = createSignal(false);
	const [result, setResult] = createSignal<ImportResponse | null>(null);
	const [error, setError] = createSignal<string | null>(null);
	let input!: HTMLInputElement;

	const importFile = async (file: File | undefined) => {
		if (!file) return;
		setBusy(true);
		setError(null);
		setResult(null);
		try {
			const r = await api.importCsv(file, mode());
			setResult(r);
			library.refetch();
			notify(`Imported ${plural(r.rows, 'row')} from ${file.name}`);
		} catch (err) {
			setError(errorMessage(err));
		} finally {
			setBusy(false);
			input.value = '';
		}
	};

	return (
		<Dialog open={props.open} onClose={props.onClose} label="Settings">
			<SheetHeader title="Settings" onClose={props.onClose} />
			<div class="min-h-0 flex-1 overflow-y-auto px-5 py-4 pb-[max(1rem,env(safe-area-inset-bottom))]">
				<section>
					<h3 class="text-[14px] font-semibold">Appearance</h3>
					<p class="mt-1 text-[14px] text-muted">System follows your device's light or dark setting.</p>
					<Segmented
						class="mt-2.5"
						label="Appearance"
						options={THEME_OPTIONS}
						value={theme()}
						onChange={setTheme}
					/>
				</section>

				<section class="mt-7">
					<h3 class="text-[14px] font-semibold">Export</h3>
					<p class="mt-1 text-[14px] text-muted">
						The Moxfield file imports straight into a Moxfield collection. The full file also carries rarity,
						colours, mana value and this app's ids.
					</p>
					<div class="mt-3 flex flex-wrap gap-2">
						<a href={api.exportUrl('moxfield')} download="" class={buttonClass('primary')}>
							<FileDown class="h-5 w-5" />
							Moxfield CSV
						</a>
						<a href={api.exportUrl('full')} download="" class={buttonClass()}>
							Full CSV
						</a>
					</div>
				</section>

				<section class="mt-7">
					<h3 class="text-[14px] font-semibold">Import from Moxfield</h3>
					<p class="mt-1 text-[14px] text-muted">
						In Moxfield, export your collection as CSV and choose that file here. Cards are matched by set and
						collector number; any the index doesn't know are kept as "needs identifying".
					</p>
					<p class="mt-4 text-[14px]">If a card is already in your library</p>
					<Segmented
						class="mt-1.5"
						label="Existing cards"
						options={MODES}
						value={mode()}
						onChange={setMode}
					/>
					<input
						ref={input}
						type="file"
						accept=".csv,text/csv"
						class="hidden"
						onChange={(e) => importFile(e.currentTarget.files?.[0])}
					/>
					<Button variant="primary" class="mt-4 w-full" onClick={() => input.click()} disabled={busy()}>
						<FileUp class="h-5 w-5" />
						{busy() ? 'Importing…' : 'Choose CSV file'}
					</Button>
					<Show when={error()}>
						<p class="mt-3 text-[14px] text-bad">{error()}</p>
					</Show>
					<Show when={result()}>
						{(r) => (
							<div class="mt-4 rounded-control bg-raised px-4 py-3 text-[14px]">
								<p class="font-semibold">Read {plural(r().rows, 'row')}</p>
								<ul class="mt-1 space-y-0.5 text-muted">
									<li>{plural(r().added, 'new card')}</li>
									<li>
										{plural(r().updated, 'card')} already in your library,{' '}
										{mode() === 'add' ? 'copies added' : "count set to the file's"}
									</li>
									<Show when={r().unmatched > 0}>
										<li class="text-ink">
											{plural(r().unmatched, 'row')} not in the card index, kept as needs identifying:{' '}
											<span class="text-muted">{r().unmatched_names.join(', ')}</span>
										</li>
									</Show>
								</ul>
							</div>
						)}
					</Show>
				</section>
			</div>
		</Dialog>
	);
}
