/** Pick the library order. One tap selects and closes. */
import { Check } from 'lucide-solid';
import { For, Show } from 'solid-js';
import { Dialog, SheetHeader } from '~/components/ui/Dialog';
import { SORT_OPTIONS, type SortKey } from '~/lib/filters';

export function SortSheet(props: {
	open: boolean;
	onClose: () => void;
	value: SortKey;
	onChange: (sort: SortKey) => void;
}) {
	return (
		<Dialog open={props.open} onClose={props.onClose} label="Sort cards">
			<SheetHeader title="Sort by" onClose={props.onClose} />
			<ul class="py-2 pb-[max(0.5rem,env(safe-area-inset-bottom))]">
				<For each={SORT_OPTIONS}>
					{(o) => (
						<li>
							<button
								type="button"
								aria-pressed={props.value === o.value}
								onClick={() => {
									props.onChange(o.value);
									props.onClose();
								}}
								class="flex h-12 w-full items-center gap-3 px-5 text-left text-[15px] hover:bg-raised aria-pressed:font-semibold"
							>
								<span class="flex-1">{o.label}</span>
								<Show when={props.value === o.value}>
									<Check class="h-5 w-5 text-accent" />
								</Show>
							</button>
						</li>
					)}
				</For>
			</ul>
		</Dialog>
	);
}
