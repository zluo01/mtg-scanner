/** A row of mutually exclusive options; replaces native <select> for short lists. */
import { For } from 'solid-js';
import { cn } from '~/lib/cn';

export interface SegmentedProps<T extends string> {
	options: { value: T; label: string }[];
	value: T;
	onChange: (value: T) => void;
	label: string;
	class?: string;
}

export function Segmented<T extends string>(props: SegmentedProps<T>) {
	return (
		<fieldset class={cn('m-0 inline-flex rounded-control border-0 bg-raised p-0.5', props.class)}>
			<legend class="sr-only">{props.label}</legend>
			<For each={props.options}>
				{(o) => (
					<button
						type="button"
						aria-pressed={props.value === o.value}
						onClick={() => props.onChange(o.value)}
						class={cn(
							'h-8 rounded-[9px] px-3 text-[13px] font-medium transition-colors',
							props.value === o.value ? 'bg-surface text-ink shadow-sm' : 'text-muted hover:text-ink',
						)}
					>
						{o.label}
					</button>
				)}
			</For>
		</fieldset>
	);
}
