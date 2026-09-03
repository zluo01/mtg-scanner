/** Foil on/off chip, used before capture, in review, and in card details. */
import { Sparkles } from 'lucide-solid';
import { cn } from '~/lib/cn';

export function FoilToggle(props: { on: boolean; onToggle: () => void; disabled?: boolean; class?: string }) {
	return (
		<button
			type="button"
			aria-pressed={props.on}
			disabled={props.disabled}
			onClick={props.onToggle}
			class={cn(
				'inline-flex h-9 items-center gap-1.5 rounded-control px-3 text-[13px] font-medium transition-colors disabled:opacity-45',
				props.on ? 'bg-accent text-on-accent' : 'bg-raised text-muted hover:text-ink',
				props.class,
			)}
		>
			<Sparkles class="h-4 w-4" />
			{props.on ? 'Foil' : 'Not foil'}
		</button>
	);
}
