/** Toggle pill for multi-select facets and for active-filter tokens. */
import type { JSX } from 'solid-js';
import { cn } from '~/lib/cn';

export interface ChipProps {
	pressed?: boolean;
	onClick: () => void;
	size?: 'sm' | 'md';
	class?: string;
	'aria-label'?: string;
	children: JSX.Element;
}

export function Chip(props: ChipProps) {
	return (
		<button
			type="button"
			aria-pressed={props.pressed}
			aria-label={props['aria-label']}
			onClick={props.onClick}
			class={cn(
				'inline-flex shrink-0 items-center gap-1.5 rounded-full border font-medium whitespace-nowrap transition-colors',
				'focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-accent focus-visible:ring-offset-2 focus-visible:ring-offset-surface',
				props.size === 'sm' ? 'h-8 px-3 text-[13px]' : 'h-9 px-3.5 text-[14px]',
				props.pressed
					? 'border-accent bg-accent/15 text-ink'
					: 'border-line bg-surface text-ink hover:bg-raised',
				props.class,
			)}
		>
			{props.children}
		</button>
	);
}

/** Mana colour swatch; `M` is the five-colour wheel. */
const SWATCH: Record<string, string> = {
	W: '#efe7c9',
	U: '#2a7fd0',
	B: '#3a3340',
	R: '#d8463f',
	G: '#279a58',
	C: '#a3abb6',
	M: 'conic-gradient(#efe7c9, #2a7fd0, #3a3340, #d8463f, #279a58, #efe7c9)',
};

export function ColorDot(props: { color: string }) {
	return (
		<span
			aria-hidden="true"
			class="h-3.5 w-3.5 shrink-0 rounded-full shadow-[inset_0_0_0_1px_rgb(0_0_0/0.25)]"
			style={{ background: SWATCH[props.color] ?? SWATCH.C }}
		/>
	);
}
