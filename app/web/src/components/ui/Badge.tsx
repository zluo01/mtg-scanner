import type { JSX } from 'solid-js';
import { cn } from '~/lib/cn';

export type BadgeVariant = 'count' | 'foil' | 'warn' | 'good' | 'muted';

const VARIANT: Record<BadgeVariant, string> = {
	count: 'bg-black/70 text-white',
	foil: 'bg-accent text-on-accent',
	warn: 'bg-bad text-white',
	good: 'bg-good text-white',
	muted: 'bg-raised text-muted',
};

export function Badge(props: { variant?: BadgeVariant; class?: string; children: JSX.Element }) {
	return (
		<span
			class={cn(
				'inline-flex items-center rounded-md px-1.5 py-0.5 text-[12px] font-semibold leading-4',
				VARIANT[props.variant ?? 'muted'],
				props.class,
			)}
		>
			{props.children}
		</span>
	);
}
