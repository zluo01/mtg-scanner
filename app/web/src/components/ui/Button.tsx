import { type JSX, splitProps } from 'solid-js';
import { cn } from '~/lib/cn';

export type ButtonVariant = 'primary' | 'secondary' | 'ghost' | 'danger';
export type ButtonSize = 'sm' | 'md' | 'lg' | 'icon';

const BASE =
	'inline-flex items-center justify-center gap-2 rounded-control font-medium whitespace-nowrap select-none ' +
	'transition-[background-color,opacity] duration-150 focus-visible:outline-none focus-visible:ring-2 ' +
	'focus-visible:ring-accent focus-visible:ring-offset-2 focus-visible:ring-offset-bg ' +
	'disabled:pointer-events-none disabled:opacity-45';

const VARIANT: Record<ButtonVariant, string> = {
	primary: 'bg-accent text-on-accent hover:brightness-105 active:brightness-95',
	secondary: 'bg-raised text-ink hover:bg-line/70',
	ghost: 'text-ink hover:bg-raised',
	danger: 'text-bad hover:bg-bad/10',
};

const SIZE: Record<ButtonSize, string> = {
	sm: 'h-9 px-3 text-[13px]',
	md: 'h-11 px-4 text-[15px]',
	lg: 'h-12 px-6 text-[15px]',
	icon: 'h-11 w-11',
};

export interface ButtonProps extends JSX.ButtonHTMLAttributes<HTMLButtonElement> {
	variant?: ButtonVariant;
	size?: ButtonSize;
}

/** The button look for elements that are not buttons (download links). */
export function buttonClass(
	variant: ButtonVariant = 'secondary',
	size: ButtonSize = 'md',
	extra?: string,
): string {
	return cn(BASE, VARIANT[variant], SIZE[size], extra);
}

export function Button(props: ButtonProps) {
	const [local, rest] = splitProps(props, ['variant', 'size', 'class', 'type']);
	return (
		<button
			type={local.type ?? 'button'}
			class={buttonClass(local.variant, local.size, local.class)}
			{...rest}
		/>
	);
}
