import { cn } from '~/lib/cn';

export function Spinner(props: { class?: string; light?: boolean }) {
	return (
		<div
			role="status"
			aria-label="Loading"
			class={cn(
				'animate-spin rounded-full border-2 border-t-transparent',
				props.light ? 'border-white' : 'border-accent',
				props.class ?? 'h-6 w-6',
			)}
		/>
	);
}
