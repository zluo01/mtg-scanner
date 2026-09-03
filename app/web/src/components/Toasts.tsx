import { createEffect, For } from 'solid-js';
import { cn } from '~/lib/cn';
import { dismiss, toasts } from '~/lib/toast';

/**
 * Transient messages. Rendered as a manual popover so they sit in the
 * browser's top layer, above any open sheet.
 */
export function Toasts() {
	let el!: HTMLDivElement;

	createEffect(() => {
		const show = toasts().length > 0;
		try {
			if (show && !el.matches(':popover-open')) el.showPopover();
			if (!show && el.matches(':popover-open')) el.hidePopover();
		} catch {
			// Older browsers without the Popover API fall back to the fixed layout in CSS.
		}
	});

	return (
		<div ref={el} popover="manual" class="toast-layer flex flex-col items-center gap-2">
			<For each={toasts()}>
				{(t) => (
					<button
						type="button"
						onClick={() => dismiss(t.id)}
						class={cn(
							'pointer-events-auto max-w-md rounded-control px-4 py-2.5 text-[14px] font-medium shadow-lg',
							t.kind === 'error' ? 'bg-bad text-white' : 'bg-ink text-bg',
						)}
					>
						{t.message}
					</button>
				)}
			</For>
		</div>
	);
}
