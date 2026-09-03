/** A card-shaped image frame; the one visual motif the whole app shares. */
import { ImageOff } from 'lucide-solid';
import { Show } from 'solid-js';
import { cn } from '~/lib/cn';

export function CardImage(props: {
	/** `null` when there is nothing to show (an unidentified card with no photo). */
	src: string | null;
	alt?: string;
	class?: string;
	onError?: (e: Event) => void;
}) {
	return (
		<div class={cn('card-frame', props.class)}>
			<Show
				when={props.src}
				fallback={
					<div
						class="absolute inset-0 flex items-center justify-center text-muted"
						role="img"
						aria-label={props.alt}
					>
						<ImageOff class="h-7 w-7" />
					</div>
				}
			>
				{(src) => (
					<img src={src()} alt={props.alt ?? ''} loading="lazy" decoding="async" onError={props.onError} />
				)}
			</Show>
		</div>
	);
}
