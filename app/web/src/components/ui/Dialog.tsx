/**
 * Sheet on the native <dialog> element: focus trapping, Escape and the top
 * layer come from the browser. Bottom sheet on phones (swipe down to
 * close), centred modal on wider screens (see styles.css `dialog.sheet`).
 *
 * What the browser does not do is keep the page behind a modal dialog
 * still: the backdrop is not a scroll container, so wheel, keys and touch
 * chain straight through to the document. styles.css locks the root while
 * a sheet is open (wheel and keys); the touch handler here lets a move
 * through only when a scrollable region inside the sheet can take it.
 */
import { X } from 'lucide-solid';
import { createEffect, type JSX, onCleanup, onMount, Show } from 'solid-js';
import { sheetFit } from '~/lib/sheet-fit';

export interface DialogProps {
	open: boolean;
	onClose: () => void;
	/** Accessible name. */
	label: string;
	children: JSX.Element;
}

/** A sheet released this far down (px) closes. */
const CLOSE_DISTANCE = 96;
/** Or a shorter but quick flick (px per ms). */
const CLOSE_VELOCITY = 0.6;

const isBottomSheet = () => !window.matchMedia('(min-width: 640px)').matches;
const scrolls = (overflow: string) => overflow === 'auto' || overflow === 'scroll';

/** True when something between the touch target and the sheet is scrolled down, so the finger should scroll it instead of dragging. */
function insideScrolledRegion(target: EventTarget | null, root: HTMLElement): boolean {
	for (let el = target as HTMLElement | null; el && el !== root; el = el.parentElement) {
		if (el.scrollTop > 0) return true;
	}
	return false;
}

/** A scrollable region inside the sheet that still has room in the finger's direction. */
function canScroll(target: EventTarget | null, root: HTMLElement, dx: number, dy: number): boolean {
	const horizontal = Math.abs(dx) > Math.abs(dy);
	for (let el = target as HTMLElement | null; el && el !== root; el = el.parentElement) {
		const style = getComputedStyle(el);
		if (horizontal) {
			if (scrolls(style.overflowX) && el.scrollWidth > el.clientWidth) {
				// Finger left -> content moves right -> needs room on the right.
				if (dx < 0 ? el.scrollLeft + el.clientWidth < el.scrollWidth - 1 : el.scrollLeft > 0) return true;
			}
		} else if (scrolls(style.overflowY) && el.scrollHeight > el.clientHeight) {
			if (dy < 0 ? el.scrollTop + el.clientHeight < el.scrollHeight - 1 : el.scrollTop > 0) return true;
		}
	}
	return false;
}

export function Dialog(props: DialogProps) {
	let el!: HTMLDialogElement;
	let fit: (() => void) | undefined;

	createEffect(() => {
		if (props.open) {
			if (!el.open) el.showModal();
		} else if (el.open) {
			el.close();
		}
		fit?.();
	});
	onCleanup(() => {
		if (el?.open) el.close();
	});

	// The on-screen keyboard shrinks the visual viewport only (see
	// lib/sheet-fit.ts): keep the sheet inside what is visible.
	onMount(() => {
		const vv = window.visualViewport;
		if (!vv) return;
		fit = () => {
			if (!el.open || !isBottomSheet()) {
				el.style.bottom = '';
				el.style.removeProperty('--sheet-max');
				return;
			}
			const { bottom, maxHeight } = sheetFit(window.innerHeight, vv);
			el.style.bottom = `${bottom}px`;
			el.style.setProperty('--sheet-max', `${maxHeight}px`);
		};
		vv.addEventListener('resize', fit);
		vv.addEventListener('scroll', fit);
		onCleanup(() => {
			vv.removeEventListener('resize', fit!);
			vv.removeEventListener('scroll', fit!);
		});
	});

	// Touch: swipe down to close on phones (the sheet follows the finger
	// once a touch starts on something that cannot scroll up; releasing past
	// CLOSE_DISTANCE or flicking closes it, less springs back), and never let
	// a move reach the page behind.
	onMount(() => {
		let startX = 0;
		let startY = 0;
		let startTime = 0;
		let offset = 0;
		let mayDrag = false;
		let dragging = false;

		const reset = () => {
			el.style.transition = '';
			el.style.transform = '';
		};
		const onStart = (e: TouchEvent) => {
			const t = e.touches[0];
			if (!t || e.touches.length !== 1) return;
			startX = t.clientX;
			startY = t.clientY;
			startTime = e.timeStamp;
			offset = 0;
			dragging = false;
			mayDrag = isBottomSheet() && !insideScrolledRegion(e.target, el);
		};
		const onMove = (e: TouchEvent) => {
			const t = e.touches[0];
			if (!t) return;
			const dx = t.clientX - startX;
			const dy = t.clientY - startY;
			if (dragging || (mayDrag && dy > 0 && dy > Math.abs(dx) && !insideScrolledRegion(e.target, el))) {
				dragging = true;
				offset = Math.max(0, dy);
				e.preventDefault();
				el.style.transition = 'none';
				el.style.transform = `translateY(${offset}px)`;
				return;
			}
			if (!canScroll(e.target, el, dx, dy)) e.preventDefault();
		};
		const onEnd = (e: TouchEvent) => {
			if (!dragging) return;
			dragging = false;
			const velocity = offset / Math.max(1, e.timeStamp - startTime);
			if (offset > CLOSE_DISTANCE || (offset > 24 && velocity > CLOSE_VELOCITY)) {
				let done = false;
				const finish = () => {
					if (done) return;
					done = true;
					el.removeEventListener('transitionend', finish);
					reset();
					props.onClose();
				};
				el.addEventListener('transitionend', finish);
				setTimeout(finish, 200);
				el.style.transition = 'transform 160ms ease-in';
				el.style.transform = 'translateY(100%)';
			} else {
				el.style.transition = 'transform 200ms cubic-bezier(0.2, 0.8, 0.2, 1)';
				el.style.transform = '';
				setTimeout(reset, 220);
			}
		};

		el.addEventListener('touchstart', onStart, { passive: true });
		el.addEventListener('touchmove', onMove, { passive: false });
		el.addEventListener('touchend', onEnd);
		el.addEventListener('touchcancel', onEnd);
		onCleanup(() => {
			el.removeEventListener('touchstart', onStart);
			el.removeEventListener('touchmove', onMove);
			el.removeEventListener('touchend', onEnd);
			el.removeEventListener('touchcancel', onEnd);
		});
	});

	return (
		// biome-ignore lint/a11y/useKeyWithClickEvents: the click closes on backdrop only; keyboard users close with Escape via onCancel
		<dialog
			ref={el}
			aria-label={props.label}
			class="sheet"
			onCancel={(e) => {
				// A dismissed file picker fires a bubbling `cancel` on its <input>;
				// only the dialog's own cancel (Escape) should close the sheet.
				if (e.target !== el) return;
				e.preventDefault();
				props.onClose();
			}}
			onClick={(e) => {
				if (e.target === el) props.onClose();
			}}
		>
			<Show when={props.open}>
				<div class="sheet-panel flex flex-col overflow-hidden sm:max-h-[inherit]">
					<div class="mx-auto mt-2 h-1 w-10 shrink-0 rounded-full bg-line sm:hidden" aria-hidden="true" />
					{props.children}
				</div>
			</Show>
		</dialog>
	);
}

/** Standard sheet header: title, optional trailing control, close. */
export function SheetHeader(props: { title: string; onClose: () => void; trailing?: JSX.Element }) {
	return (
		<div class="flex h-14 shrink-0 items-center gap-2 border-b border-line px-2">
			<button
				type="button"
				onClick={props.onClose}
				aria-label="Close"
				class="flex h-11 w-11 items-center justify-center rounded-control text-ink hover:bg-raised"
			>
				<X class="h-5 w-5" />
			</button>
			<h2 class="flex-1 truncate text-[17px] font-semibold">{props.title}</h2>
			{props.trailing}
		</div>
	);
}

/** Standard sheet footer: quiet/destructive on the left, primary on the right. */
export function SheetFooter(props: { children: JSX.Element }) {
	return (
		<div class="flex shrink-0 items-center gap-2 border-t border-line px-4 py-3 pb-[max(0.75rem,env(safe-area-inset-bottom))]">
			{props.children}
		</div>
	);
}
