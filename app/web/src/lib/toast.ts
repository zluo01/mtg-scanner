/** Minimal transient notifications. */
import { createSignal } from 'solid-js';

export interface Toast {
	id: number;
	message: string;
	kind: 'info' | 'error';
}

const [toasts, setToasts] = createSignal<Toast[]>([]);
let nextId = 1;

export { toasts };

export function notify(message: string, kind: Toast['kind'] = 'info', ttlMs = 4000): void {
	const id = nextId++;
	setToasts((list) => [...list, { id, message, kind }]);
	setTimeout(() => dismiss(id), ttlMs);
}

export function dismiss(id: number): void {
	setToasts((list) => list.filter((t) => t.id !== id));
}
