/**
 * Typed client for the server API (contract in shared/api.ts). All paths
 * are same-origin: Vite proxies them in development, the server serves the
 * built app in production.
 */
import type {
	AddCardResponse,
	CardResponse,
	DeleteResponse,
	ErrorResponse,
	ExportFormat,
	IdentifyResponse,
	ImportMode,
	ImportResponse,
	LibraryResponse,
	SearchResponse,
	UpdateCardRequest,
} from '@shared/api';

export class ApiError extends Error {
	readonly status: number;

	constructor(status: number, message: string) {
		super(message);
		this.name = 'ApiError';
		this.status = status;
	}
}

async function request<T>(path: string, init: RequestInit = {}): Promise<T> {
	let res: Response;
	try {
		res = await fetch(path, init);
	} catch {
		throw new ApiError(0, 'Network error. Is the server running?');
	}
	if (!res.ok) {
		let message = `Request failed (${res.status})`;
		try {
			const body = (await res.json()) as Partial<ErrorResponse>;
			if (typeof body.error === 'string') message = body.error;
		} catch {
			// non-JSON error body; keep the generic message
		}
		throw new ApiError(res.status, message);
	}
	return (await res.json()) as T;
}

const JSON_HEADERS = { 'content-type': 'application/json' };

export const api = {
	library: () => request<LibraryResponse>('/api/library'),

	/** What the index makes of a rectified card photo. Stores nothing. */
	identify: (image: Blob) =>
		request<IdentifyResponse>('/api/identify', {
			method: 'POST',
			body: image,
			headers: { 'content-type': image.type || 'image/jpeg' },
		}),

	/** Add the scanned card as confirmed, with its photo. */
	addScan: (cardId: string, image: Blob, printing: { scryfall_id: string | null; foil: boolean }) =>
		request<AddCardResponse>(
			`/api/cards?card_id=${encodeURIComponent(cardId)}&scryfall_id=${encodeURIComponent(printing.scryfall_id ?? '')}&foil=${printing.foil ? 1 : 0}`,
			{ method: 'POST', body: image, headers: { 'content-type': image.type || 'image/jpeg' } },
		),

	update: (cardId: string, patch: UpdateCardRequest) =>
		request<CardResponse>(`/api/cards/${encodeURIComponent(cardId)}`, {
			method: 'PUT',
			body: JSON.stringify(patch),
			headers: JSON_HEADERS,
		}),

	remove: (cardId: string) =>
		request<DeleteResponse>(`/api/cards/${encodeURIComponent(cardId)}`, { method: 'DELETE' }),

	search: (query: string, signal?: AbortSignal) =>
		request<SearchResponse>(`/api/search?q=${encodeURIComponent(query)}`, { signal }),

	/** Upload a Moxfield collection CSV. */
	importCsv: (file: Blob, mode: ImportMode) =>
		request<ImportResponse>(`/api/import?mode=${mode}`, {
			method: 'POST',
			body: file,
			headers: { 'content-type': 'text/csv' },
		}),

	exportUrl: (format: ExportFormat = 'full') =>
		format === 'full' ? '/api/export' : `/api/export?format=${format}`,
};

export function errorMessage(err: unknown): string {
	if (err instanceof Error) return err.message;
	return 'Something went wrong';
}
