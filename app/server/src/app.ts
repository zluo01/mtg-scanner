/**
 * HTTP surface. Everything the browser talks to lives here:
 *
 *   /api/*     JSON API (see shared/api.ts for the contract)
 *   /scans/*   user scan photos from DATA_DIR/scans
 *   /models/*  browser-side ONNX models from DATA_DIR/models
 *   /*         the built frontend (single-page app)
 *
 * Business logic stays in the modules this file wires together, so it can be
 * tested without HTTP and reused by another host (e.g. a Lambda adapter).
 */
import { serveStatic } from '@hono/node-server/serve-static';
import { Hono, type MiddlewareHandler } from 'hono';
import { bodyLimit } from 'hono/body-limit';
import { HTTPException } from 'hono/http-exception';
import type {
	AddCardResponse,
	CardEntry,
	CardResponse,
	DeleteResponse,
	ErrorResponse,
	HealthResponse,
	IdentifyResponse,
	ImportMode,
	ImportResponse,
	LibraryResponse,
	SearchResponse,
	UpdateCardRequest,
} from '../../shared/api.ts';
import { libraryToCsv, libraryToMoxfieldCsv } from './csv.ts';
import type { CardStore, StoredCard } from './db.ts';
import { badRequest, HttpError, notFound } from './errors.ts';
import type { CardIdentifier } from './identify.ts';
import { assertCardId } from './ids.ts';
import type { ImageStore } from './images.ts';
import { Library } from './library.ts';
import type { CardCatalog } from './metadata.ts';
import { importMoxfield, parseMoxfieldCsv } from './moxfield.ts';
import { addScannedCard, identifyScan } from './scan.ts';
import { MIN_QUERY_LENGTH, type NameSearch } from './search.ts';

export interface AppDeps {
	cards: CardStore;
	images: ImageStore;
	identifier: CardIdentifier;
	search: NameSearch;
	/** Index metadata; supplies the printing attributes attached to every card response. */
	catalog: CardCatalog;
	/** Directory for `/models/*` (browser-side ONNX files). */
	modelsDir: string;
	/** Built frontend directory, or `null` to run API-only. */
	webDist: string | null;
	/** Request logger; defaults to console. Pass `null` to silence. */
	log?: ((line: string) => void) | null;
}

const MAX_SCAN_BYTES = 12 * 1024 * 1024;
const MAX_JSON_BYTES = 64 * 1024;
/** A Moxfield export is ~120 bytes per row; this allows ~70K rows. */
const MAX_CSV_BYTES = 8 * 1024 * 1024;
const IMAGE_TYPES = new Set(['image/jpeg', 'image/png', 'application/octet-stream']);

const isRecord = (v: unknown): v is Record<string, unknown> =>
	typeof v === 'object' && v !== null && !Array.isArray(v);

async function readJson(c: { req: { json: () => Promise<unknown> } }): Promise<Record<string, unknown>> {
	let body: unknown;
	try {
		body = await c.req.json();
	} catch {
		throw badRequest('Request body must be valid JSON');
	}
	if (!isRecord(body)) throw badRequest('Request body must be a JSON object');
	return body;
}

function optionalNullableString(
	body: Record<string, unknown>,
	key: string,
	out: Record<string, unknown>,
): void {
	if (!(key in body)) return;
	const v = body[key];
	if (v !== null && typeof v !== 'string') throw badRequest(`${key} must be a string or null`);
	out[key] = v === '' ? null : v;
}

/** Validate a `PUT /api/cards/:id` body into a typed partial update. */
export function parseUpdateRequest(body: Record<string, unknown>): UpdateCardRequest {
	const out: Record<string, unknown> = {};
	optionalNullableString(body, 'scryfall_id', out);
	optionalNullableString(body, 'set_code', out);
	optionalNullableString(body, 'collector_number', out);
	if ('name' in body) {
		const name = body.name;
		if (typeof name !== 'string' || name.trim() === '') throw badRequest('name must be a non-empty string');
		out.name = name.trim();
	}
	if ('foil' in body) {
		if (typeof body.foil !== 'boolean') throw badRequest('foil must be a boolean');
		out.foil = body.foil;
	}
	if ('count' in body) {
		const count = body.count;
		if (typeof count !== 'number' || !Number.isInteger(count) || count < 1 || count > 9999) {
			throw badRequest('count must be an integer between 1 and 9999');
		}
		out.count = count;
	}
	return out as UpdateCardRequest;
}

function parseBool(value: string | undefined): boolean {
	return value === '1' || value === 'true';
}

/** The raw image body of a scan request (JPEG or PNG, non-empty). */
async function readImage(c: {
	req: { header: (name: string) => string | undefined; arrayBuffer: () => Promise<ArrayBuffer> };
}): Promise<Uint8Array> {
	const type = (c.req.header('content-type') ?? '').split(';')[0]?.trim().toLowerCase() ?? '';
	if (!IMAGE_TYPES.has(type)) throw badRequest('Body must be image/jpeg or image/png');
	const image = new Uint8Array(await c.req.arrayBuffer());
	if (image.byteLength === 0) throw badRequest('Empty image body');
	return image;
}

const IMMUTABLE = 'public, max-age=31536000, immutable';
const NO_CACHE = 'no-cache';

/** Set `Cache-Control` on successful responses produced downstream. */
function cacheControl(value: (path: string) => string): MiddlewareHandler {
	return async (c, next) => {
		await next();
		if (c.res.ok) c.res.headers.set('Cache-Control', value(c.req.path));
	};
}

export function createApp(deps: AppDeps): Hono {
	const { cards, images, identifier, search, catalog } = deps;
	const library = new Library(cards, images);
	const log = deps.log === undefined ? (line: string) => console.log(line) : deps.log;
	const app = new Hono();

	/** A stored row plus the attributes of its printing and whether a photo exists. */
	const enrich = (stored: StoredCard): CardEntry => ({
		...stored,
		...catalog.attributes(stored.scryfall_id),
		has_photo: images.has(stored.card_id),
	});

	app.use('*', async (c, next) => {
		const started = performance.now();
		await next();
		const status = c.res.status;
		if (log && (c.req.path.startsWith('/api/') || status >= 400)) {
			log(`${c.req.method} ${c.req.path} ${status} ${(performance.now() - started).toFixed(1)}ms`);
		}
	});

	app.onError((err, c) => {
		if (err instanceof HttpError) {
			const body: ErrorResponse = { error: err.message, status: err.status };
			return c.json(body, err.status as 400);
		}
		if (err instanceof HTTPException) {
			const body: ErrorResponse = { error: err.message || 'Request rejected', status: err.status };
			return c.json(body, err.status);
		}
		console.error(`${c.req.method} ${c.req.path} failed:`, err);
		const body: ErrorResponse = { error: 'Internal server error', status: 500 };
		return c.json(body, 500);
	});

	// ----------------------------------------------------------------- API

	app.get('/api/health', (c) => {
		const body: HealthResponse = { ok: true, cards_indexed: identifier.indexed, library_size: cards.count() };
		return c.json(body);
	});

	app.get('/api/library', (c) => {
		const body: LibraryResponse = { cards: cards.list().map(enrich) };
		return c.json(body);
	});

	// Step one of a scan: what does the index make of the photo? Stores nothing.
	app.post('/api/identify', bodyLimit({ maxSize: MAX_SCAN_BYTES }), async (c) => {
		const body: IdentifyResponse = await identifyScan(identifier, await readImage(c));
		return c.json(body);
	});

	// Step two: the user has confirmed the printing and foil; store the card with its photo.
	app.post('/api/cards', bodyLimit({ maxSize: MAX_SCAN_BYTES }), async (c) => {
		const cardId = assertCardId(c.req.query('card_id'));
		const scryfallId = c.req.query('scryfall_id') || null;
		const foil = parseBool(c.req.query('foil'));
		const image = await readImage(c);
		const body: AddCardResponse = await addScannedCard(
			{ library, catalog },
			{ cardId, scryfallId, foil, image },
		);
		return c.json(body, 201);
	});

	app.get('/api/cards/:id', (c) => {
		const card = cards.get(assertCardId(c.req.param('id')));
		if (!card) throw notFound('Card not found');
		const body: CardResponse = { card: enrich(card) };
		return c.json(body);
	});

	app.put('/api/cards/:id', bodyLimit({ maxSize: MAX_JSON_BYTES }), async (c) => {
		const id = assertCardId(c.req.param('id'));
		const patch = parseUpdateRequest(await readJson(c));
		const body: CardResponse = { card: enrich(await library.change(id, patch)) };
		return c.json(body);
	});

	app.delete('/api/cards/:id', async (c) => {
		const id = assertCardId(c.req.param('id'));
		if (!(await library.remove(id))) throw notFound('Card not found');
		const body: DeleteResponse = { success: true };
		return c.json(body);
	});

	app.get('/api/search', (c) => {
		const q = (c.req.query('q') ?? '').trim();
		if (q.length < MIN_QUERY_LENGTH) throw badRequest(`q must be at least ${MIN_QUERY_LENGTH} characters`);
		const body: SearchResponse = { cards: search.search(q) };
		return c.json(body);
	});

	app.get('/api/export', (c) => {
		const format = c.req.query('format') ?? 'full';
		if (format !== 'full' && format !== 'moxfield') throw badRequest('format must be "full" or "moxfield"');
		const date = new Date().toISOString().slice(0, 10);
		const library = cards.list().map(enrich);
		c.header('Content-Type', 'text/csv; charset=utf-8');
		if (format === 'moxfield') {
			c.header('Content-Disposition', `attachment; filename="moxfield-${date}.csv"`);
			// Moxfield names double-faced cards "Front // Back"; rows hold the front face.
			const named = library.map((card) => ({
				...card,
				name: catalog.fullName(card.scryfall_id) ?? card.name,
			}));
			return c.body(libraryToMoxfieldCsv(named));
		}
		c.header('Content-Disposition', `attachment; filename="mtg-library-${date}.csv"`);
		return c.body(libraryToCsv(library));
	});

	app.post('/api/import', bodyLimit({ maxSize: MAX_CSV_BYTES }), async (c) => {
		const mode = c.req.query('mode') ?? 'set';
		if (mode !== 'set' && mode !== 'add') throw badRequest('mode must be "set" or "add"');
		const text = await c.req.text();
		if (text.trim() === '') throw badRequest('Empty file');
		const rows = parseMoxfieldCsv(text);
		if (rows.length === 0) throw badRequest('No cards in the file');
		const body: ImportResponse = importMoxfield({ cards, catalog }, rows, mode as ImportMode);
		log?.(
			`import: ${body.rows} rows, ${body.added} added, ${body.updated} updated, ${body.unmatched} unmatched`,
		);
		return c.json(body);
	});

	app.all('/api/*', () => {
		throw notFound('No such endpoint');
	});

	// ------------------------------------------------------------- static

	app.use(
		'/scans/*',
		cacheControl(() => 'private, max-age=0, must-revalidate'),
		serveStatic({ root: images.dir, rewriteRequestPath: (p) => p.replace(/^\/scans/, '') }),
	);
	app.get('/scans/*', () => {
		throw notFound('No such scan');
	});

	app.use(
		'/models/*',
		cacheControl(() => 'public, max-age=86400'),
		serveStatic({ root: deps.modelsDir, rewriteRequestPath: (p) => p.replace(/^\/models/, '') }),
	);
	app.get('/models/*', () => {
		throw notFound('No such model');
	});

	if (deps.webDist) {
		const webDist = deps.webDist;
		// Vite emits hashed file names under /assets/, so those can be cached
		// forever; everything else (index.html, sw.js, manifest) revalidates.
		app.use(
			'/*',
			cacheControl((p) => (p.startsWith('/assets/') || p.endsWith('.wasm') ? IMMUTABLE : NO_CACHE)),
			serveStatic({ root: webDist }),
		);
		// Single-page app: every other GET renders the shell, except paths that
		// look like files (a missing hashed asset must be a 404, not HTML).
		app.get(
			'*',
			cacheControl(() => NO_CACHE),
			async (c, next) => {
				if (/\.[a-z0-9]+$/i.test(c.req.path)) throw notFound('Not found');
				await next();
			},
			serveStatic({ root: webDist, path: 'index.html' }),
		);
	}

	return app;
}
