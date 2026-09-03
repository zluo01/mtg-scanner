import assert from 'node:assert/strict';
import { mkdir, writeFile } from 'node:fs/promises';
import path from 'node:path';
import { test } from 'node:test';
import type {
	AddCardResponse,
	CardResponse,
	ErrorResponse,
	HealthResponse,
	IdentifyResponse,
	ImportResponse,
	LibraryResponse,
	SearchResponse,
} from '../../shared/api.ts';
import { createApp } from '../src/app.ts';
import { NameSearch } from '../src/search.ts';
import { FAKE_JPEG, fixedIdentifier, meta, stores } from './helpers.ts';

async function harness(hits = [{ card: meta(), similarity: 0.95 }]) {
	const s = await stores();
	const modelsDir = path.join(s.dir, 'models');
	const webDist = path.join(s.dir, 'dist');
	await mkdir(modelsDir, { recursive: true });
	await mkdir(path.join(webDist, 'assets'), { recursive: true });
	await writeFile(path.join(modelsDir, 'card-detector.onnx'), 'onnx-bytes');
	await writeFile(path.join(s.dir, 'cards.db'), 'secret-db-bytes');
	await writeFile(path.join(webDist, 'index.html'), '<!doctype html><title>shell</title>');
	await writeFile(path.join(webDist, 'assets', 'app-abc123.js'), 'console.log(1)');
	const app = createApp({
		cards: s.cards,
		images: s.images,
		identifier: fixedIdentifier(hits),
		search: new NameSearch(
			[
				meta(),
				meta({ scryfall_id: 'sf-chain', name: 'Chain Lightning', set_code: 'leg', collector_number: '1' }),
			],
			s.catalog,
		),
		catalog: s.catalog,
		modelsDir,
		webDist,
		log: null,
	});
	const image = (url: string, body: RequestInit['body'] = FAKE_JPEG, type = 'image/jpeg') =>
		app.request(url, { method: 'POST', body, headers: { 'content-type': type } });
	const identify = (body?: RequestInit['body'], type?: string) => image('/api/identify', body, type);
	/** Add a scanned Lightning Bolt (`scryfall_id` `''` adds an unidentified card). */
	const add = (id: string, foil = false, scryfallId = 'sf-bolt', body?: RequestInit['body'], type?: string) =>
		image(`/api/cards?card_id=${id}&scryfall_id=${scryfallId}&foil=${foil ? 1 : 0}`, body, type);
	const json = (method: string, url: string, body: unknown) =>
		app.request(url, { method, body: JSON.stringify(body), headers: { 'content-type': 'application/json' } });
	return { app, image, identify, add, json, ...s };
}

test('health and empty library', async () => {
	const h = await harness();
	try {
		const health = (await (await h.app.request('/api/health')).json()) as HealthResponse;
		assert.deepEqual(health, { ok: true, cards_indexed: 1, library_size: 0 });
		const lib = (await (await h.app.request('/api/library')).json()) as LibraryResponse;
		assert.deepEqual(lib, { cards: [] });
	} finally {
		await h.cleanup();
	}
});

test('identify -> add -> library -> update -> delete round trip', async () => {
	const h = await harness();
	try {
		const idRes = await h.identify();
		assert.equal(idRes.status, 200);
		const found = (await idRes.json()) as IdentifyResponse;
		assert.equal(found.confidence, 'CONFIDENT');
		assert.equal(found.candidates[0]?.scryfall_id, 'sf-bolt');
		assert.equal(((await (await h.app.request('/api/health')).json()) as HealthResponse).library_size, 0);

		const res = await h.add('c1');
		assert.equal(res.status, 201);
		const added = (await res.json()) as AddCardResponse;
		assert.equal(added.merged, false);
		assert.equal(added.card.card_id, 'c1');
		assert.equal(added.card.name, 'Lightning Bolt');
		assert.equal(added.card.has_photo, true);

		const photo = await h.app.request('/scans/c1.jpg');
		assert.equal(photo.status, 200);
		assert.equal(photo.headers.get('cache-control'), 'private, max-age=0, must-revalidate');
		assert.deepEqual(new Uint8Array(await photo.arrayBuffer()), FAKE_JPEG);

		const lib = (await (await h.app.request('/api/library')).json()) as LibraryResponse;
		assert.equal(lib.cards.length, 1);

		const upd = await h.json('PUT', '/api/cards/c1', { count: 3, foil: true });
		assert.equal(upd.status, 200);
		const updated = (await upd.json()) as CardResponse;
		assert.equal(updated.card.count, 3);
		assert.equal(updated.card.foil, true);
		assert.equal(updated.card.name, 'Lightning Bolt');

		const got = (await (await h.app.request('/api/cards/c1')).json()) as CardResponse;
		assert.equal(got.card.count, 3);

		const del = await h.app.request('/api/cards/c1', { method: 'DELETE' });
		assert.equal(del.status, 200);
		assert.equal((await h.app.request('/scans/c1.jpg')).status, 404);
		assert.equal((await h.app.request('/api/cards/c1')).status, 404);
		assert.equal((await h.app.request('/api/cards/c1', { method: 'DELETE' })).status, 404);
	} finally {
		await h.cleanup();
	}
});

test('identify and add validation', async () => {
	const h = await harness();
	try {
		assert.equal((await h.identify(FAKE_JPEG, 'text/plain')).status, 400);
		assert.equal((await h.identify(new Uint8Array(0))).status, 400);
		assert.equal((await h.image('/api/cards?scryfall_id=sf-bolt&foil=0')).status, 400, 'card_id required');
		assert.equal((await h.add('../etc/passwd')).status, 400);
		assert.equal((await h.add('c1', false, 'sf-bolt', FAKE_JPEG, 'text/plain')).status, 400);
		assert.equal((await h.add('c1', false, 'sf-bolt', new Uint8Array(0))).status, 400);
		assert.equal((await h.add('c1', false, 'sf-unknown')).status, 400, 'printing must exist in the index');
		assert.equal((await h.add('c1')).status, 201);
		const dupId = await h.add('c1');
		assert.equal(dupId.status, 409);
		assert.equal(((await dupId.json()) as ErrorResponse).status, 409);

		const placeholder = (await (await h.add('u1', true, '')).json()) as AddCardResponse;
		assert.equal(placeholder.card.scryfall_id, null);
		assert.equal(placeholder.card.name, 'Unknown');
		assert.equal(placeholder.card.foil, true);
	} finally {
		await h.cleanup();
	}
});

test('update validation', async () => {
	const h = await harness();
	try {
		await h.add('c1');
		assert.equal((await h.json('PUT', '/api/cards/c1', { count: 0 })).status, 400);
		assert.equal((await h.json('PUT', '/api/cards/c1', { count: 1.5 })).status, 400);
		assert.equal((await h.json('PUT', '/api/cards/c1', { name: '' })).status, 400);
		assert.equal((await h.json('PUT', '/api/cards/c1', { foil: 'yes' })).status, 400);
		assert.equal((await h.json('PUT', '/api/cards/c1', { set_code: 5 })).status, 400);
		assert.equal(
			(
				await h.app.request('/api/cards/c1', {
					method: 'PUT',
					body: '{not json',
					headers: { 'content-type': 'application/json' },
				})
			).status,
			400,
		);
		assert.equal((await h.json('PUT', '/api/cards/c1', [1])).status, 400);
		assert.equal((await h.json('PUT', '/api/cards/missing', { count: 2 })).status, 404);
		// Empty strings clear nullable fields.
		const cleared = (await (
			await h.json('PUT', '/api/cards/c1', { collector_number: '' })
		).json()) as CardResponse;
		assert.equal(cleared.card.collector_number, null);
		// Printing attributes are attached from the catalog, never stored or accepted.
		assert.equal(cleared.card.artist, 'Christopher Rush');
		assert.equal(cleared.card.rarity, 'common');
		const unidentified = (await (
			await h.json('PUT', '/api/cards/c1', { scryfall_id: null })
		).json()) as CardResponse;
		assert.equal(unidentified.card.artist, null);
		assert.equal(unidentified.card.colors, null);
	} finally {
		await h.cleanup();
	}
});

test('a second scan of the same printing folds into the first; edits fold too', async () => {
	const h = await harness();
	try {
		await h.add('old');
		const res = await h.add('new');
		assert.equal(res.status, 201);
		const dup = (await res.json()) as AddCardResponse;
		assert.equal(dup.merged, true);
		assert.equal(dup.card.card_id, 'old');
		assert.equal(dup.card.count, 2);
		assert.equal((await h.app.request('/api/cards/new')).status, 404);
		assert.equal(await h.images.exists('new'), false);
		assert.equal(((await (await h.app.request('/api/library')).json()) as LibraryResponse).cards.length, 1);

		// A foil copy is its own card until it is flipped to non-foil, then it folds.
		await h.add('shiny', true);
		const folded = (await (await h.json('PUT', '/api/cards/shiny', { foil: false })).json()) as CardResponse;
		assert.equal(folded.card.card_id, 'old');
		assert.equal(folded.card.count, 3);
		assert.equal((await h.app.request('/api/cards/shiny')).status, 404);
		assert.equal(await h.images.exists('shiny'), false, 'the folded row’s photo is gone; old keeps its own');
	} finally {
		await h.cleanup();
	}
});

test('search endpoint', async () => {
	const h = await harness();
	try {
		assert.equal((await h.app.request('/api/search')).status, 400);
		assert.equal((await h.app.request('/api/search?q=l')).status, 400);
		const res = (await (await h.app.request('/api/search?q=lightning')).json()) as SearchResponse;
		assert.deepEqual(
			res.cards.map((c) => c.name),
			['Lightning Bolt', 'Chain Lightning'],
		);
		const byNumber = (await (await h.app.request('/api/search?q=M11%20146')).json()) as SearchResponse;
		assert.equal(byNumber.cards[0]?.scryfall_id, 'sf-bolt');
	} finally {
		await h.cleanup();
	}
});

test('export returns a CSV attachment, in the app layout or Moxfield layout', async () => {
	const h = await harness();
	try {
		await h.add('c1');
		const res = await h.app.request('/api/export');
		assert.equal(res.status, 200);
		assert.match(res.headers.get('content-type') ?? '', /text\/csv/);
		assert.match(
			res.headers.get('content-disposition') ?? '',
			/attachment; filename="mtg-library-\d{4}-\d{2}-\d{2}\.csv"/,
		);
		const text = await res.text();
		assert.ok(text.startsWith('name,set_code,set_name,'));
		assert.ok(text.includes('Lightning Bolt,m11,Magic 2011,146,common,Christopher Rush,R,1,false,1,sf-bolt'));

		const mox = await h.app.request('/api/export?format=moxfield');
		assert.equal(mox.status, 200);
		assert.match(mox.headers.get('content-disposition') ?? '', /filename="moxfield-\d{4}-\d{2}-\d{2}\.csv"/);
		const lines = (await mox.text()).trimEnd().split('\n');
		assert.ok(lines[0]?.startsWith('"Count","Tradelist Count","Name","Edition"'));
		assert.ok(lines[1]?.startsWith('"1","1","Lightning Bolt","m11","Near Mint","English","",""'));
		assert.equal((await h.app.request('/api/export?format=xlsx')).status, 400);
	} finally {
		await h.cleanup();
	}
});

test('import creates cards from a Moxfield CSV and reports what happened', async () => {
	const h = await harness();
	try {
		const csv =
			'"Count","Tradelist Count","Name","Edition","Condition","Language","Foil","Tags","Last Modified","Collector Number","Alter","Proxy","Purchase Price"\r\n' +
			'"2","2","Lightning Bolt","m11","Near Mint","English","foil","","2026-09-01 00:24:01.653000","146","False","False",""\r\n' +
			'"1","1","Nope","zzz","Near Mint","English","","","2026-09-01 00:24:01.653000","1","False","False",""\r\n';
		const post = (body: string, query = '') =>
			h.app.request(`/api/import${query}`, { method: 'POST', body, headers: { 'content-type': 'text/csv' } });

		const res = await post(csv);
		assert.equal(res.status, 200);
		const result = (await res.json()) as ImportResponse;
		assert.deepEqual(result, { rows: 2, added: 2, updated: 0, unmatched: 1, unmatched_names: ['Nope'] });

		const lib = (await (await h.app.request('/api/library')).json()) as LibraryResponse;
		assert.equal(lib.cards.length, 2);
		const bolt = lib.cards.find((c) => c.scryfall_id === 'sf-bolt');
		assert.equal(bolt?.count, 2);
		assert.equal(bolt?.foil, true);
		assert.equal(bolt?.has_photo, false);
		assert.equal(bolt?.artist, 'Christopher Rush'); // enriched like any other card
		assert.equal(lib.cards.find((c) => c.scryfall_id === null)?.name, 'Nope');

		// Re-import in set mode changes nothing; add mode stacks.
		assert.deepEqual(((await (await post(csv)).json()) as ImportResponse).added, 0);
		await post(csv, '?mode=add');
		const after = (await (await h.app.request('/api/library')).json()) as LibraryResponse;
		assert.equal(after.cards.find((c) => c.scryfall_id === 'sf-bolt')?.count, 4);

		// A scan of the imported foil folds into it and gives it the photo.
		const scan = (await (await h.add('s1', true)).json()) as AddCardResponse;
		assert.equal(scan.merged, true);
		assert.equal(scan.card.count, 5);
		assert.equal(scan.card.has_photo, true);

		assert.equal((await post(csv, '?mode=merge')).status, 400);
		assert.equal((await post('   ')).status, 400);
		assert.equal((await post('a,b\n1,2\n')).status, 400);
		assert.equal((await post(csv.split('\r\n')[0] ?? '')).status, 400); // header only
	} finally {
		await h.cleanup();
	}
});

test('has_photo follows the scan photo', async () => {
	const h = await harness();
	try {
		const added = (await (await h.add('c1')).json()) as AddCardResponse;
		assert.equal(added.card.has_photo, true);
		const got = (await (await h.app.request('/api/cards/c1')).json()) as CardResponse;
		assert.equal(got.card.has_photo, true);
		assert.equal(h.images.has('c1'), true);
		await h.images.remove('c1');
		assert.equal(h.images.has('c1'), false);
		const later = (await (await h.app.request('/api/cards/c1')).json()) as CardResponse;
		assert.equal(later.card.has_photo, false);
	} finally {
		await h.cleanup();
	}
});

test('static routes: models, SPA shell, hashed assets, JSON 404 for API', async () => {
	const h = await harness();
	try {
		const model = await h.app.request('/models/card-detector.onnx');
		assert.equal(model.status, 200);
		assert.equal(await model.text(), 'onnx-bytes');
		assert.equal((await h.app.request('/models/nope.onnx')).status, 404);
		// Traversal: the URL parser collapses a literal `..` before routing and
		// the encoded form is rejected by the static handler. Neither may leak.
		for (const url of ['/models/../cards.db', '/models/%2e%2e/cards.db', '/scans/%2e%2e/cards.db']) {
			const res = await h.app.request(url);
			assert.ok(!(await res.text()).includes('secret-db-bytes'), `${url} leaked the database`);
		}

		const shell = await h.app.request('/');
		assert.equal(shell.status, 200);
		assert.match(await shell.text(), /shell/);
		assert.equal(shell.headers.get('cache-control'), 'no-cache');
		const deep = await h.app.request('/some/client/route');
		assert.equal(deep.status, 200);
		assert.equal((await h.app.request('/assets/app-stale00.js')).status, 404);
		assert.equal((await h.app.request('/favicon.ico')).status, 404);

		const asset = await h.app.request('/assets/app-abc123.js');
		assert.equal(asset.status, 200);
		assert.equal(asset.headers.get('cache-control'), 'public, max-age=31536000, immutable');

		const missingApi = await h.app.request('/api/nope');
		assert.equal(missingApi.status, 404);
		assert.equal(((await missingApi.json()) as ErrorResponse).status, 404);
		assert.equal((await h.app.request('/scans/nope.jpg')).status, 404);
	} finally {
		await h.cleanup();
	}
});
