import assert from 'node:assert/strict';
import { test } from 'node:test';
import type { Embedder } from '../src/embedder.ts';
import { FlatIndex } from '../src/faiss.ts';
import { OnnxCardIdentifier } from '../src/identify.ts';
import { meta } from './helpers.ts';

const unit = (...v: number[]) => {
	const n = Math.hypot(...v);
	return Float32Array.from(v.map((x) => x / n));
};

/** Upright image looks like row 1 weakly; rotated looks like row 0 strongly. */
function fakeEmbedder(calls: string[]): Embedder {
	return {
		dim: 2,
		async embed(_image, options) {
			calls.push(options?.rotate180 ? 'rotated' : 'upright');
			return options?.rotate180 ? unit(1, 0) : unit(0.6, 0.8);
		},
	};
}

const index = new FlatIndex(2, Float32Array.from([...unit(1, 0), ...unit(0, 1)]));
const metadata = [meta({ scryfall_id: 'row0', name: 'Row 0' }), meta({ scryfall_id: 'row1', name: 'Row 1' })];

test('retries rotated when the best hit is below the threshold and keeps the better orientation', async () => {
	const calls: string[] = [];
	const id = new OnnxCardIdentifier(fakeEmbedder(calls), index, metadata, { rotateRetryBelow: 0.9 });
	const hits = await id.identify(new Uint8Array(1), 2);
	assert.deepEqual(calls, ['upright', 'rotated']);
	assert.equal(hits[0]!.card.scryfall_id, 'row0');
	assert.ok(Math.abs(hits[0]!.similarity - 1) < 1e-6);
});

test('does not retry when the upright hit is strong enough', async () => {
	const calls: string[] = [];
	const id = new OnnxCardIdentifier(fakeEmbedder(calls), index, metadata, { rotateRetryBelow: 0.5 });
	const hits = await id.identify(new Uint8Array(1), 1);
	assert.deepEqual(calls, ['upright']);
	assert.equal(hits[0]!.card.scryfall_id, 'row1');
});

test('validates index/metadata alignment and dims', () => {
	const calls: string[] = [];
	assert.throws(() => new OnnxCardIdentifier(fakeEmbedder(calls), index, metadata.slice(0, 1)), /metadata/);
	assert.throws(() => new OnnxCardIdentifier({ ...fakeEmbedder(calls), dim: 3 }, index, metadata), /dim/);
});

test('bounds concurrency', async () => {
	let inFlight = 0;
	let peak = 0;
	const slow: Embedder = {
		dim: 2,
		async embed() {
			inFlight++;
			peak = Math.max(peak, inFlight);
			await new Promise((r) => setTimeout(r, 5));
			inFlight--;
			return unit(1, 0);
		},
	};
	const id = new OnnxCardIdentifier(slow, index, metadata, { concurrency: 2 });
	await Promise.all(Array.from({ length: 6 }, () => id.identify(new Uint8Array(1), 1)));
	assert.equal(peak, 2);
});
