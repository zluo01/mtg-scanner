import assert from 'node:assert/strict';
import { writeFile } from 'node:fs/promises';
import path from 'node:path';
import { test } from 'node:test';
import { FlatIndex, parseFlatIndexHeader, readFlatIndex } from '../src/faiss.ts';
import { tempDir } from './helpers.ts';

const unit = (...v: number[]) => {
	const n = Math.hypot(...v);
	return v.map((x) => x / n);
};

test('search returns best inner products first', () => {
	const rows = [unit(1, 0, 0, 0), unit(0, 1, 0, 0), unit(1, 1, 0, 0), unit(0, 0, 1, 0)];
	const index = new FlatIndex(4, Float32Array.from(rows.flat()));
	assert.equal(index.ntotal, 4);
	const hits = index.search(Float32Array.from(unit(1, 0.1, 0, 0)), 2);
	assert.deepEqual(
		hits.map((h) => h.index),
		[0, 2],
	);
	assert.ok(hits[0]!.score > hits[1]!.score);
	assert.ok(Math.abs(hits[0]!.score - unit(1, 0.1, 0, 0)[0]!) < 1e-6);
});

test('search handles k larger than ntotal and k = 0', () => {
	const index = new FlatIndex(2, Float32Array.from([1, 0, 0, 1]));
	assert.equal(index.search(Float32Array.from([1, 0]), 10).length, 2);
	assert.equal(index.search(Float32Array.from([1, 0]), 0).length, 0);
});

test('search rejects wrong query dim', () => {
	const index = new FlatIndex(2, Float32Array.from([1, 0]));
	assert.throws(() => index.search(Float32Array.from([1, 0, 0]), 1));
});

test('search works for dims that are not multiples of 4', () => {
	const index = new FlatIndex(5, Float32Array.from([...unit(1, 1, 1, 1, 1), ...unit(0, 0, 0, 0, 1)]));
	const hits = index.search(Float32Array.from(unit(0, 0, 0, 0, 1)), 1);
	assert.equal(hits[0]!.index, 1);
});

test('round-trips through the FAISS on-disk format', async () => {
	const { dir, cleanup } = await tempDir();
	try {
		const original = new FlatIndex(
			3,
			Float32Array.from([...unit(1, 2, 3), ...unit(-1, 0, 1), ...unit(0, 0, 1)]),
		);
		const file = path.join(dir, 'index.faiss');
		await writeFile(file, original.toBuffer());

		const loaded = await readFlatIndex(file);
		assert.equal(loaded.dim, 3);
		assert.equal(loaded.ntotal, 3);
		assert.deepEqual(Array.from(loaded.vectors), Array.from(original.vectors));
		assert.equal(loaded.search(Float32Array.from(unit(0, 0, 1)), 1)[0]!.index, 2);
	} finally {
		await cleanup();
	}
});

test('header parsing rejects other index types and corrupt counts', () => {
	const good = new FlatIndex(2, Float32Array.from([1, 0])).toBuffer();
	assert.deepEqual(parseFlatIndexHeader(good), { dim: 2, ntotal: 1, dataOffset: 45 });

	const l2 = Buffer.from(good);
	l2.write('IxF2', 0, 'ascii');
	assert.throws(() => parseFlatIndexHeader(l2), /IndexFlatIP/);

	const bad = Buffer.from(good);
	bad.writeBigUInt64LE(99n, 37);
	assert.throws(() => parseFlatIndexHeader(bad), /vector count/);

	assert.throws(() => parseFlatIndexHeader(Buffer.alloc(10)), /too small/);
});

test('truncated file is rejected', async () => {
	const { dir, cleanup } = await tempDir();
	try {
		const buf = new FlatIndex(4, Float32Array.from([1, 0, 0, 0, 0, 1, 0, 0])).toBuffer();
		const file = path.join(dir, 'short.faiss');
		await writeFile(file, buf.subarray(0, buf.length - 4));
		await assert.rejects(readFlatIndex(file), /truncated/);
	} finally {
		await cleanup();
	}
});
