import assert from 'node:assert/strict';
import { test } from 'node:test';
import { applyHomography, computeHomography, invert3, type Quad } from '../src/lib/rectify.ts';

const close = (a: number, b: number, eps = 1e-6) => Math.abs(a - b) < eps;

test('homography maps each source corner onto its destination corner', () => {
	const src: Quad = [
		[120, 80],
		[610, 110],
		[590, 790],
		[100, 760],
	];
	const dst: Quad = [
		[0, 0],
		[487, 0],
		[487, 679],
		[0, 679],
	];
	const H = computeHomography(src, dst);
	for (let i = 0; i < 4; i++) {
		const [u, v] = applyHomography(H, src[i]![0], src[i]![1]);
		assert.ok(close(u, dst[i]![0], 1e-6) && close(v, dst[i]![1], 1e-6), `corner ${i}: ${u},${v}`);
	}
	const Hinv = invert3(H);
	const [x, y] = applyHomography(Hinv, 100, 200);
	const [u, v] = applyHomography(H, x, y);
	assert.ok(close(u, 100) && close(v, 200));
});

test('degenerate corners are rejected', () => {
	const line: Quad = [
		[0, 0],
		[1, 1],
		[2, 2],
		[3, 3],
	];
	assert.throws(() => computeHomography(line, line));
});
