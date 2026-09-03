import assert from 'node:assert/strict';
import { test } from 'node:test';
import {
	boxCorners,
	boxIoU,
	decodeOutput,
	nms,
	normalizePortrait,
	type RawDetection,
} from '../src/lib/yolo-decode.ts';

const close = (a: number, b: number, eps = 1e-6) => Math.abs(a - b) < eps;

test('decodes end-to-end [1, N, 7] rows and drops weak/invalid ones', () => {
	// rows: cx, cy, w, h, conf, cls, angle
	const rows = [
		[323, 322, 244, 340, 0.71, 0, -0.347],
		[322, 321, 244, 340, 0.42, 0, -0.349],
		[544, 94, 138, 186, 0.0, 0, -0.378],
		[700, 10, 20, 30, 0.9, 0, 0.1], // cx outside the 640 letterbox
		[10, 10, 0.5, 30, 0.9, 0, 0.1], // degenerate width
	].flat();
	const dets = decodeOutput(rows, [1, 5, 7], 640, 0.25);
	assert.equal(dets.length, 2);
	assert.deepEqual(dets[0], { cx: 323, cy: 322, w: 244, h: 340, angle: -0.347, conf: 0.71 });
});

test('decodes raw [1, 6, N] channel layout', () => {
	const N = 3;
	const ch = {
		cx: [100, 200, 300],
		cy: [100, 200, 300],
		w: [50, 60, 70],
		h: [80, 90, 100],
		conf: [0.9, 0.1, 0.5],
		angle: [0.1, 0.2, 0.3],
	};
	const data = [...ch.cx, ...ch.cy, ...ch.w, ...ch.h, ...ch.conf, ...ch.angle];
	const dets = decodeOutput(data, [1, 6, N], 640, 0.25);
	assert.deepEqual(
		dets.map((d) => d.cx),
		[100, 300],
	);
	assert.equal(dets[1]!.angle, 0.3);
});

test('rejects unknown layouts', () => {
	assert.throws(() => decodeOutput([], [1, 5, 5], 640, 0.25));
	assert.throws(() => decodeOutput([], [2, 6, 10], 640, 0.25));
});

test('nms keeps the strongest of overlapping boxes', () => {
	const a: RawDetection = { cx: 100, cy: 100, w: 50, h: 70, angle: 0, conf: 0.9 };
	const b: RawDetection = { cx: 102, cy: 101, w: 50, h: 70, angle: 0.02, conf: 0.5 };
	const c: RawDetection = { cx: 300, cy: 300, w: 50, h: 70, angle: 0, conf: 0.7 };
	assert.ok(boxIoU(a, b) > 0.8);
	assert.equal(boxIoU(a, c), 0);
	assert.deepEqual(
		nms([b, c, a], 0.5).map((d) => d.conf),
		[0.9, 0.7],
	);
});

test('normalizePortrait swaps landscape boxes and wraps the angle', () => {
	const landscape = normalizePortrait({ cx: 0, cy: 0, w: 340, h: 244, angle: 1.22, conf: 1 });
	assert.equal(landscape.w, 244);
	assert.equal(landscape.h, 340);
	// 1.22 + π/2 = 2.79 wraps to 2.79 - π = -0.35
	assert.ok(close(landscape.angle, 1.22 + Math.PI / 2 - Math.PI, 1e-9));

	const upright = normalizePortrait({ cx: 0, cy: 0, w: 244, h: 340, angle: -0.347, conf: 1 });
	assert.ok(close(upright.angle, -0.347));

	const flipped = normalizePortrait({ cx: 0, cy: 0, w: 244, h: 340, angle: Math.PI - 0.1, conf: 1 });
	assert.ok(close(flipped.angle, -0.1));
	assert.ok(
		close(normalizePortrait({ cx: 0, cy: 0, w: 1, h: 2, angle: -Math.PI / 2, conf: 1 }).angle, Math.PI / 2),
	);
});

test('boxCorners returns TL, TR, BR, BL in the card frame', () => {
	const axis = boxCorners({ cx: 100, cy: 200, w: 40, h: 60, angle: 0, conf: 1 });
	assert.deepEqual(axis, [
		[80, 170],
		[120, 170],
		[120, 230],
		[80, 230],
	]);

	// A counter-clockwise tilt (negative angle, y down) moves the top-left
	// corner left and down relative to the upright box; top-right is the
	// highest corner on screen but must NOT be reported as top-left.
	const tilted = boxCorners({ cx: 0, cy: 0, w: 244, h: 340, angle: -0.347, conf: 1 });
	const [tl, tr] = tilted;
	assert.ok(tl![0] < -122 && tl![1] > -170);
	assert.ok(tr![1] < tl![1], 'top-right sits higher than top-left for a CCW tilt');
});
