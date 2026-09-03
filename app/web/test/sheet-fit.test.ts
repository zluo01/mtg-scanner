import assert from 'node:assert/strict';
import { test } from 'node:test';
import { sheetFit } from '../src/lib/sheet-fit.ts';

test('no keyboard: sheet on the bottom edge, 92% tall', () => {
	assert.deepEqual(sheetFit(932, { height: 932, offsetTop: 0 }), { bottom: 0, maxHeight: 857 });
});

test('keyboard up: anchored above it and capped to what is visible', () => {
	assert.deepEqual(sheetFit(932, { height: 560, offsetTop: 0 }), { bottom: 372, maxHeight: 515 });
});

test('page panned by the browser: the pan is taken off the bottom offset', () => {
	assert.deepEqual(sheetFit(932, { height: 560, offsetTop: 100 }), { bottom: 272, maxHeight: 515 });
	assert.equal(sheetFit(932, { height: 932, offsetTop: 50 }).bottom, 0, 'never below the edge');
});
