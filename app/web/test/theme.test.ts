import assert from 'node:assert/strict';
import { test } from 'node:test';
import { isTheme, resolveTheme, THEME_OPTIONS } from '../src/lib/theme.ts';

test('resolveTheme follows the OS only for "system"', () => {
	assert.equal(resolveTheme('system', true), 'dark');
	assert.equal(resolveTheme('system', false), 'light');
	assert.equal(resolveTheme('light', true), 'light');
	assert.equal(resolveTheme('dark', false), 'dark');
});

test('isTheme accepts only the three values', () => {
	for (const o of THEME_OPTIONS) assert.ok(isTheme(o.value));
	assert.equal(isTheme('auto'), false);
	assert.equal(isTheme(null), false);
});
