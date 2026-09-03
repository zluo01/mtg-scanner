import { badRequest } from './errors.ts';

/**
 * Card ids are client-generated (`crypto.randomUUID()`), but they are also
 * used as file names under the scans directory, so they must never contain
 * path separators or dots. Letters, digits, `-` and `_` only.
 */
export const CARD_ID_PATTERN = /^[A-Za-z0-9][A-Za-z0-9_-]{0,63}$/;

export function isCardId(value: unknown): value is string {
	return typeof value === 'string' && CARD_ID_PATTERN.test(value);
}

/** Returns the id unchanged or throws a 400. */
export function assertCardId(value: unknown, field = 'card_id'): string {
	if (!isCardId(value)) throw badRequest(`Invalid ${field}`);
	return value;
}
