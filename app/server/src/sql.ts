/**
 * Every SQL statement the store runs, one per file in ./sql, read once when
 * this module loads. A missing file fails the process at startup rather
 * than at the first request that needs it.
 */
import { readFileSync } from 'node:fs';
import path from 'node:path';

const NAMES = [
	'schema',
	'printing-rule',
	'printing-rule-enforced',
	'has-duplicate-printings',
	'insert',
	'upsert',
	'upsert-set',
	'get',
	'list',
	'count',
	'find-duplicates',
	'update',
	'delete',
	'merge',
] as const;

export type StatementName = (typeof NAMES)[number];

function load(): Record<StatementName, string> {
	const dir = path.join(import.meta.dirname, 'sql');
	const out = {} as Record<StatementName, string>;
	for (const name of NAMES) {
		try {
			out[name] = readFileSync(path.join(dir, `${name}.sql`), 'utf8');
		} catch (err) {
			throw new Error(`SQL statement "${name}" is missing from ${dir}: ${(err as Error).message}`);
		}
	}
	return out;
}

export const SQL = load();
