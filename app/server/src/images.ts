/**
 * Scan photos on the local filesystem: `<scansDir>/<card_id>.jpg`.
 *
 * The set of ids with a photo is read once at startup and kept current by
 * the write/copy/remove methods, so `has()` is a synchronous lookup that
 * the library listing can afford per card.
 */
import { mkdirSync, readdirSync } from 'node:fs';
import { copyFile, rm, stat, writeFile } from 'node:fs/promises';
import path from 'node:path';

const EXT = '.jpg';

export class ImageStore {
	readonly dir: string;
	readonly #known = new Set<string>();

	constructor(dir: string) {
		this.dir = dir;
		mkdirSync(dir, { recursive: true });
		for (const name of readdirSync(dir)) {
			if (name.endsWith(EXT)) this.#known.add(name.slice(0, -EXT.length));
		}
	}

	pathFor(cardId: string): string {
		return path.join(this.dir, `${cardId}${EXT}`);
	}

	/** Whether a photo is on record for the card (no filesystem access). */
	has(cardId: string): boolean {
		return this.#known.has(cardId);
	}

	async write(cardId: string, data: Uint8Array): Promise<void> {
		await writeFile(this.pathFor(cardId), data);
		this.#known.add(cardId);
	}

	/** Returns `true` if a file was removed. Missing files are not an error. */
	async remove(cardId: string): Promise<boolean> {
		this.#known.delete(cardId);
		if (!(await this.exists(cardId))) return false;
		await rm(this.pathFor(cardId), { force: true });
		return true;
	}

	async copy(fromId: string, toId: string): Promise<void> {
		await copyFile(this.pathFor(fromId), this.pathFor(toId));
		this.#known.add(toId);
	}

	async exists(cardId: string): Promise<boolean> {
		try {
			return (await stat(this.pathFor(cardId))).isFile();
		} catch {
			return false;
		}
	}
}
