import { mkdtemp, rm } from 'node:fs/promises';
import os from 'node:os';
import path from 'node:path';
import { CardStore } from '../src/db.ts';
import type { CardIdentifier, ScoredCard } from '../src/identify.ts';
import { ImageStore } from '../src/images.ts';
import { Library } from '../src/library.ts';
import { CardCatalog, type CardMetadata } from '../src/metadata.ts';

export async function tempDir(): Promise<{ dir: string; cleanup: () => Promise<void> }> {
	const dir = await mkdtemp(path.join(os.tmpdir(), 'mtg-test-'));
	return { dir, cleanup: () => rm(dir, { recursive: true, force: true }) };
}

export function meta(overrides: Partial<CardMetadata> = {}): CardMetadata {
	return {
		scryfall_id: 'sf-bolt',
		name: 'Lightning Bolt',
		set_code: 'm11',
		collector_number: '146',
		artist: 'Christopher Rush',
		type_line: 'Instant',
		rarity: 'common',
		set_name: 'Magic 2011',
		colors: 'R',
		mana_value: 1,
		released_at: '2010-07-16',
		lang: 'en',
		filename: null,
		...overrides,
	};
}

/** Identifier that returns a fixed hit list regardless of the image. */
export function fixedIdentifier(hits: ScoredCard[]): CardIdentifier {
	return {
		indexed: hits.length,
		async identify(_image, k) {
			return hits.slice(0, k);
		},
	};
}

export interface Stores {
	cards: CardStore;
	images: ImageStore;
	library: Library;
	catalog: CardCatalog;
	dir: string;
	cleanup: () => Promise<void>;
}

/** In-memory SQLite, a temp scans dir, and a catalog of the given printings. */
export async function stores(printings: CardMetadata[] = [meta()]): Promise<Stores> {
	const { dir, cleanup } = await tempDir();
	const cards = CardStore.open(':memory:');
	const images = new ImageStore(path.join(dir, 'scans'));
	return {
		cards,
		images,
		library: new Library(cards, images),
		catalog: new CardCatalog(printings),
		dir,
		cleanup: async () => {
			cards.close();
			await cleanup();
		},
	};
}

/** Smallest valid JPEG-ish payload; the fake identifier never decodes it. */
export const FAKE_JPEG = new Uint8Array([0xff, 0xd8, 0xff, 0xe0, 0x4a, 0x46, 0x49, 0x46, 0xff, 0xd9]);
