/**
 * Embed a card photo and look up its nearest printings in the visual index.
 */
import type { Embedder } from './embedder.ts';
import type { FlatIndex } from './faiss.ts';
import type { CardMetadata } from './metadata.ts';

export interface ScoredCard {
	card: CardMetadata;
	similarity: number;
}

export interface CardIdentifier {
	/** Number of printings in the index. */
	readonly indexed: number;
	/** Top-`k` printings for the image, best first. */
	identify(image: Uint8Array, k: number): Promise<ScoredCard[]>;
}

export interface IdentifierOptions {
	/** Max concurrent forward passes (bounds memory). */
	concurrency?: number;
	/**
	 * If the best hit scores below this, embed the image rotated 180° and
	 * keep whichever orientation scores higher. Cards photographed upside
	 * down are otherwise misidentified, and the retry only costs a second
	 * forward pass on weak matches.
	 */
	rotateRetryBelow?: number;
}

/** Minimal counting semaphore. */
class Semaphore {
	#free: number;
	#waiters: (() => void)[] = [];

	constructor(n: number) {
		this.#free = n;
	}

	async acquire(): Promise<() => void> {
		if (this.#free > 0) {
			this.#free--;
		} else {
			await new Promise<void>((resolve) => this.#waiters.push(resolve));
		}
		let released = false;
		return () => {
			if (released) return;
			released = true;
			const next = this.#waiters.shift();
			if (next) next();
			else this.#free++;
		};
	}
}

export class OnnxCardIdentifier implements CardIdentifier {
	readonly indexed: number;
	#embedder: Embedder;
	#index: FlatIndex;
	#metadata: CardMetadata[];
	#gate: Semaphore;
	#rotateRetryBelow: number;

	constructor(
		embedder: Embedder,
		index: FlatIndex,
		metadata: CardMetadata[],
		options: IdentifierOptions = {},
	) {
		if (index.ntotal !== metadata.length) {
			throw new Error(`Index has ${index.ntotal} vectors but metadata has ${metadata.length} rows`);
		}
		if (index.dim !== embedder.dim) {
			throw new Error(`Index dim ${index.dim} != embedder dim ${embedder.dim}`);
		}
		this.#embedder = embedder;
		this.#index = index;
		this.#metadata = metadata;
		this.indexed = index.ntotal;
		this.#gate = new Semaphore(Math.max(1, options.concurrency ?? 2));
		this.#rotateRetryBelow = options.rotateRetryBelow ?? Number.NEGATIVE_INFINITY;
	}

	async identify(image: Uint8Array, k: number): Promise<ScoredCard[]> {
		const release = await this.#gate.acquire();
		try {
			let hits = this.#index.search(await this.#embedder.embed(image), k);
			const best = hits[0]?.score ?? Number.NEGATIVE_INFINITY;
			if (best < this.#rotateRetryBelow) {
				const rotated = this.#index.search(await this.#embedder.embed(image, { rotate180: true }), k);
				if ((rotated[0]?.score ?? Number.NEGATIVE_INFINITY) > best) hits = rotated;
			}
			return hits.map((h) => ({ card: this.#metadata[h.index]!, similarity: h.score }));
		} finally {
			release();
		}
	}
}
