/**
 * Preprocessing parity check.
 *
 * The visual index was built by the PyTorch pipeline. This script embeds a
 * sample of the same source images with the server's ONNX + sharp pipeline
 * and compares each embedding against the vector stored in the index for
 * that row. It also measures top-1 self-retrieval, which is what actually
 * matters for scanning.
 *
 * Usage (from app/):
 *   pnpm parity -- [--images DIR] [--n 100] [--seed 1]
 *
 * Defaults: DATA_DIR from the environment (or ~/.config/mtg-scanner) and
 * images from ../training/_data/scryfall/images (relative to the repo root).
 */
import { existsSync } from 'node:fs';
import { readFile } from 'node:fs/promises';
import path from 'node:path';
import { parseArgs } from 'node:util';
import { loadConfig } from '../src/config.ts';
import { createEmbedder } from '../src/embedder.ts';
import { readFlatIndex } from '../src/faiss.ts';
import { loadCardMetadata } from '../src/metadata.ts';

const { values } = parseArgs({
	options: {
		images: {
			type: 'string',
			default: path.resolve(import.meta.dirname, '../../../training/_data/scryfall/images'),
		},
		n: { type: 'string', default: '100' },
		seed: { type: 'string', default: '1' },
	},
});

/** Deterministic PRNG so runs are comparable. */
function mulberry32(seed: number): () => number {
	let a = seed >>> 0;
	return () => {
		a = (a + 0x6d2b79f5) >>> 0;
		let t = a;
		t = Math.imul(t ^ (t >>> 15), t | 1);
		t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
		return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
	};
}

function cosine(a: Float32Array, b: Float32Array): number {
	let dot = 0;
	let na = 0;
	let nb = 0;
	for (let i = 0; i < a.length; i++) {
		dot += a[i]! * b[i]!;
		na += a[i]! * a[i]!;
		nb += b[i]! * b[i]!;
	}
	return dot / Math.sqrt(na * nb);
}

const config = loadConfig();
const sampleSize = Number.parseInt(values.n, 10);
const rand = mulberry32(Number.parseInt(values.seed, 10));

console.log(`Loading index + metadata from ${config.dataDir} ...`);
const [index, metadata, embedder] = await Promise.all([
	readFlatIndex(config.indexPath),
	loadCardMetadata(config.metadataPath),
	createEmbedder(config.embedModelPath, { threads: config.embedThreads }),
]);
if (index.ntotal !== metadata.length) throw new Error('index/metadata length mismatch');

// Sample rows whose source image is available locally.
const rows: number[] = [];
let attempts = 0;
while (rows.length < sampleSize && attempts < sampleSize * 50) {
	attempts++;
	const i = Math.floor(rand() * index.ntotal);
	const filename = metadata[i]!.filename;
	if (filename && existsSync(path.join(values.images, filename))) rows.push(i);
}
if (rows.length === 0) throw new Error(`No source images found under ${values.images}`);

let sumCos = 0;
let minCos = 1;
let top1 = 0;
let worst = { row: -1, cos: 1, name: '' };
const started = performance.now();
for (const i of rows) {
	const image = await readFile(path.join(values.images, metadata[i]!.filename!));
	const emb = await embedder.embed(image);
	const cos = cosine(emb, index.row(i));
	sumCos += cos;
	if (cos < minCos) minCos = cos;
	if (cos < worst.cos)
		worst = {
			row: i,
			cos,
			name: `${metadata[i]!.name} (${metadata[i]!.set_code} #${metadata[i]!.collector_number})`,
		};
	const best = index.search(emb, 1)[0];
	if (best?.index === i) top1++;
	else if (best) {
		const b = metadata[best.index]!;
		console.log(
			`  rank-1 miss: ${metadata[i]!.name} (${metadata[i]!.set_code} #${metadata[i]!.collector_number}) -> ${b.name} (${b.set_code} #${b.collector_number}) score ${best.score.toFixed(4)} vs self ${cos.toFixed(4)}`,
		);
	}
}
const perImage = (performance.now() - started) / rows.length;

console.log(`\nSamples:            ${rows.length}`);
console.log(`Mean cosine:        ${(sumCos / rows.length).toFixed(5)}`);
console.log(`Min cosine:         ${minCos.toFixed(5)}  ${worst.name}`);
console.log(`Top-1 self-match:   ${top1}/${rows.length} (${((100 * top1) / rows.length).toFixed(1)}%)`);
console.log(`Embed time/image:   ${perImage.toFixed(0)} ms (${config.embedThreads} threads)`);
// Rank-1 misses between printings that share identical art are expected;
// the cosine floor is the actual parity criterion.
process.exit(minCos > 0.99 ? 0 : 1);
