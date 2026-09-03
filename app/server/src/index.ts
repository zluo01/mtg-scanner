/**
 * Entry point. Validates the data directory, loads the models and index
 * into memory once, then serves the API and the built frontend from a
 * single Node process.
 */
import { existsSync, mkdirSync } from 'node:fs';
import path from 'node:path';
import { serve } from '@hono/node-server';
import { createApp } from './app.ts';
import { loadConfig, REQUIRED_FILES } from './config.ts';
import { CardStore } from './db.ts';
import { createEmbedder } from './embedder.ts';
import { readFlatIndex } from './faiss.ts';
import { OnnxCardIdentifier } from './identify.ts';
import { ImageStore } from './images.ts';
import { Library } from './library.ts';
import { CardCatalog, loadCardMetadata } from './metadata.ts';
import { ROTATION_RETRY_THRESHOLD } from './scan.ts';
import { NameSearch } from './search.ts';

const log = (msg: string): void => console.log(`[${new Date().toISOString()}] ${msg}`);

async function timed<T>(label: string, work: () => Promise<T>): Promise<T> {
	const started = performance.now();
	const result = await work();
	log(`${label} (${((performance.now() - started) / 1000).toFixed(1)}s)`);
	return result;
}

async function main(): Promise<void> {
	const config = loadConfig();
	log(`Data directory: ${config.dataDir}`);

	const missing = REQUIRED_FILES.map((rel) => path.join(config.dataDir, rel)).filter((p) => !existsSync(p));
	if (missing.length > 0) {
		console.error('Missing required files:');
		for (const p of missing) console.error(`  - ${p}`);
		console.error(
			'\nProvision them from the training pipeline (see docs/phase5-interface.md), then start the server again.',
		);
		process.exit(2);
	}
	if (config.webDist && !existsSync(path.join(config.webDist, 'index.html'))) {
		console.error(
			`Frontend build not found at ${config.webDist}. Run "pnpm build" or set WEB_DIST= for API-only mode.`,
		);
		process.exit(2);
	}

	mkdirSync(config.scansDir, { recursive: true });
	const cards = CardStore.open(config.dbPath);
	const images = new ImageStore(config.scansDir);
	// One row per printing + foil; rows written before that rule fold now.
	const folded = await new Library(cards, images).dedupeAll();
	if (folded > 0) log(`Folded ${folded} duplicate rows into the cards that own their printing`);

	const [index, metadata, embedder] = await Promise.all([
		timed('Loaded visual index', () => readFlatIndex(config.indexPath)),
		timed('Loaded card metadata', () => loadCardMetadata(config.metadataPath)),
		timed('Loaded SigLIP2 model', () =>
			createEmbedder(config.embedModelPath, { threads: config.embedThreads }),
		),
	]);
	const identifier = new OnnxCardIdentifier(embedder, index, metadata, {
		concurrency: config.scanConcurrency,
		rotateRetryBelow: ROTATION_RETRY_THRESHOLD,
	});
	const catalog = new CardCatalog(metadata);
	const search = new NameSearch(metadata, catalog);
	log(`Ready: ${identifier.indexed} printings indexed, ${cards.count()} cards in library`);

	const app = createApp({
		cards,
		images,
		identifier,
		search,
		catalog,
		modelsDir: config.modelsDir,
		webDist: config.webDist,
	});
	const server = serve({ fetch: app.fetch, hostname: config.host, port: config.port }, (info) => {
		log(`Listening on http://${info.address}:${info.port}${config.webDist ? '' : ' (API only)'}`);
	});

	const shutdown = (signal: string) => {
		log(`${signal} received, shutting down`);
		server.close(() => {
			cards.close();
			process.exit(0);
		});
		setTimeout(() => process.exit(0), 3000).unref();
	};
	process.once('SIGINT', () => shutdown('SIGINT'));
	process.once('SIGTERM', () => shutdown('SIGTERM'));
}

main().catch((err) => {
	console.error('Fatal:', err);
	process.exit(1);
});
