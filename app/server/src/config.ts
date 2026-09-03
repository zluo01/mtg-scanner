import os from 'node:os';
import path from 'node:path';

export interface Config {
	/** Root of all persistent state. Everything below derives from it. */
	dataDir: string;
	dbPath: string;
	scansDir: string;
	modelsDir: string;
	indexPath: string;
	metadataPath: string;
	embedModelPath: string;
	detectorModelPath: string;
	/** Built frontend to serve (`app/dist`), or `null` for API-only mode. */
	webDist: string | null;
	host: string;
	port: number;
	/** Max concurrent SigLIP2 forward passes. */
	scanConcurrency: number;
	/** Intra-op threads for the ONNX session. */
	embedThreads: number;
}

/** Files that must be provisioned under `dataDir` before the server starts. */
export const REQUIRED_FILES = [
	'index/card_index.faiss',
	'index/card_metadata.parquet',
	'models/siglip2-base.onnx',
	'models/card-detector.onnx',
] as const;

function intEnv(value: string | undefined, fallback: number): number {
	const n = Number.parseInt(value ?? '', 10);
	return Number.isFinite(n) && n > 0 ? n : fallback;
}

export function loadConfig(env: NodeJS.ProcessEnv = process.env): Config {
	// Default: the repository's data/ directory (gitignored). In the container
	// the same relative path lands on the /data volume.
	const dataDir = path.resolve(env.DATA_DIR ?? path.join(import.meta.dirname, '../../../data'));
	const defaultDist = path.resolve(import.meta.dirname, '../../dist');
	const webDist = env.WEB_DIST === '' ? null : path.resolve(env.WEB_DIST ?? defaultDist);
	const cpuCount = os.availableParallelism();

	return {
		dataDir,
		dbPath: path.join(dataDir, 'cards.db'),
		scansDir: path.join(dataDir, 'scans'),
		modelsDir: path.join(dataDir, 'models'),
		indexPath: path.join(dataDir, 'index', 'card_index.faiss'),
		metadataPath: path.join(dataDir, 'index', 'card_metadata.parquet'),
		embedModelPath: path.join(dataDir, 'models', 'siglip2-base.onnx'),
		detectorModelPath: path.join(dataDir, 'models', 'card-detector.onnx'),
		webDist,
		host: env.HOST ?? '0.0.0.0',
		port: intEnv(env.PORT, 3000),
		scanConcurrency: intEnv(env.SCAN_CONCURRENCY, 2),
		embedThreads: intEnv(env.EMBED_THREADS, Math.max(1, Math.min(cpuCount, 8))),
	};
}
