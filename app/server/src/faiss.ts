/**
 * Reader and brute-force search for FAISS `IndexFlatIP` files.
 *
 * On-disk layout (little-endian), as written by `faiss.write_index`:
 *
 *   fourcc      4 bytes  "IxFI" (inner product)
 *   d           int32    vector dimensionality
 *   ntotal      int64    number of vectors
 *   dummy       int64    (1 << 20, unused)
 *   dummy       int64    (1 << 20, unused)
 *   is_trained  uint8
 *   metric_type int32    0 = inner product
 *   metric_arg  float32  only present when metric_type > 1
 *   nfloats     uint64   == d * ntotal
 *   data        float32[d * ntotal], row-major
 *
 * The vectors are L2-normalised SigLIP2 embeddings, so the inner product is
 * the cosine similarity.
 */
import { open } from 'node:fs/promises';

const FOURCC_INNER_PRODUCT = 'IxFI';
const METRIC_INNER_PRODUCT = 0;
/** Header size when metric_type <= 1 (no metric_arg). */
const HEADER_BYTES = 4 + 4 + 8 + 8 + 8 + 1 + 4 + 8;

export interface Hit {
	/** Row index into the companion metadata. */
	index: number;
	/** Inner product with the query. */
	score: number;
}

export class FlatIndex {
	readonly dim: number;
	readonly ntotal: number;
	/** Row-major `ntotal * dim` floats. */
	readonly vectors: Float32Array;

	constructor(dim: number, vectors: Float32Array) {
		if (dim <= 0 || vectors.length % dim !== 0) {
			throw new Error(`Vector buffer length ${vectors.length} is not a multiple of dim ${dim}`);
		}
		this.dim = dim;
		this.ntotal = vectors.length / dim;
		this.vectors = vectors;
	}

	/** Row `i` as a view (no copy). */
	row(i: number): Float32Array {
		return this.vectors.subarray(i * this.dim, (i + 1) * this.dim);
	}

	/** Top-`k` rows by inner product, best first. */
	search(query: Float32Array, k: number): Hit[] {
		if (query.length !== this.dim) {
			throw new Error(`Query dim ${query.length} != index dim ${this.dim}`);
		}
		const d = this.dim;
		const n = this.ntotal;
		const v = this.vectors;
		const limit = Math.min(k, n);
		if (limit <= 0) return [];

		// Small sorted top-k buffer; k is tiny compared to n.
		const topIdx = new Int32Array(limit);
		const topScore = new Float32Array(limit).fill(Number.NEGATIVE_INFINITY);
		let filled = 0;

		for (let i = 0; i < n; i++) {
			const base = i * d;
			let s0 = 0;
			let s1 = 0;
			let s2 = 0;
			let s3 = 0;
			let j = 0;
			for (; j + 3 < d; j += 4) {
				s0 += v[base + j]! * query[j]!;
				s1 += v[base + j + 1]! * query[j + 1]!;
				s2 += v[base + j + 2]! * query[j + 2]!;
				s3 += v[base + j + 3]! * query[j + 3]!;
			}
			for (; j < d; j++) s0 += v[base + j]! * query[j]!;
			const score = s0 + s1 + s2 + s3;

			if (filled === limit && score <= topScore[limit - 1]!) continue;
			// Insert into the sorted buffer.
			let pos = filled < limit ? filled : limit - 1;
			while (pos > 0 && topScore[pos - 1]! < score) {
				topScore[pos] = topScore[pos - 1]!;
				topIdx[pos] = topIdx[pos - 1]!;
				pos--;
			}
			topScore[pos] = score;
			topIdx[pos] = i;
			if (filled < limit) filled++;
		}

		const hits: Hit[] = [];
		for (let i = 0; i < filled; i++) hits.push({ index: topIdx[i]!, score: topScore[i]! });
		return hits;
	}

	/** Serialise in the FAISS on-disk format (used by tests and tooling). */
	toBuffer(): Buffer {
		const nfloats = this.vectors.length;
		const buf = Buffer.alloc(HEADER_BYTES + nfloats * 4);
		let off = 0;
		buf.write(FOURCC_INNER_PRODUCT, off, 'ascii');
		off += 4;
		buf.writeInt32LE(this.dim, off);
		off += 4;
		buf.writeBigInt64LE(BigInt(this.ntotal), off);
		off += 8;
		buf.writeBigInt64LE(1n << 20n, off);
		off += 8;
		buf.writeBigInt64LE(1n << 20n, off);
		off += 8;
		buf.writeUInt8(1, off);
		off += 1;
		buf.writeInt32LE(METRIC_INNER_PRODUCT, off);
		off += 4;
		buf.writeBigUInt64LE(BigInt(nfloats), off);
		off += 8;
		for (let i = 0; i < nfloats; i++) buf.writeFloatLE(this.vectors[i]!, off + i * 4);
		return buf;
	}
}

/** Parse the header and return `{ dim, ntotal, dataOffset }`. */
export function parseFlatIndexHeader(header: Buffer): { dim: number; ntotal: number; dataOffset: number } {
	if (header.length < HEADER_BYTES) throw new Error('FAISS file too small to contain a header');
	const fourcc = header.toString('ascii', 0, 4);
	if (fourcc !== FOURCC_INNER_PRODUCT) {
		throw new Error(
			`Unsupported FAISS index type "${fourcc}" (expected IndexFlatIP "${FOURCC_INNER_PRODUCT}")`,
		);
	}
	const dim = header.readInt32LE(4);
	const ntotal = Number(header.readBigInt64LE(8));
	const metricType = header.readInt32LE(33);
	if (metricType !== METRIC_INNER_PRODUCT) {
		throw new Error(`Unsupported FAISS metric ${metricType} (expected inner product)`);
	}
	const nfloats = Number(header.readBigUInt64LE(37));
	if (nfloats !== dim * ntotal) {
		throw new Error(`FAISS vector count ${nfloats} != dim ${dim} * ntotal ${ntotal}`);
	}
	return { dim, ntotal, dataOffset: HEADER_BYTES };
}

/** Load an `IndexFlatIP` file into memory. */
export async function readFlatIndex(file: string): Promise<FlatIndex> {
	const fh = await open(file, 'r');
	try {
		const header = Buffer.alloc(HEADER_BYTES);
		const { bytesRead } = await fh.read(header, 0, HEADER_BYTES, 0);
		const { dim, ntotal, dataOffset } = parseFlatIndexHeader(header.subarray(0, bytesRead));

		const byteLength = dim * ntotal * 4;
		const { size } = await fh.stat();
		if (size < dataOffset + byteLength) {
			throw new Error(`FAISS file truncated: expected ${dataOffset + byteLength} bytes, got ${size}`);
		}

		// Read straight into an aligned buffer so we can view it as float32
		// without a second copy.
		const bytes = new Uint8Array(byteLength);
		let filled = 0;
		while (filled < byteLength) {
			const { bytesRead: n } = await fh.read(bytes, filled, byteLength - filled, dataOffset + filled);
			if (n === 0) throw new Error('Unexpected EOF while reading FAISS vectors');
			filled += n;
		}
		return new FlatIndex(dim, new Float32Array(bytes.buffer));
	} finally {
		await fh.close();
	}
}
