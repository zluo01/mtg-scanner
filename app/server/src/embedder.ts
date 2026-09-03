/**
 * SigLIP2 image embedding via ONNX Runtime.
 *
 * Preprocessing mirrors the Python training pipeline
 * (`training/models/card_embedding_model.py`): RGB, resize to 384x384 with a
 * bicubic filter, scale to [0, 1], then normalise with mean 0.5 / std 0.5,
 * which is the same as `pixel / 127.5 - 1`. The exported graph
 * (`training/scripts/export_siglip2_onnx.py`) L2-normalises its output.
 */
import { InferenceSession, Tensor } from 'onnxruntime-node';
import sharp from 'sharp';

export const INPUT_SIZE = 384;
export const EMBEDDING_DIM = 768;
const INPUT_NAME = 'pixel_values';
const OUTPUT_NAME = 'output_embedding';

export interface EmbedOptions {
	/** Rotate the image 180 degrees before embedding. */
	rotate180?: boolean;
}

export interface Embedder {
	readonly dim: number;
	embed(image: Uint8Array, options?: EmbedOptions): Promise<Float32Array>;
}

/** Decode + resize + normalise into a `[1, 3, 384, 384]` CHW float tensor. */
export async function preprocess(image: Uint8Array, options: EmbedOptions = {}): Promise<Float32Array> {
	let pipeline = sharp(image, { failOn: 'error' });
	if (options.rotate180) pipeline = pipeline.rotate(180);
	const { data, info } = await pipeline
		.resize(INPUT_SIZE, INPUT_SIZE, { fit: 'fill', kernel: 'cubic' })
		.removeAlpha()
		.toColourspace('srgb')
		.raw()
		.toBuffer({ resolveWithObject: true });
	if (info.channels !== 3 || info.width !== INPUT_SIZE || info.height !== INPUT_SIZE) {
		throw new Error(`Unexpected preprocessed shape ${info.width}x${info.height}x${info.channels}`);
	}
	const plane = INPUT_SIZE * INPUT_SIZE;
	const out = new Float32Array(3 * plane);
	for (let i = 0; i < plane; i++) {
		const p = i * 3;
		out[i] = data[p]! / 127.5 - 1;
		out[i + plane] = data[p + 1]! / 127.5 - 1;
		out[i + 2 * plane] = data[p + 2]! / 127.5 - 1;
	}
	return out;
}

export function l2Normalize(v: Float32Array): Float32Array {
	let sum = 0;
	for (let i = 0; i < v.length; i++) sum += v[i]! * v[i]!;
	const norm = Math.sqrt(sum);
	if (norm > 0 && Math.abs(norm - 1) > 1e-4) {
		for (let i = 0; i < v.length; i++) v[i] = v[i]! / norm;
	}
	return v;
}

export async function createEmbedder(
	modelPath: string,
	options: { threads?: number } = {},
): Promise<Embedder> {
	const session = await InferenceSession.create(modelPath, {
		executionProviders: ['cpu'],
		graphOptimizationLevel: 'all',
		intraOpNumThreads: options.threads,
	});
	if (!session.inputNames.includes(INPUT_NAME) || !session.outputNames.includes(OUTPUT_NAME)) {
		throw new Error(
			`Unexpected ONNX graph: inputs=${session.inputNames.join(',')} outputs=${session.outputNames.join(',')}`,
		);
	}

	return {
		dim: EMBEDDING_DIM,
		async embed(image, embedOptions) {
			const chw = await preprocess(image, embedOptions);
			const input = new Tensor('float32', chw, [1, 3, INPUT_SIZE, INPUT_SIZE]);
			const outputs = await session.run({ [INPUT_NAME]: input });
			const out = outputs[OUTPUT_NAME];
			if (!out) throw new Error(`Model returned no ${OUTPUT_NAME}`);
			const data = out.data as Float32Array;
			if (data.length !== EMBEDDING_DIM) {
				throw new Error(`Expected ${EMBEDDING_DIM}-dim embedding, got ${data.length}`);
			}
			return l2Normalize(Float32Array.from(data));
		},
	};
}
