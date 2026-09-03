/**
 * Card detection in the browser with onnxruntime-web (wasm backend).
 *
 * The model (`/models/card-detector.onnx`, served by the API server) is a
 * YOLO26n-OBB trained on synthetic card scenes. It takes a `[1, 3, 640, 640]`
 * float tensor in `[0, 1]` and returns oriented boxes; decoding lives in
 * `yolo-decode.ts`.
 */

import mjsUrl from 'onnxruntime-web/ort-wasm-simd-threaded.mjs?url';
import wasmUrl from 'onnxruntime-web/ort-wasm-simd-threaded.wasm?url';
import * as ort from 'onnxruntime-web/wasm';
import {
	boxCorners,
	type Corners,
	decodeOutput,
	nms,
	normalizePortrait,
	type RawDetection,
} from './yolo-decode.ts';

export const MODEL_URL = '/models/card-detector.onnx';
const IMGSZ = 640;

/** A detected card in source-image pixel coordinates. */
export interface OrientedBox {
	cx: number;
	cy: number;
	width: number;
	height: number;
	/** Radians, image space, in (-π/2, π/2]. */
	angle: number;
	confidence: number;
	/** TL, TR, BR, BL in source pixels; input for `rectifyCard`. */
	corners: Corners;
}

export interface DetectOptions {
	/** Minimum confidence to keep (default 0.35). */
	confThreshold?: number;
	/** IoU above which overlapping boxes are merged (default 0.5). */
	iouThreshold?: number;
	/** Cap on returned boxes (default 12: a binder page is 9). */
	maxDetections?: number;
}

export type ImageSource = HTMLImageElement | HTMLCanvasElement | HTMLVideoElement | ImageBitmap;

let sessionPromise: Promise<ort.InferenceSession> | null = null;

/** Load (once) and cache the detector session. */
export function loadDetector(): Promise<ort.InferenceSession> {
	if (!sessionPromise) {
		ort.env.wasm.wasmPaths = { wasm: wasmUrl, mjs: mjsUrl };
		sessionPromise = ort.InferenceSession.create(MODEL_URL, {
			executionProviders: ['wasm'],
			graphOptimizationLevel: 'all',
		}).catch((err) => {
			sessionPromise = null;
			throw err;
		});
	}
	return sessionPromise;
}

export function sourceSize(source: ImageSource): { width: number; height: number } {
	if (source instanceof HTMLVideoElement) return { width: source.videoWidth, height: source.videoHeight };
	if (source instanceof HTMLImageElement) return { width: source.naturalWidth, height: source.naturalHeight };
	return { width: source.width, height: source.height };
}

/** Scale-to-fit into a grey square and remember the transform. */
function letterbox(source: ImageSource) {
	const { width, height } = sourceSize(source);
	if (width === 0 || height === 0) throw new Error('Source image has no pixels yet');
	const scale = Math.min(IMGSZ / width, IMGSZ / height);
	const drawW = Math.round(width * scale);
	const drawH = Math.round(height * scale);
	const padX = Math.floor((IMGSZ - drawW) / 2);
	const padY = Math.floor((IMGSZ - drawH) / 2);
	const canvas = document.createElement('canvas');
	canvas.width = IMGSZ;
	canvas.height = IMGSZ;
	const ctx = canvas.getContext('2d', { willReadFrequently: true });
	if (!ctx) throw new Error('2D canvas unavailable');
	ctx.fillStyle = 'rgb(114,114,114)';
	ctx.fillRect(0, 0, IMGSZ, IMGSZ);
	ctx.drawImage(source, padX, padY, drawW, drawH);
	const { data } = ctx.getImageData(0, 0, IMGSZ, IMGSZ);
	const plane = IMGSZ * IMGSZ;
	const tensor = new Float32Array(3 * plane);
	for (let i = 0; i < plane; i++) {
		tensor[i] = data[i * 4]! / 255;
		tensor[i + plane] = data[i * 4 + 1]! / 255;
		tensor[i + 2 * plane] = data[i * 4 + 2]! / 255;
	}
	return { tensor, scale, padX, padY, width, height };
}

/** Run the detector and return oriented boxes in source pixel coordinates. */
export async function detectCards(source: ImageSource, options: DetectOptions = {}): Promise<OrientedBox[]> {
	const { confThreshold = 0.35, iouThreshold = 0.5, maxDetections = 12 } = options;
	const session = await loadDetector();
	const { tensor, scale, padX, padY, width, height } = letterbox(source);

	const input = new ort.Tensor('float32', tensor, [1, 3, IMGSZ, IMGSZ]);
	const results = await session.run({ [session.inputNames[0]!]: input });
	const output = results[session.outputNames[0]!];
	if (!output) throw new Error('Detector returned no output');

	const raw = decodeOutput(output.data as Float32Array, output.dims, IMGSZ, confThreshold);
	const kept = nms(raw, iouThreshold).slice(0, maxDetections);

	return kept.map((d) => {
		const inSource: RawDetection = {
			cx: (d.cx - padX) / scale,
			cy: (d.cy - padY) / scale,
			w: d.w / scale,
			h: d.h / scale,
			angle: d.angle,
			conf: d.conf,
		};
		const box = normalizePortrait(inSource);
		const corners = boxCorners(box).map(([x, y]) => [
			Math.min(width, Math.max(0, x)),
			Math.min(height, Math.max(0, y)),
		]) as Corners;
		return {
			cx: box.cx,
			cy: box.cy,
			width: box.w,
			height: box.h,
			angle: box.angle,
			confidence: box.conf,
			corners,
		};
	});
}
