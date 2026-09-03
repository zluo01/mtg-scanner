/**
 * Perspective-rectify a detected card into a flat 488x680 image (the
 * proportions of an MTG card) for the embedding model.
 *
 * A 3x3 homography is solved by direct linear transform from the four
 * detected corners, then each destination pixel is back-projected and
 * bilinearly sampled. Roughly 150 lines instead of the 10 MB OpenCV.js
 * build, and well under 50 ms on a phone for a single card.
 */

export const CARD_WIDTH = 488;
export const CARD_HEIGHT = 680;

export type Matrix3 = [number, number, number, number, number, number, number, number, number];
export type Quad = [[number, number], [number, number], [number, number], [number, number]];

/** Gaussian elimination with partial pivoting for an n×n system. */
function solve(A: number[][], b: number[]): number[] {
	const n = b.length;
	const M = A.map((row, i) => [...row, b[i]!]);
	for (let col = 0; col < n; col++) {
		let pivot = col;
		for (let r = col + 1; r < n; r++) if (Math.abs(M[r]![col]!) > Math.abs(M[pivot]![col]!)) pivot = r;
		if (Math.abs(M[pivot]![col]!) < 1e-12) throw new Error('Degenerate corners; cannot rectify');
		if (pivot !== col) [M[col], M[pivot]] = [M[pivot]!, M[col]!];
		for (let r = col + 1; r < n; r++) {
			const f = M[r]![col]! / M[col]![col]!;
			for (let c = col; c <= n; c++) M[r]![c]! -= f * M[col]![c]!;
		}
	}
	const x = new Array<number>(n).fill(0);
	for (let r = n - 1; r >= 0; r--) {
		let s = M[r]![n]!;
		for (let c = r + 1; c < n; c++) s -= M[r]![c]! * x[c]!;
		x[r] = s / M[r]![r]!;
	}
	return x;
}

/** Homography H with `dst ~ H · src` for the four point pairs. */
export function computeHomography(src: Quad, dst: Quad): Matrix3 {
	const A: number[][] = [];
	const b: number[] = [];
	for (let i = 0; i < 4; i++) {
		const [x, y] = src[i]!;
		const [u, v] = dst[i]!;
		A.push([x, y, 1, 0, 0, 0, -u * x, -u * y]);
		b.push(u);
		A.push([0, 0, 0, x, y, 1, -v * x, -v * y]);
		b.push(v);
	}
	const h = solve(A, b);
	return [h[0]!, h[1]!, h[2]!, h[3]!, h[4]!, h[5]!, h[6]!, h[7]!, 1];
}

export function invert3(m: Matrix3): Matrix3 {
	const [a, b, c, d, e, f, g, h, i] = m;
	const A = e * i - f * h;
	const B = -(d * i - f * g);
	const C = d * h - e * g;
	const det = a * A + b * B + c * C;
	if (Math.abs(det) < 1e-12) throw new Error('Singular homography');
	const inv = 1 / det;
	return [
		A * inv,
		-(b * i - c * h) * inv,
		(b * f - c * e) * inv,
		B * inv,
		(a * i - c * g) * inv,
		-(a * f - c * d) * inv,
		C * inv,
		-(a * h - b * g) * inv,
		(a * e - b * d) * inv,
	];
}

export function applyHomography(m: Matrix3, x: number, y: number): [number, number] {
	const w = m[6] * x + m[7] * y + m[8];
	return [(m[0] * x + m[1] * y + m[2]) / w, (m[3] * x + m[4] * y + m[5]) / w];
}

type ImageSource = HTMLImageElement | HTMLCanvasElement | HTMLVideoElement | ImageBitmap;

function toImageData(source: ImageSource): ImageData {
	let w: number;
	let h: number;
	if (source instanceof HTMLVideoElement) {
		w = source.videoWidth;
		h = source.videoHeight;
	} else if (source instanceof HTMLImageElement) {
		w = source.naturalWidth;
		h = source.naturalHeight;
	} else {
		w = source.width;
		h = source.height;
	}
	const canvas = document.createElement('canvas');
	canvas.width = w;
	canvas.height = h;
	const ctx = canvas.getContext('2d', { willReadFrequently: true });
	if (!ctx) throw new Error('2D canvas unavailable');
	ctx.drawImage(source, 0, 0, w, h);
	return ctx.getImageData(0, 0, w, h);
}

/**
 * Warp the quadrilateral `corners` (TL, TR, BR, BL in source pixels) of
 * `source` onto a `width`×`height` canvas.
 */
export function rectifyCard(
	source: ImageSource,
	corners: Quad,
	width = CARD_WIDTH,
	height = CARD_HEIGHT,
): HTMLCanvasElement {
	const src = toImageData(source);
	const dst: Quad = [
		[0, 0],
		[width - 1, 0],
		[width - 1, height - 1],
		[0, height - 1],
	];
	const Hinv = invert3(computeHomography(corners, dst));

	const out = new ImageData(width, height);
	const sw = src.width;
	const sh = src.height;
	const sd = src.data;
	const od = out.data;
	for (let y = 0; y < height; y++) {
		for (let x = 0; x < width; x++) {
			const [sx, sy] = applyHomography(Hinv, x, y);
			const o = (y * width + x) * 4;
			if (sx < 0 || sy < 0 || sx >= sw - 1 || sy >= sh - 1) {
				od[o] = od[o + 1] = od[o + 2] = 0;
				od[o + 3] = 255;
				continue;
			}
			const x0 = Math.floor(sx);
			const y0 = Math.floor(sy);
			const fx = sx - x0;
			const fy = sy - y0;
			const i00 = (y0 * sw + x0) * 4;
			const i10 = i00 + 4;
			const i01 = i00 + sw * 4;
			const i11 = i01 + 4;
			for (let c = 0; c < 3; c++) {
				od[o + c] = Math.round(
					sd[i00 + c]! * (1 - fx) * (1 - fy) +
						sd[i10 + c]! * fx * (1 - fy) +
						sd[i01 + c]! * (1 - fx) * fy +
						sd[i11 + c]! * fx * fy,
				);
			}
			od[o + 3] = 255;
		}
	}

	const canvas = document.createElement('canvas');
	canvas.width = width;
	canvas.height = height;
	const ctx = canvas.getContext('2d');
	if (!ctx) throw new Error('2D canvas unavailable');
	ctx.putImageData(out, 0, 0);
	return canvas;
}

export function canvasToJpeg(canvas: HTMLCanvasElement, quality = 0.9): Promise<Blob> {
	return new Promise((resolve, reject) => {
		canvas.toBlob((b) => (b ? resolve(b) : reject(new Error('JPEG encoding failed'))), 'image/jpeg', quality);
	});
}
