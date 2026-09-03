/**
 * Pure post-processing for the card detector's ONNX output. No DOM, no
 * onnxruntime — so it can be unit-tested under Node.
 *
 * Two export layouts are supported:
 *
 * - End-to-end (YOLO26 OBB, the trained model): `[1, N, 7]` rows of
 *   `[cx, cy, w, h, conf, class, angle]` already filtered by the one-to-one
 *   head. Near-duplicates can still appear, so a light NMS is applied.
 * - Raw (YOLO11 OBB): `[1, 6, N]` channels `[cx, cy, w, h, conf, angle]`
 *   for every anchor; NMS is required.
 *
 * Coordinates are letterbox pixels; the angle is radians in image space
 * (y down), so a positive angle is a clockwise tilt on screen.
 */

export interface RawDetection {
	cx: number;
	cy: number;
	w: number;
	h: number;
	/** Radians, image space. */
	angle: number;
	conf: number;
}

export type Corners = [[number, number], [number, number], [number, number], [number, number]];

/** Parse either output layout into detections above `confThreshold`. */
export function decodeOutput(
	data: ArrayLike<number>,
	dims: readonly number[],
	imgsz: number,
	confThreshold: number,
): RawDetection[] {
	if (dims.length !== 3 || dims[0] !== 1) {
		throw new Error(`Unexpected detector output dims [${dims.join(',')}]`);
	}
	const d1 = dims[1]!;
	const d2 = dims[2]!;

	let count: number;
	let read: (i: number, field: 'cx' | 'cy' | 'w' | 'h' | 'conf' | 'angle') => number;
	if (d2 === 7) {
		// [1, N, 7] end-to-end rows.
		count = d1;
		const col = { cx: 0, cy: 1, w: 2, h: 3, conf: 4, angle: 6 } as const;
		read = (i, f) => data[i * 7 + col[f]]!;
	} else if (d1 === 6) {
		// [1, 6, N] raw channels.
		count = d2;
		const ch = { cx: 0, cy: 1, w: 2, h: 3, conf: 4, angle: 5 } as const;
		read = (i, f) => data[ch[f] * count + i]!;
	} else {
		throw new Error(`Unsupported detector output layout [${dims.join(',')}]`);
	}

	const out: RawDetection[] = [];
	for (let i = 0; i < count; i++) {
		const conf = read(i, 'conf');
		if (!(conf >= confThreshold)) continue;
		const cx = read(i, 'cx');
		const cy = read(i, 'cy');
		const w = read(i, 'w');
		const h = read(i, 'h');
		const angle = read(i, 'angle');
		if (![cx, cy, w, h, angle].every(Number.isFinite)) continue;
		if (w <= 1 || h <= 1 || cx < 0 || cy < 0 || cx > imgsz || cy > imgsz) continue;
		out.push({ cx, cy, w, h, angle, conf });
	}
	return out;
}

/** Axis-aligned bounding box of a rotated box. */
function aabb(d: RawDetection): { x1: number; y1: number; x2: number; y2: number } {
	const cos = Math.abs(Math.cos(d.angle));
	const sin = Math.abs(Math.sin(d.angle));
	const hw = (d.w * cos + d.h * sin) / 2;
	const hh = (d.w * sin + d.h * cos) / 2;
	return { x1: d.cx - hw, y1: d.cy - hh, x2: d.cx + hw, y2: d.cy + hh };
}

/** IoU of the axis-aligned envelopes; adequate for single-class NMS. */
export function boxIoU(a: RawDetection, b: RawDetection): number {
	const A = aabb(a);
	const B = aabb(b);
	const iw = Math.max(0, Math.min(A.x2, B.x2) - Math.max(A.x1, B.x1));
	const ih = Math.max(0, Math.min(A.y2, B.y2) - Math.max(A.y1, B.y1));
	const inter = iw * ih;
	const union = (A.x2 - A.x1) * (A.y2 - A.y1) + (B.x2 - B.x1) * (B.y2 - B.y1) - inter;
	return union <= 0 ? 0 : inter / union;
}

/** Greedy NMS: keep the most confident box, drop overlaps above `iouThreshold`. */
export function nms(dets: RawDetection[], iouThreshold: number): RawDetection[] {
	const sorted = [...dets].sort((a, b) => b.conf - a.conf);
	const kept: RawDetection[] = [];
	for (const d of sorted) {
		if (kept.every((k) => boxIoU(d, k) <= iouThreshold)) kept.push(d);
	}
	return kept;
}

/**
 * Rewrite `(w, h, angle)` so the box is portrait (`h >= w`) with the angle
 * in `(-π/2, π/2]`. The detector's parametrisation is ambiguous
 * (`(w, h, θ)` ≡ `(h, w, θ ± π/2)` ≡ `(w, h, θ + π)`), and MTG cards are
 * portrait, so this pins the box's local frame to the card's own frame for
 * any tilt short of upside down.
 */
export function normalizePortrait(d: RawDetection): RawDetection {
	let { w, h, angle } = d;
	if (w > h) {
		[w, h] = [h, w];
		angle += Math.PI / 2;
	}
	// Wrap into (-π/2, π/2].
	angle = angle - Math.PI * Math.floor((angle + Math.PI / 2) / Math.PI);
	if (angle <= -Math.PI / 2) angle += Math.PI;
	return { ...d, w, h, angle };
}

/**
 * Corners of a (portrait-normalised) box in clockwise order starting at the
 * card's top-left: TL, TR, BR, BL.
 */
export function boxCorners(d: RawDetection): Corners {
	const cos = Math.cos(d.angle);
	const sin = Math.sin(d.angle);
	const hw = d.w / 2;
	const hh = d.h / 2;
	const local: [number, number][] = [
		[-hw, -hh],
		[hw, -hh],
		[hw, hh],
		[-hw, hh],
	];
	return local.map(([x, y]) => [d.cx + x * cos - y * sin, d.cy + x * sin + y * cos]) as Corners;
}
