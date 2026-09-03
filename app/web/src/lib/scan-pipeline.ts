/**
 * Camera frame or uploaded photo -> detected, rectified, JPEG-encoded cards
 * ready for `POST /api/scan`.
 */
import { newCardId } from './ids.ts';
import { canvasToJpeg, rectifyCard } from './rectify.ts';
import { type DetectOptions, detectCards, type ImageSource, type OrientedBox } from './yolo.ts';

/** Longest side a photo is reduced to before detection (plenty for a 488x680 crop). */
const MAX_PHOTO_SIDE = 2048;

export interface DetectedCard {
	/** Client-generated id; becomes the library `card_id`. */
	cardId: string;
	box: OrientedBox;
	/** Rectified 488x680 JPEG. */
	jpeg: Blob;
	/** Object URL of `jpeg` for previews. Release with `releaseDetections`. */
	previewUrl: string;
}

export async function detectAndRectify(
	source: ImageSource,
	options?: DetectOptions,
): Promise<DetectedCard[]> {
	const boxes = await detectCards(source, options);
	const cards: DetectedCard[] = [];
	for (const box of boxes) {
		const jpeg = await canvasToJpeg(rectifyCard(source, box.corners), 0.9);
		cards.push({ cardId: newCardId(), box, jpeg, previewUrl: URL.createObjectURL(jpeg) });
	}
	return cards;
}

export function releaseDetections(cards: DetectedCard[]): void {
	for (const c of cards) URL.revokeObjectURL(c.previewUrl);
}

/**
 * Decode a photo honouring EXIF orientation, downscaled so that phone
 * photos (12-48 MP) stay within canvas limits and rectify quickly.
 */
export async function fileToBitmap(file: Blob): Promise<ImageBitmap> {
	const full = await createImageBitmap(file, { imageOrientation: 'from-image' });
	const longest = Math.max(full.width, full.height);
	if (longest <= MAX_PHOTO_SIDE) return full;
	const scale = MAX_PHOTO_SIDE / longest;
	const small = await createImageBitmap(full, {
		resizeWidth: Math.round(full.width * scale),
		resizeHeight: Math.round(full.height * scale),
		resizeQuality: 'high',
	});
	full.close();
	return small;
}
