/**
 * Scanning is two steps. `identifyScan` runs the photo through the index
 * and reports confidence + alternatives; nothing is stored. Once the user
 * has confirmed the printing and foil, `addScannedCard` writes the card and
 * its photo, folding into the card that already holds that printing + foil
 * (see `library.ts`).
 */
import type { AddCardResponse, Confidence, IdentifyResponse, ScanCandidate } from '../../shared/api.ts';
import { badRequest, conflict } from './errors.ts';
import type { CardIdentifier } from './identify.ts';
import type { Library } from './library.ts';
import type { CardCatalog } from './metadata.ts';

/** Cosine similarity at or above which the top hit is accepted outright. */
export const CONFIDENT_THRESHOLD = 0.7;
/** Below this the scan is treated as unrecognised. */
export const AMBIGUOUS_THRESHOLD = 0.4;
/**
 * If the best hit scores below this, the image is also embedded rotated
 * 180° and the better orientation wins. Cards photographed upside down
 * still produce plausible-looking (but wrong) matches around 0.75-0.8, so
 * the CONFIDENT threshold alone is not a safe trigger. Correct matches on
 * real phone photos land in the 0.85-0.92 band (Phase 4 eval), so this only
 * costs a second forward pass on the weaker half of scans.
 */
export const ROTATION_RETRY_THRESHOLD = 0.9;
/** Alternatives returned with every scan. */
export const TOP_K = 5;

export const PLACEHOLDER_NAME = 'Unknown';

export function classifyConfidence(similarity: number): Confidence {
	if (similarity >= CONFIDENT_THRESHOLD) return 'CONFIDENT';
	if (similarity >= AMBIGUOUS_THRESHOLD) return 'AMBIGUOUS';
	return 'NO_MATCH';
}

export async function identifyScan(identifier: CardIdentifier, image: Uint8Array): Promise<IdentifyResponse> {
	const hits = await identifier.identify(image, TOP_K);
	const candidates: ScanCandidate[] = hits.map((h) => ({
		scryfall_id: h.card.scryfall_id,
		name: h.card.name,
		set_code: h.card.set_code,
		collector_number: h.card.collector_number,
		artist: h.card.artist,
		similarity: h.similarity,
	}));
	const similarity = candidates[0]?.similarity ?? 0;
	return { confidence: candidates[0] ? classifyConfidence(similarity) : 'NO_MATCH', similarity, candidates };
}

export interface AddScanDeps {
	library: Library;
	catalog: CardCatalog;
}

export interface AddScanInput {
	cardId: string;
	/** `null` adds an unidentified placeholder. */
	scryfallId: string | null;
	foil: boolean;
	/** JPEG or PNG bytes of the rectified card. */
	image: Uint8Array;
}

export async function addScannedCard(
	{ library, catalog }: AddScanDeps,
	input: AddScanInput,
): Promise<AddCardResponse> {
	if (library.get(input.cardId)) throw conflict(`Card ${input.cardId} already exists`);
	const printing = input.scryfallId ? catalog.get(input.scryfallId) : undefined;
	if (input.scryfallId && !printing) throw badRequest(`Unknown printing ${input.scryfallId}`);

	const { card, merged } = await library.addScan(
		{
			card_id: input.cardId,
			scryfall_id: printing?.scryfall_id ?? null,
			name: printing?.name ?? PLACEHOLDER_NAME,
			set_code: printing?.set_code ?? null,
			collector_number: printing?.collector_number ?? null,
			foil: input.foil,
		},
		input.image,
	);
	return {
		card: { ...card, ...catalog.attributes(card.scryfall_id), has_photo: library.hasPhoto(card.card_id) },
		merged,
	};
}
