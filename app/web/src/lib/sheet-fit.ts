/**
 * Where a bottom sheet should sit when the on-screen keyboard is up.
 *
 * The keyboard shrinks the visual viewport only; the layout viewport a
 * fixed sheet is anchored to stays full height, and iOS pans the page to
 * reveal the focused input, which pushes the top of the sheet off-screen.
 * So the sheet is anchored to the bottom of the visible area and capped to
 * its height instead.
 */
export interface Visible {
	height: number;
	/** Offset of the visible area from the top of the layout viewport. */
	offsetTop: number;
}

/** Sheet height as a share of the visible area, same as the CSS default (92dvh). */
export const SHEET_SHARE = 0.92;

export function sheetFit(layoutHeight: number, visible: Visible): { bottom: number; maxHeight: number } {
	const bottom = Math.max(0, Math.round(layoutHeight - (visible.offsetTop + visible.height)));
	return { bottom, maxHeight: Math.round(visible.height * SHEET_SHARE) };
}
