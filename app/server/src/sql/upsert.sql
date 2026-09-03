-- Insert, or, when the printing + foil is already held, add the copies to
-- that row and give it the newer added date. One statement, so the outcome
-- is the same whatever else is writing. Returns the row that now holds the
-- copies; its card_id differs from the one sent when it folded.
INSERT INTO cards (card_id, scryfall_id, name, set_code, collector_number, foil, count, created_at, updated_at)
VALUES (:card_id, :scryfall_id, :name, :set_code, :collector_number, :foil, :count, :created_at, :updated_at)
ON CONFLICT (scryfall_id, foil) WHERE scryfall_id IS NOT NULL DO UPDATE SET
	count      = count + excluded.count,
	created_at = MAX(created_at, excluded.created_at),
	updated_at = excluded.updated_at
RETURNING *;
