-- Import in "set" mode: insert, or make the held row's count the file's
-- count. The added date moves forward only when copies were gained, and a
-- row whose count did not change is not touched at all.
INSERT INTO cards (card_id, scryfall_id, name, set_code, collector_number, foil, count, created_at, updated_at)
VALUES (:card_id, :scryfall_id, :name, :set_code, :collector_number, :foil, :count, :created_at, :updated_at)
ON CONFLICT (scryfall_id, foil) WHERE scryfall_id IS NOT NULL DO UPDATE SET
	created_at = CASE WHEN excluded.count > count THEN MAX(created_at, excluded.created_at) ELSE created_at END,
	updated_at = CASE WHEN excluded.count = count THEN updated_at ELSE excluded.updated_at END,
	count      = excluded.count
RETURNING *;
