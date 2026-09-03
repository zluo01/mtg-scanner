-- The library. One row per card: what identifies it and what the user chose.
-- Everything else about a printing is attached at read time from the index
-- metadata (metadata.ts), never stored twice.
CREATE TABLE IF NOT EXISTS cards (
	card_id          TEXT PRIMARY KEY,
	scryfall_id      TEXT,
	name             TEXT NOT NULL,
	set_code         TEXT,
	collector_number TEXT,
	foil             INTEGER NOT NULL DEFAULT 0,
	count            INTEGER NOT NULL DEFAULT 1,
	created_at       TEXT NOT NULL,
	updated_at       TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_cards_scryfall_id ON cards (scryfall_id);
