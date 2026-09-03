-- The library's one rule, stated to the database: a printing + foil lives in
-- exactly one row. Placeholders (no printing) are exempt. upsert.sql folds
-- through this index. A database written before the rule existed is folded
-- once at startup (Library.dedupeAll) before this runs.
CREATE UNIQUE INDEX IF NOT EXISTS uq_cards_printing
	ON cards (scryfall_id, foil)
	WHERE scryfall_id IS NOT NULL;
