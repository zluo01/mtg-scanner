-- Whether the unique printing + foil index exists yet.
SELECT 1 AS present
FROM sqlite_master
WHERE type = 'index' AND name = 'uq_cards_printing';
