-- Identified cards with the same printing and foil status, oldest first.
SELECT * FROM cards
WHERE scryfall_id = :scryfall_id AND foil = :foil
ORDER BY created_at;
