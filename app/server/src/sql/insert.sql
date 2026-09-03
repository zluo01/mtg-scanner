-- A new row. Refused by the printing rule if the printing + foil is held.
INSERT INTO cards (card_id, scryfall_id, name, set_code, collector_number, foil, count, created_at, updated_at)
VALUES (:card_id, :scryfall_id, :name, :set_code, :collector_number, :foil, :count, :created_at, :updated_at);
