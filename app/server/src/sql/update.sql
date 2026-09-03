-- Partial update: each column changes only when its set_* flag is 1, so an
-- omitted field is left alone and an explicit NULL still clears a nullable
-- column. Always bumps updated_at.
UPDATE cards SET
	scryfall_id      = CASE WHEN :set_scryfall_id      THEN :scryfall_id      ELSE scryfall_id      END,
	name             = CASE WHEN :set_name             THEN :name             ELSE name             END,
	set_code         = CASE WHEN :set_set_code         THEN :set_code         ELSE set_code         END,
	collector_number = CASE WHEN :set_collector_number THEN :collector_number ELSE collector_number END,
	foil             = CASE WHEN :set_foil             THEN :foil             ELSE foil             END,
	count            = CASE WHEN :set_count            THEN :count            ELSE count            END,
	created_at       = CASE WHEN :set_created_at       THEN :created_at       ELSE created_at       END,
	updated_at       = :updated_at
WHERE card_id = :card_id;
