-- Fold a source row into the target: counts add and the target takes the
-- newer added date. The caller deletes the source afterwards (delete.sql).
UPDATE cards SET
	count      = count + :count,
	created_at = MAX(created_at, :created_at),
	updated_at = :updated_at
WHERE card_id = :card_id;
