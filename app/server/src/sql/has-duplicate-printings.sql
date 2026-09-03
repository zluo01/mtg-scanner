-- Rows that share a printing + foil (only possible before the rule was enforced).
SELECT COUNT(*) AS n
FROM (
	SELECT 1
	FROM cards
	WHERE scryfall_id IS NOT NULL
	GROUP BY scryfall_id, foil
	HAVING COUNT(*) > 1
);
