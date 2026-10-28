-- Point-in-time features for quote lines (illustrative SQL, synthetic demo_* tables).
-- Written for DuckDB; ClickHouse has ASOF JOIN and window functions with a
-- slightly different syntax (check your engine's documentation).

WITH offers_eur AS (
    -- 1) convert each offer with the FX rate valid on its own day (as-of join)
    SELECT o.*,
           CASE WHEN o.currency = 'USD' THEN o.unit_price / fx.usd_per_eur
                ELSE o.unit_price END AS price_eur
    FROM demo_offers AS o
    ASOF LEFT JOIN fx_rates AS fx
         ON o.observed_at >= fx.day
),
last_offer AS (
    -- 2) the last offer for the same part strictly BEFORE the quote
    SELECT q.quote_line_id, o.observed_at AS last_offer_at, o.price_eur AS last_eur
    FROM demo_quote_lines AS q
    ASOF LEFT JOIN offers_eur AS o
         ON q.part_id = o.part_id AND q.issued_at > o.observed_at
),
window30 AS (
    -- 3) median and count over the 30 days before the quote: [t - 30d, t)
    SELECT q.quote_line_id,
           median(o.price_eur) AS med30_eur,
           count(o.offer_id)   AS n30
    FROM demo_quote_lines AS q
    LEFT JOIN offers_eur AS o
           ON o.part_id = q.part_id
          AND o.observed_at >= q.issued_at - INTERVAL 30 DAY
          AND o.observed_at <  q.issued_at
    GROUP BY q.quote_line_id
)
SELECT q.quote_line_id, q.part_id, q.issued_at, q.qty,
       l.last_offer_at, round(l.last_eur, 4) AS last_eur,
       round(w.med30_eur, 4) AS med30_eur, w.n30,
       q.unit_price AS label
FROM demo_quote_lines AS q
LEFT JOIN last_offer AS l USING (quote_line_id)
LEFT JOIN window30   AS w USING (quote_line_id)
ORDER BY q.quote_line_id
LIMIT 5;
