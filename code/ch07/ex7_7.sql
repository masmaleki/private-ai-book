-- Exercise 7-7: 30-day window with min price and distinct supplier count.
WITH offers_eur AS (
    SELECT o.*,
           CASE WHEN o.currency = 'USD' THEN o.unit_price / fx.usd_per_eur
                ELSE o.unit_price END AS price_eur
    FROM demo_offers AS o
    ASOF LEFT JOIN fx_rates AS fx ON o.observed_at >= fx.day
)
SELECT q.quote_line_id,
       count(o.offer_id)              AS n30,
       count(DISTINCT o.supplier_id)  AS n_suppliers30,
       round(min(o.price_eur), 4)     AS min30_eur,
       round(median(o.price_eur), 4)  AS med30_eur
FROM demo_quote_lines AS q
LEFT JOIN offers_eur AS o
       ON o.part_id = q.part_id
      AND o.observed_at >= q.issued_at - INTERVAL 30 DAY
      AND o.observed_at <  q.issued_at
GROUP BY q.quote_line_id
ORDER BY q.quote_line_id
LIMIT 5;
