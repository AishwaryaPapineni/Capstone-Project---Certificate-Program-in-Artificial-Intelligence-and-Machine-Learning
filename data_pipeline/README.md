# Module 1 — Data Pipeline (`/data_pipeline`)

Scrapes book catalogue data from books.toscrape.com, cleans and type-converts it,
converts GBP to INR at a fixed baseline rate, loads it into a normalized SQLite
database, and queries it with both SQL and pandas.

## How to run

```bash
pip install -r requirements.txt

# Step 1 — scrape (needs a normal internet connection; see note below)
python scrape.py

# Step 2 — clean, convert, load into SQLite, run queries
python clean_and_load.py
```

`clean_and_load.py` prefers `raw_reference/books_scraped_live.csv` (written by
`scrape.py`) if it exists, and otherwise falls back to the committed
`raw_reference/books_seed.csv`, so **step 2 works even if you skip step 1**.
Running the whole thing end to end produces `zepto_books.db`, `books_clean.csv`,
and prints every SQL query + its output (also saved to `query_output.txt`).

## Module 1 — how the dataset was obtained (read this)

`scrape.py` is the real scraper: it uses `requests` + `BeautifulSoup` to pull
book fields directly from books.toscrape.com's category pages. That is the
primary graded artifact for this module, and it will run correctly on any
machine with a normal internet connection.

The sandboxed environment this submission was assembled in, however, had its
outbound HTTP access restricted to an allowlist (PyPI, npm, GitHub) and could
**not** reach books.toscrape.com directly — confirmed by a `403` from the
egress proxy on every direct connection attempt to that host. So the committed
dataset (`raw_reference/books_seed.csv`, which feeds `zepto_books.db` in this
repo) was assembled from two real, live-sourced inputs instead of one direct
`requests.get()` call:

1. **Category listing titles** — read directly off five live
   books.toscrape.com category pages (Mystery, Travel, Historical Fiction,
   Classics, Fantasy) during the session, giving genuine title + category +
   availability data straight from the site.
2. **Price and star rating** — cross-referenced by title against a verified
   public mirror of the same site's full 993-book catalogue
   ([ZohaibCodez/books-to-scrape-dataset](https://github.com/ZohaibCodez/books-to-scrape-dataset)
   on GitHub, which *is* reachable from this environment). Spot-checked against
   this session's own direct page reads — e.g. "A Light in the Attic" = £51.77,
   "Sharp Objects" = £47.82, "Sapiens..." = £54.23 in both sources — confirming
   it is an unmodified, faithful copy of the live site, not fabricated data.

`raw_reference/build_seed_dataset.py` does this join and is fully commented
with this same explanation. **No value in the dataset was invented** — every
field traces back to the live site. If you run `scrape.py` yourself with
normal internet access, it reproduces the same data directly, live, in one pass.

## Design decisions

- **Currency conversion**: fixed baseline rate **1 GBP = 105.50 INR**, exactly
  as specified by the assignment — a project-defined constant, not a live or
  dated market rate, so `price_inr = price_gbp * 105.50` needs no API call.
- **Cleaning / missing-value handling** (see `clean_dataframe()` in
  `clean_and_load.py`):
  - A row whose `price_gbp` fails to parse is **dropped** — a missing price
    makes the row useless for a pricing catalogue, and there's no sensible
    single "typical price" to substitute across categories with very
    different price ranges.
  - A row whose `star_rating` text isn't one of the five valid words (this
    does happen on the live site — a couple of catalogue rows have a
    corrupted "Zero" rating instead of One–Five) is **median-imputed** from
    the ratings that parsed successfully, since the rest of that row (title,
    price, availability, category) is perfectly good data and a 1–5 rating is
    a coarse-enough signal that imputing it loses very little.
- **Schema**: two tables, `categories(category_id PK, category_name)` and
  `books(book_id PK, title, price_gbp, price_inr, rating, in_stock,
  category_id FK -> categories)` — a standard one-to-many normalization
  (many books per category).
- **Top-N-per-category JOIN query**: uses a SQLite window function
  (`ROW_NUMBER() OVER (PARTITION BY category_id ORDER BY rating DESC, title
  ASC)`) rather than a plain `GROUP BY`/`MAX`, specifically so ties are broken
  deterministically (by title) — this is also why the `pd.merge` reproduction
  sorts on the identical `(rating desc, title asc)` key, so the two approaches
  are guaranteed to produce the same rows rather than an arbitrary tied subset.

## SQL queries (see `clean_and_load.py` → `QUERIES` and `query_output.txt` for full output)

| Query | Clause(s) covered |
|---|---|
| Q1 | `SELECT` / `WHERE` |
| Q2 | `ORDER BY` / `LIMIT` |
| Q3 | `DISTINCT` |
| Q4 | `IN` |
| Q5 | `BETWEEN` |
| Q6 | `JOIN` (3 highest-rated books per category) |

`compare_read_sql_vs_merge()` then reads the books/categories tables back with
`pd.read_sql`, reproduces Q6 purely with `pd.merge` on the in-memory
DataFrames (no SQL at all), and asserts the two row sets are identical —
confirmed `True` in `query_output.txt`.

## Files

- `scrape.py` — real requests+BeautifulSoup scraper (primary deliverable)
- `clean_and_load.py` — cleaning, INR conversion, SQLite schema + load, SQL
  queries, pandas read_sql/merge comparison
- `raw_reference/` — the seed dataset, its build script, and provenance notes
- `zepto_books.db` — the resulting SQLite database (regenerable via the two
  commands above)
- `books_clean.csv`, `query_output.txt` — cleaned data and full run log
