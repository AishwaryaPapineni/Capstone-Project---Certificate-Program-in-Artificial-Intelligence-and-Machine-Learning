"""
clean_and_load.py — Module 1, Steps 2-5: clean, convert, load into SQLite, query.

Run:
    python clean_and_load.py

What it does, in order:
  1. Loads the raw scraped rows (prefers a live scrape from scrape.py if present,
     else falls back to the committed raw_reference/books_seed.csv — see that
     file's docstring for exactly how it was obtained).
  2. Cleans each field into a proper type, handling rows that fail to parse.
  3. Converts price_gbp -> price_inr using the fixed baseline rate 1 GBP = 105.50 INR.
  4. Builds a normalized two-table SQLite schema (categories, books) and loads the
     cleaned data into it.
  5. Runs >=5 required SQL queries against the database and prints their output.
  6. Reads two query results back with pd.read_sql, and reproduces the JOIN query
     with pd.merge on in-memory DataFrames, printing both side by side.

All printed output of this script is also captured to query_output.txt for the
repository (redirect stdout to a file, e.g. `python clean_and_load.py > query_output.txt`),
satisfying the "executed queries with output"
requirement in text form.
"""
import sqlite3
from pathlib import Path

import pandas as pd

HERE = Path(__file__).parent
LIVE_CSV = HERE / "raw_reference" / "books_scraped_live.csv"
SEED_CSV = HERE / "raw_reference" / "books_seed.csv"
DB_PATH = HERE / "zepto_books.db"

FIXED_GBP_TO_INR = 105.50  # project-defined constant baseline rate, no date reference

RATING_WORD_TO_INT = {"One": 1, "Two": 2, "Three": 3, "Four": 4, "Five": 5}


# ---------------------------------------------------------------------------
# Step 1: load raw rows
# ---------------------------------------------------------------------------
def load_raw() -> pd.DataFrame:
    src = LIVE_CSV if LIVE_CSV.exists() else SEED_CSV
    print(f"Loading raw scraped data from: {src.relative_to(HERE)}")
    df = pd.read_csv(src)
    print(f"Raw rows loaded: {len(df)}")
    return df


# ---------------------------------------------------------------------------
# Step 2: cleaning — price_gbp (float), rating (int 1-5), in_stock (bool)
# ---------------------------------------------------------------------------
def clean_price(raw: str):
    """'£47.82' -> 47.82. Returns None if the field can't be parsed."""
    try:
        return float(str(raw).replace("£", "").replace(",", "").strip())
    except (ValueError, TypeError):
        return None


def clean_rating(word: str):
    """'Four' -> 4. Returns None for anything not in the known word set
    (e.g. a corrupted/unexpected value like 'Zero' — this genuinely occurs a
    couple of times in the live books.toscrape.com catalogue)."""
    return RATING_WORD_TO_INT.get(str(word).strip(), None)


def clean_availability(text: str):
    """'In stock' / 'In stock (19 available)' -> True; anything else -> False."""
    return "in stock" in str(text).strip().lower()


def clean_dataframe(raw: pd.DataFrame) -> pd.DataFrame:
    df = raw.copy()
    df["price_gbp"] = df["price_gbp_raw"].apply(clean_price)
    df["rating"] = df["star_rating_text"].apply(clean_rating)
    df["in_stock"] = df["availability_text"].apply(clean_availability)

    n_before = len(df)
    bad_price = df["price_gbp"].isna().sum()
    bad_rating = df["rating"].isna().sum()
    print(f"\nParse-failure audit: {bad_price} unparseable price(s), {bad_rating} unparseable rating(s)")

    # Design decision (stated + justified, per assignment):
    #   - price_gbp failing to parse -> DROP the row. A missing price makes the
    #     row useless for a pricing/catalogue database and there is no sensible
    #     "typical price" to substitute across wildly different categories.
    #   - rating failing to parse (e.g. an unexpected word like "Zero", which is
    #     not one of the site's five valid star-rating labels) -> MEDIAN-IMPUTE
    #     using the median of the *successfully parsed* ratings so far, because a
    #     book's rating is a coarse 1-5 signal where the row is otherwise
    #     perfectly usable (title/price/availability/category all fine) and
    #     dropping it would discard real catalogue data over one soft field.
    if bad_price:
        df = df[df["price_gbp"].notna()].copy()
        print(f"Dropped {bad_price} row(s) with unparseable price_gbp")

    if df["rating"].isna().any():
        median_rating = int(df["rating"].median(skipna=True))
        n_imputed = df["rating"].isna().sum()
        df["rating"] = df["rating"].fillna(median_rating)
        print(f"Median-imputed {n_imputed} row(s) with unparseable rating -> {median_rating}")

    df["rating"] = df["rating"].astype(int)
    df["price_inr"] = (df["price_gbp"] * FIXED_GBP_TO_INR).round(2)

    df = df[["title", "price_gbp", "price_inr", "rating", "in_stock", "category"]].reset_index(drop=True)
    print(f"Clean rows: {len(df)} (from {n_before} raw rows)")
    return df


# ---------------------------------------------------------------------------
# Step 3/4: normalized SQLite schema + load
# ---------------------------------------------------------------------------
SCHEMA_SQL = """
DROP TABLE IF EXISTS books;
DROP TABLE IF EXISTS categories;

CREATE TABLE categories (
    category_id   INTEGER PRIMARY KEY AUTOINCREMENT,
    category_name TEXT UNIQUE NOT NULL
);

CREATE TABLE books (
    book_id     INTEGER PRIMARY KEY AUTOINCREMENT,
    title       TEXT NOT NULL,
    price_gbp   REAL NOT NULL,
    price_inr   REAL NOT NULL,
    rating      INTEGER NOT NULL CHECK (rating BETWEEN 1 AND 5),
    in_stock    INTEGER NOT NULL CHECK (in_stock IN (0, 1)),
    category_id INTEGER NOT NULL REFERENCES categories(category_id)
);
"""


def build_database(df: pd.DataFrame, db_path: Path) -> sqlite3.Connection:
    if db_path.exists():
        db_path.unlink()
    conn = sqlite3.connect(db_path)
    conn.executescript(SCHEMA_SQL)

    categories = sorted(df["category"].unique())
    conn.executemany(
        "INSERT INTO categories (category_name) VALUES (?)", [(c,) for c in categories]
    )
    cat_id = {name: cid for cid, name in conn.execute("SELECT category_id, category_name FROM categories")}

    book_rows = [
        (row.title, row.price_gbp, row.price_inr, row.rating, int(row.in_stock), cat_id[row.category])
        for row in df.itertuples()
    ]
    conn.executemany(
        """INSERT INTO books (title, price_gbp, price_inr, rating, in_stock, category_id)
           VALUES (?, ?, ?, ?, ?, ?)""",
        book_rows,
    )
    conn.commit()
    print(f"\nLoaded {len(categories)} categories and {len(book_rows)} books into {db_path.name}")
    return conn


# ---------------------------------------------------------------------------
# Step 5: required SQL queries
# ---------------------------------------------------------------------------
QUERIES = {
    "Q1_select_where_instock_expensive": """
        -- SELECT / WHERE
        SELECT title, price_gbp, rating
        FROM books
        WHERE in_stock = 1 AND price_gbp > 40
        ORDER BY price_gbp DESC;
    """,
    "Q2_order_by_limit_top10_priciest": """
        -- ORDER BY + LIMIT
        SELECT title, price_inr
        FROM books
        ORDER BY price_inr DESC
        LIMIT 10;
    """,
    "Q3_distinct_categories": """
        -- DISTINCT
        SELECT DISTINCT category_name
        FROM categories
        ORDER BY category_name;
    """,
    "Q4_in_rating_filter": """
        -- IN
        SELECT title, rating
        FROM books
        WHERE rating IN (4, 5)
        ORDER BY rating DESC, title;
    """,
    "Q5_between_price_range": """
        -- BETWEEN
        SELECT title, price_gbp
        FROM books
        WHERE price_gbp BETWEEN 20 AND 30
        ORDER BY price_gbp;
    """,
    "Q6_join_top_rated_per_category": """
        -- JOIN: 3 highest-rated books per category (ties broken by title, so the
        -- ranking is deterministic and reproducible with pd.merge + groupby below)
        WITH ranked AS (
            SELECT
                c.category_name AS category_name,
                b.title AS title,
                b.rating AS rating,
                b.price_inr AS price_inr,
                ROW_NUMBER() OVER (
                    PARTITION BY b.category_id
                    ORDER BY b.rating DESC, b.title ASC
                ) AS rnk
            FROM books b
            JOIN categories c ON b.category_id = c.category_id
        )
        SELECT category_name, title, rating, price_inr
        FROM ranked
        WHERE rnk <= 3
        ORDER BY category_name, rating DESC, title;
    """,
}


def run_queries(conn: sqlite3.Connection):
    results = {}
    for name, sql in QUERIES.items():
        print(f"\n{'=' * 70}\n{name}\n{sql.strip()}\n{'-' * 70}")
        df = pd.read_sql(sql, conn) if name in ("Q3_distinct_categories", "Q6_join_top_rated_per_category") else None
        cur = conn.execute(sql)
        cols = [d[0] for d in cur.description]
        rows = cur.fetchall()
        out_df = pd.DataFrame(rows, columns=cols)
        print(out_df.to_string(index=False))
        results[name] = out_df
    return results


# ---------------------------------------------------------------------------
# Step 6: pd.read_sql vs pd.merge equivalence for the JOIN query
# ---------------------------------------------------------------------------
def compare_read_sql_vs_merge(conn: sqlite3.Connection):
    print(f"\n{'=' * 70}\npd.read_sql vs pd.merge equivalence check (JOIN query)\n{'=' * 70}")

    # (a) pd.read_sql on the two source tables
    books_df = pd.read_sql("SELECT * FROM books", conn)
    categories_df = pd.read_sql("SELECT * FROM categories", conn)
    print(f"pd.read_sql: books={len(books_df)} rows, categories={len(categories_df)} rows")

    # (b) The join result straight from SQL (top-3-per-category), read via pd.read_sql
    sql_join_result = pd.read_sql(QUERIES["Q6_join_top_rated_per_category"], conn)
    sql_join_result = sql_join_result.sort_values(["category_name", "title"]).reset_index(drop=True)

    # (c) The equivalent join reproduced purely in-memory with pd.merge (no SQL)
    merged = pd.merge(books_df, categories_df, on="category_id", how="inner")
    # Top-3-per-category selection via merge/groupby, using the SAME tie-break
    # (rating desc, title asc) as the SQL ROW_NUMBER() window above, so the two
    # approaches are deterministic and directly comparable:
    merged_sorted = merged.sort_values(["category_name", "rating", "title"], ascending=[True, False, True])
    top3 = merged_sorted.groupby("category_name").head(3)
    merge_join_result = top3[["category_name", "title", "rating", "price_inr"]].sort_values(
        ["category_name", "title"]
    ).reset_index(drop=True)

    print("\n-- via pd.read_sql (SQL JOIN + correlated subquery) --")
    print(sql_join_result.sort_values(["category_name", "title"]).reset_index(drop=True).to_string(index=False))
    print("\n-- via pd.merge (in-memory, no SQL) --")
    print(merge_join_result.to_string(index=False))

    sql_set = set(map(tuple, sql_join_result[["category_name", "title"]].values))
    merge_set = set(map(tuple, merge_join_result[["category_name", "title"]].values))
    match = sql_set == merge_set
    print(f"\nRow sets match between pd.read_sql and pd.merge approaches: {match}")
    assert match, "pd.read_sql and pd.merge results diverged!"


def main():
    raw = load_raw()
    clean = clean_dataframe(raw)

    clean_csv = HERE / "books_clean.csv"
    clean.to_csv(clean_csv, index=False)
    print(f"Saved cleaned dataset -> {clean_csv.name}")

    conn = build_database(clean, DB_PATH)
    run_queries(conn)
    compare_read_sql_vs_merge(conn)
    conn.close()
    print(f"\nDone. Database file: {DB_PATH}")


if __name__ == "__main__":
    main()
