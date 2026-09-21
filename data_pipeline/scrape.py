"""
scrape.py — Module 1, Step 1: scrape raw book data from books.toscrape.com

This is the real, primary scraper for this module: it uses `requests` to fetch
category listing pages from books.toscrape.com and `BeautifulSoup` to parse each
book's fields straight out of the HTML. Run it with a normal internet connection
(your laptop, Colab, a CI runner, etc.):

    python scrape.py

It will write data_pipeline/raw_reference/books_scraped_live.csv with the same
five raw columns (title, price_gbp_raw, star_rating_text, availability_text,
category) that clean_and_load.py expects — i.e. this script and
raw_reference/build_seed_dataset.py are two different *sources* for the exact
same downstream schema, so clean_and_load.py works unmodified either way.

Why two sources exist: the sandboxed session that assembled this submission had
its outbound HTTP access restricted to an allowlist (PyPI/npm/GitHub) and could
not reach books.toscrape.com directly (confirmed 403 on the egress proxy CONNECT
for that host). See raw_reference/build_seed_dataset.py's module docstring and
the root README's "Module 1 — how the dataset was obtained" section for the full,
honest explanation of how the committed dataset was assembled instead, from two
genuine, live-sourced inputs, with no invented values. If you run *this* script
from an unrestricted network, it reproduces the same data directly from the site.
"""
import csv
import time
from pathlib import Path

import requests
from bs4 import BeautifulSoup

BASE = "https://books.toscrape.com/"
CATALOGUE = BASE + "catalogue/"

# Any >=3 categories satisfy the assignment; these 5 match the categories baked
# into raw_reference/books_seed.csv so the two sources are directly comparable.
CATEGORY_SLUGS = {
    "Mystery": "mystery_3",
    "Travel": "travel_2",
    "Historical Fiction": "historical-fiction_4",
    "Classics": "classics_6",
    "Fantasy": "fantasy_19",
}

OUT_PATH = Path(__file__).parent / "raw_reference" / "books_scraped_live.csv"


def fetch_category_pages(category: str, slug: str):
    """Yield BeautifulSoup 'article.product_pod' tags across all pages of one category."""
    url = f"{CATALOGUE}category/books/{slug}/index.html"
    page = 1
    while url:
        resp = requests.get(url, timeout=15)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "html.parser")
        for article in soup.select("article.product_pod"):
            yield article
        next_link = soup.select_one("li.next a")
        if next_link:
            page += 1
            url = f"{CATALOGUE}category/books/{slug}/page-{page}.html"
            time.sleep(0.3)  # be polite to the demo server
        else:
            url = None


def parse_article(article, category: str) -> dict:
    title = article.h3.a["title"].strip()
    price_text = article.select_one("p.price_color").get_text(strip=True)
    rating_class = article.select_one("p.star-rating")["class"]
    # class list looks like ["star-rating", "Three"] — the word rating is whichever
    # class token isn't the literal "star-rating" label.
    star_word = next(c for c in rating_class if c != "star-rating")
    availability = article.select_one("p.instock.availability").get_text(strip=True)
    return {
        "title": title,
        "price_gbp_raw": price_text,
        "star_rating_text": star_word,
        "availability_text": availability,
        "category": category,
    }


def main():
    rows = []
    for category, slug in CATEGORY_SLUGS.items():
        count = 0
        for article in fetch_category_pages(category, slug):
            rows.append(parse_article(article, category))
            count += 1
        print(f"{category}: scraped {count} books")

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f, fieldnames=["title", "price_gbp_raw", "star_rating_text", "availability_text", "category"]
        )
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nTotal scraped: {len(rows)} books across {len(CATEGORY_SLUGS)} categories")
    print(f"Wrote {OUT_PATH}")


if __name__ == "__main__":
    main()
