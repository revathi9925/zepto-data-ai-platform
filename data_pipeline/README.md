# Module 1 — Data Pipeline

Scrapes live product data from [books.toscrape.com](https://books.toscrape.com/),
cleans it, converts prices to INR, stores it in a normalized SQLite database,
and queries it with both SQL and pandas.

**Pipeline:** scrape -> clean -> convert -> store -> query

## Install

Requires Python 3.9+.

```bash
pip install requests beautifulsoup4 pandas
```

## Run

From inside the `data_pipeline` folder:

```bash
python data_pipeline.py
```

The script runs end to end with no manual steps. It produces:

| File | Contents |
|---|---|
| `zepto_catalog.db` | SQLite database (regenerated from scratch on every run) |
| `pipeline_output.txt` | Full log: validation checks, every SQL query string with its output, and the SQL vs pandas join comparison |

## What the script does

1. **Scrape** — Walks the site's category list and scrapes every page of each
   category (following the "Next" link) until it has at least 60 books from at
   least 3 categories. Captured per book: title, raw price text, raw star-rating
   word, raw availability text, and category.
2. **Clean** — see decisions below.
3. **Convert** — adds `price_inr`.
4. **Store** — loads two normalized tables into SQLite.
5. **Query** — runs 5 SQL queries, then reproduces the JOIN with `pd.merge()`.

## Cleaning and parsing decisions

- **Price:** the number is extracted with a regex (`\d+(\.\d+)?`) from the listed
  text (e.g. `£51.77`) and converted to `float` (`price_gbp`). A regex is used
  because the site does not declare a charset, and a naive `"£"` strip can leave
  stray characters if the symbol is mis-decoded. Pages are also parsed from raw
  bytes (`response.content`) so BeautifulSoup detects UTF-8 correctly.
- **Star rating:** the class word (`One`...`Five`) is mapped to an integer 1–5
  (`star_rating`).
- **Availability:** parsed to a boolean `in_stock` (`True` if the text contains
  "In stock").
- **Messy values:** if a numeric field (`price_gbp`, `star_rating`) fails to
  parse for a row, it is filled with the **median** of the valid values in that
  column. The median was chosen over the mean because it is not skewed by a few
  unusually expensive or cheap books, and imputing (rather than dropping) keeps
  the row count above the 60-book minimum. Individual request or parse failures
  are printed instead of silently ignored, and never crash the pipeline.

## Currency conversion

`price_inr = price_gbp × 105.50`

The project's fixed baseline rate is **1 GBP = 105.50 INR**. No live exchange-rate
API is used. `price_inr` is computed after missing-value handling, so it always
derives from the fixed rate, rounded to 2 decimals.

## Database schema

Two tables with a primary/foreign key relationship:

```sql
categories(
    category_id   INTEGER PRIMARY KEY AUTOINCREMENT,
    category_name TEXT UNIQUE NOT NULL
)

books(
    book_id     INTEGER PRIMARY KEY AUTOINCREMENT,
    title       TEXT NOT NULL,
    price_gbp   REAL NOT NULL,
    price_inr   REAL NOT NULL,
    star_rating INTEGER NOT NULL CHECK(star_rating BETWEEN 1 AND 5),
    in_stock    INTEGER NOT NULL CHECK(in_stock IN (0, 1)),
    category_id INTEGER NOT NULL REFERENCES categories(category_id)
)
```

Category names are stored once in `categories`, and each book references its
category by ID (normalization). Foreign keys are enforced with
`PRAGMA foreign_keys = ON`.

## SQL queries

| # | Demonstrates |
|---|---|
| 1 | `SELECT` / `WHERE` / `ORDER BY` / `LIMIT` — top 5 most expensive books rated 4+ stars |
| 2 | `DISTINCT` — distinct star ratings |
| 3 | `BETWEEN` — books priced £15–£35 |
| 4 | `JOIN` — books with their category names |
| 5 | `JOIN` + `GROUP BY` + `AVG` + `COUNT` — average INR price and book count per category |

The exact query strings and their results are printed by the script and saved in
`pipeline_output.txt`.

## SQL vs pandas

The Query 4 join is read with `pd.read_sql_query(...)`. The same result is then
reproduced without SQL by loading both tables into DataFrames and calling
`pd.merge(df_books, df_categories, on="category_id")`. The script prints both
outputs and asserts they are equal with `pd.testing.assert_frame_equal`
(`[PASS]` on success).