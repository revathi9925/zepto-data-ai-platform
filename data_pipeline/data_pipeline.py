import re
import sqlite3
import sys
from urllib.parse import urljoin

import pandas as pd
import requests
from bs4 import BeautifulSoup


# ============================================================
# CONFIGURATION
# ============================================================

BASE_URL = "https://books.toscrape.com/"
GBP_TO_INR = 105.50
DB_PATH = "zepto_catalog.db"

HEADERS = {
    "User-Agent": "Mozilla/5.0"
}

RATING_MAP = {
    "One": 1,
    "Two": 2,
    "Three": 3,
    "Four": 4,
    "Five": 5,
}


# ============================================================
# 1. GET BOOK CATEGORIES
# ============================================================

def get_categories():
    """Fetch all book categories and their URLs."""

    try:
        response = requests.get(
            BASE_URL,
            headers=HEADERS,
            timeout=15
        )
        response.raise_for_status()
    except requests.RequestException as exc:
        raise RuntimeError(
            f"Could not reach Books to Scrape: {exc}"
        )

    # FIX 1: use raw bytes so BeautifulSoup detects UTF-8 itself
    # (response.text guesses ISO-8859-1 and corrupts the £ symbol).
    soup = BeautifulSoup(response.content, "html.parser")

    category_container = soup.find(
        "div",
        class_="side_categories"
    )

    if not category_container:
        raise RuntimeError(
            "Could not find the category list."
        )

    categories = []

    # The first link is "Books" (all books), so skip it.
    for tag in category_container.find_all("a")[1:]:
        category_name = tag.get_text(strip=True)
        category_url = urljoin(
            BASE_URL,
            tag.get("href", "")
        )

        if category_name and category_url:
            categories.append(
                (category_name, category_url)
            )

    return categories


# ============================================================
# 2. PARSE AND CLEAN ONE BOOK
# ============================================================

def parse_book(card, category_name):
    """Extract and clean the required fields from one book."""

    # --------------------------------------------------------
    # Title
    # --------------------------------------------------------

    title_tag = card.select_one("h3 a")

    if title_tag:
        title = (
            title_tag.get("title", "").strip()
            or title_tag.get_text(strip=True)
        )
    else:
        title = "Unknown Title"

    # --------------------------------------------------------
    # Price in GBP
    # --------------------------------------------------------

    price_element = card.select_one("p.price_color")

    price_gbp = None
    raw_price = ""

    if price_element:
        raw_price = price_element.get_text(strip=True)

        # FIX 2: extract the number with a regex so it works
        # regardless of any currency symbol or stray characters.
        match = re.search(r"\d+(?:\.\d+)?", raw_price)

        if match:
            price_gbp = float(match.group())

    # --------------------------------------------------------
    # Star Rating
    # --------------------------------------------------------

    rating_element = card.select_one("p.star-rating")

    star_rating = None
    rating_word = None

    if rating_element:
        classes = rating_element.get("class", [])

        rating_word = next(
            (
                item
                for item in classes
                if item != "star-rating"
            ),
            None
        )

        star_rating = RATING_MAP.get(rating_word)

    # --------------------------------------------------------
    # Availability
    # --------------------------------------------------------

    availability_element = card.select_one(
        "p.instock.availability"
    )

    availability_text = (
        availability_element.get_text(
            " ",
            strip=True
        )
        if availability_element
        else ""
    )

    in_stock = "In stock" in availability_text

    # price_inr is computed later, after missing values are handled,
    # so it always comes from the fixed-rate baseline.
    return {
        "title": title,
        "raw_price": raw_price,
        "price_gbp": price_gbp,
        "raw_rating": rating_word,
        "star_rating": star_rating,
        "raw_availability": availability_text,
        "in_stock": in_stock,
        "category_name": category_name,
    }


# ============================================================
# 3. SCRAPE ALL PAGES OF A CATEGORY
# ============================================================

def scrape_category(category_name, category_url):
    """
    Scrape every page in one category.
    Follows the Next button until there are no more pages.
    """

    records = []
    page_url = category_url

    while page_url:

        try:
            response = requests.get(
                page_url,
                headers=HEADERS,
                timeout=15
            )
            response.raise_for_status()
        except requests.RequestException as exc:
            # FIX 3: report the failure instead of hiding it.
            print(f"Request failed for {page_url}: {exc}")
            break

        # FIX 1 (again): use raw bytes for correct UTF-8 decoding.
        soup = BeautifulSoup(
            response.content,
            "html.parser"
        )

        cards = soup.select(
            "article.product_pod"
        )

        for card in cards:
            try:
                records.append(
                    parse_book(
                        card,
                        category_name
                    )
                )
            except Exception as exc:
                print(
                    f"Skipping a book in {category_name}: {exc}"
                )
                continue

        # Follow the category's Next link.
        next_link = soup.select_one("li.next a")

        if next_link and next_link.get("href"):
            page_url = urljoin(
                page_url,
                next_link["href"]
            )
        else:
            page_url = None

    return records


# ============================================================
# 4. COLLECT AT LEAST 60 BOOKS FROM 3 CATEGORIES
# ============================================================

def collect_minimum_books(
    min_books=60,
    min_categories=3
):
    """
    Collect at least 60 books from at least 3 categories.
    """

    categories = get_categories()

    if len(categories) < min_categories:
        raise RuntimeError(
            f"Only {len(categories)} categories found."
        )

    collected = []
    categories_used = []

    for category_name, category_url in categories:

        records = scrape_category(
            category_name,
            category_url
        )

        if records:
            collected.extend(records)
            categories_used.append(category_name)

        if (
            len(collected) >= min_books
            and len(categories_used) >= min_categories
        ):
            break

    df = pd.DataFrame(collected)

    if df.empty:
        raise RuntimeError(
            "No book data was collected."
        )

    # --------------------------------------------------------
    # Handle missing numeric values
    # --------------------------------------------------------

    # Median imputation: robust to outliers, keeps every row.
    for column in ["price_gbp", "star_rating"]:

        if df[column].isna().any():

            valid_values = df.loc[
                df[column].notna(),
                column
            ]

            if valid_values.empty:
                raise RuntimeError(
                    f"No valid values available for {column}."
                )

            df[column] = df[column].fillna(
                valid_values.median()
            )

    # --------------------------------------------------------
    # Correct data types
    # --------------------------------------------------------

    df["price_gbp"] = df["price_gbp"].astype(float)
    df["price_inr"] = (df["price_gbp"] * GBP_TO_INR).round(2)
    df["star_rating"] = df["star_rating"].round().astype(int)
    df["in_stock"] = df["in_stock"].astype(bool)

    # --------------------------------------------------------
    # Final requirement validation
    # --------------------------------------------------------

    actual_categories = df["category_name"].nunique()

    if (
        len(df) < min_books
        or actual_categories < min_categories
    ):
        raise RuntimeError(
            f"Requirement failed: collected {len(df)} books "
            f"across {actual_categories} categories. "
            f"Required at least {min_books} books across "
            f"{min_categories} categories."
        )

    return (
        df,
        sorted(df["category_name"].unique())
    )


# ============================================================
# 5. CREATE SQLITE DATABASE
# ============================================================

def create_database(df, db_path=DB_PATH):
    """Create the SQLite database and load cleaned data."""

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    try:
        cursor.execute(
            "PRAGMA foreign_keys = ON;"
        )

        # Drop child table first because it has the FK.
        cursor.execute(
            "DROP TABLE IF EXISTS books;"
        )
        cursor.execute(
            "DROP TABLE IF EXISTS categories;"
        )

        # ----------------------------------------------------
        # Categories table
        # ----------------------------------------------------

        cursor.execute(
            """
            CREATE TABLE categories (
                category_id INTEGER PRIMARY KEY AUTOINCREMENT,
                category_name TEXT UNIQUE NOT NULL
            );
            """
        )

        # ----------------------------------------------------
        # Books table
        # ----------------------------------------------------

        cursor.execute(
            """
            CREATE TABLE books (
                book_id INTEGER PRIMARY KEY AUTOINCREMENT,
                title TEXT NOT NULL,
                price_gbp REAL NOT NULL,
                price_inr REAL NOT NULL,
                star_rating INTEGER NOT NULL
                    CHECK(star_rating BETWEEN 1 AND 5),
                in_stock INTEGER NOT NULL
                    CHECK(in_stock IN (0, 1)),
                category_id INTEGER NOT NULL,
                FOREIGN KEY (category_id)
                    REFERENCES categories(category_id)
            );
            """
        )

        # ----------------------------------------------------
        # Insert categories
        # ----------------------------------------------------

        category_names = sorted(
            df["category_name"].unique()
        )

        for category_name in category_names:
            cursor.execute(
                """
                INSERT INTO categories (category_name)
                VALUES (?);
                """,
                (category_name,)
            )

        # Create category -> ID mapping.
        category_map = {
            row[1]: row[0]
            for row in cursor.execute(
                """
                SELECT category_id, category_name
                FROM categories;
                """
            ).fetchall()
        }

        # ----------------------------------------------------
        # Insert books
        # ----------------------------------------------------

        for row in df.itertuples(index=False):

            cursor.execute(
                """
                INSERT INTO books (
                    title,
                    price_gbp,
                    price_inr,
                    star_rating,
                    in_stock,
                    category_id
                )
                VALUES (?, ?, ?, ?, ?, ?);
                """,
                (
                    row.title,
                    row.price_gbp,
                    row.price_inr,
                    row.star_rating,
                    int(row.in_stock),
                    category_map[row.category_name],
                )
            )

        conn.commit()

    except Exception:
        conn.rollback()
        raise

    finally:
        conn.close()


# ============================================================
# 6. RUN REQUIRED SQL QUERIES
# ============================================================

def run_sql_queries(db_path=DB_PATH):
    """
    Run the required SQL queries and validate
    the SQL JOIN with pandas.merge().
    """

    conn = sqlite3.connect(db_path)

    try:

        # ====================================================
        # QUERY 1
        # SELECT + WHERE + ORDER BY + LIMIT
        # ====================================================

        print("\n" + "=" * 70)
        print(
            "QUERY 1 — SELECT / WHERE / ORDER BY / LIMIT"
        )
        print("=" * 70)

        q1 = """
            SELECT
                title,
                price_gbp,
                price_inr,
                star_rating
            FROM books
            WHERE star_rating >= 4
            ORDER BY price_gbp DESC
            LIMIT 5;
        """

        print(q1.strip())
        print()
        print(
            pd.read_sql_query(q1, conn)
        )

        # ====================================================
        # QUERY 2
        # DISTINCT
        # ====================================================

        print("\n" + "=" * 70)
        print("QUERY 2 — DISTINCT")
        print("=" * 70)

        q2 = """
            SELECT DISTINCT
                star_rating
            FROM books
            ORDER BY star_rating;
        """

        print(q2.strip())
        print()
        print(
            pd.read_sql_query(q2, conn)
        )

        # ====================================================
        # QUERY 3
        # BETWEEN
        # ====================================================

        print("\n" + "=" * 70)
        print("QUERY 3 — BETWEEN")
        print("=" * 70)

        q3 = """
            SELECT
                title,
                price_gbp,
                price_inr
            FROM books
            WHERE price_gbp BETWEEN 15 AND 35
            ORDER BY price_gbp
            LIMIT 5;
        """

        print(q3.strip())
        print()
        print(
            pd.read_sql_query(q3, conn)
        )

        # ====================================================
        # QUERY 4
        # JOIN
        # ====================================================

        print("\n" + "=" * 70)
        print("QUERY 4 — JOIN")
        print("=" * 70)

        q4 = """
            SELECT
                b.book_id,
                b.title,
                b.price_gbp,
                b.price_inr,
                b.star_rating,
                c.category_name
            FROM books AS b
            JOIN categories AS c
                ON b.category_id = c.category_id
            ORDER BY b.book_id
            LIMIT 5;
        """

        sql_join_preview = pd.read_sql_query(
            q4,
            conn
        )

        print(q4.strip())
        print()
        print(sql_join_preview)

        # ====================================================
        # QUERY 5
        # JOIN + GROUP BY + AVG + COUNT
        # ====================================================

        print("\n" + "=" * 70)
        print(
            "QUERY 5 — JOIN + GROUP BY + AVG + COUNT"
        )
        print("=" * 70)

        q5 = """
            SELECT
                c.category_name,
                ROUND(AVG(b.price_inr), 2) AS avg_price_inr,
                COUNT(*) AS total_books
            FROM books AS b
            JOIN categories AS c
                ON b.category_id = c.category_id
            GROUP BY c.category_name
            ORDER BY avg_price_inr DESC;
        """

        print(q5.strip())
        print()
        print(
            pd.read_sql_query(q5, conn)
        )

        # ====================================================
        # FULL SQL JOIN FOR PANDAS VALIDATION
        # ====================================================

        sql_join_df = pd.read_sql_query(
            """
            SELECT
                b.book_id,
                b.title,
                b.price_gbp,
                b.price_inr,
                b.star_rating,
                c.category_name
            FROM books AS b
            JOIN categories AS c
                ON b.category_id = c.category_id
            ORDER BY b.book_id;
            """,
            conn
        )

        # ====================================================
        # READ TABLES INTO PANDAS
        # ====================================================

        df_books = pd.read_sql_query(
            "SELECT * FROM books;",
            conn
        )

        df_categories = pd.read_sql_query(
            "SELECT * FROM categories;",
            conn
        )

        # ====================================================
        # REPRODUCE SQL JOIN USING pandas.merge()
        # ====================================================

        pandas_merge_df = (
            pd.merge(
                df_books,
                df_categories,
                on="category_id",
                how="inner"
            )[
                [
                    "book_id",
                    "title",
                    "price_gbp",
                    "price_inr",
                    "star_rating",
                    "category_name"
                ]
            ]
            .sort_values("book_id")
            .reset_index(drop=True)
        )

        sql_join_df = (
            sql_join_df
            .reset_index(drop=True)
        )

        # ====================================================
        # VALIDATE BOTH RESULTS
        # ====================================================

        print("\n" + "=" * 70)
        print("JOIN VIA SQL  ->  pd.read_sql_query(...)  (first 5 rows)")
        print("=" * 70)
        print(sql_join_df.head())

        print("\n" + "=" * 70)
        print("JOIN VIA PANDAS  ->  pd.merge(...)  (first 5 rows)")
        print("=" * 70)
        print(pandas_merge_df.head())

        pd.testing.assert_frame_equal(
            sql_join_df,
            pandas_merge_df,
            check_dtype=False
        )

        print("\n" + "=" * 70)
        print("PANDAS MERGE VALIDATION")
        print("=" * 70)
        print(
            "[PASS] pandas.merge() matches the SQL JOIN output."
        )

    finally:
        conn.close()


# ============================================================
# 7. MAIN PROGRAM
# ============================================================

def main():

    print("=" * 70)
    print(
        "ZEPTO DATA & AI PLATFORM — "
        "MODULE 1: DATA PIPELINE"
    )
    print("=" * 70)

    print(f"Data source: {BASE_URL}")
    print(
        f"GBP to INR conversion: "
        f"1 GBP = {GBP_TO_INR:.2f} INR"
    )

    # Collect at least 60 books from at least 3 categories.
    df, categories = collect_minimum_books(
        min_books=60,
        min_categories=3
    )

    # --------------------------------------------------------
    # Collection validation
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("COLLECTION VALIDATION")
    print("=" * 70)

    print(f"Books collected       : {len(df)}")
    print(
        f"Categories represented: {len(categories)}"
    )
    print(
        "Categories            : "
        + ", ".join(categories)
    )

    print(
        "60+ books requirement : "
        + ("PASS" if len(df) >= 60 else "FAIL")
    )

    print(
        "3+ categories         : "
        + ("PASS" if len(categories) >= 3 else "FAIL")
    )

    # --------------------------------------------------------
    # Create SQLite database
    # --------------------------------------------------------

    create_database(df)

    print(
        f"\nSQLite database created: {DB_PATH}"
    )

    # --------------------------------------------------------
    # Run SQL queries
    # --------------------------------------------------------

    run_sql_queries()

    # --------------------------------------------------------
    # Final status
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("MODULE 1 SUBMISSION CHECK: PASS")
    print("=" * 70)
    print(f"Database created: {DB_PATH}")


# ============================================================
# PROGRAM ENTRY POINT
# ============================================================

class Tee:
    """Write everything printed to both the console and a log file."""

    def __init__(self, *streams):
        self.streams = streams

    def write(self, data):
        for stream in self.streams:
            stream.write(data)

    def flush(self):
        for stream in self.streams:
            stream.flush()


if __name__ == "__main__":
    original_stdout = sys.stdout

    with open(
        "pipeline_output.txt",
        "w",
        encoding="utf-8"
    ) as log_file:
        sys.stdout = Tee(original_stdout, log_file)

        try:
            main()
        finally:
            sys.stdout = original_stdout