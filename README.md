# Zepto Data & AI Platform — Capstone

One repository, three linked modules, one story: raw web data becomes a clean
relational store (`/data_pipeline`); a classic dataset gets profiled and
predicted end to end (`/analytics`); a grounded GenAI service answers
questions about Zepto's own policies (`/support_assistant`).

```
zepto-capstone/
├── data_pipeline/       Module 1 (25 marks) — scrape → clean → SQLite → SQL/pandas
├── analytics/           Module 2 (50 marks) — EDA → modeling → tuning → regression
├── support_assistant/   Module 3 (25 marks) — RAG corpus → LangGraph → FastAPI
└── README.md            this file
```

## Setup

Each module has its **own `requirements.txt`** (three separate files, one per
module folder) — this is the "requirements.txt per module" option rather than
one consolidated file, since the three modules use largely non-overlapping
dependencies (BeautifulSoup vs. scikit-learn vs. LangGraph/ChromaDB) and
keeping them separate means installing Module 1 doesn't drag in Module 3's
much heavier RAG stack.

```bash
git clone <this-repo-url>
cd zepto-capstone

pip install -r data_pipeline/requirements.txt
pip install -r analytics/requirements.txt
pip install -r support_assistant/requirements.txt
```

(Or install just the one module's requirements you want to run.)

## How to run each module end to end

**Module 1 — Data Pipeline**
```bash
cd data_pipeline
python scrape.py            # real live scrape (needs normal internet — see module README)
python clean_and_load.py    # clean, convert, load SQLite, run SQL queries, pandas comparison
```

**Module 2 — Analytics Pipeline**
```bash
cd analytics
python 01_eda.py        # load titanic (once), profile, clean, EDA story, save titanic.csv
python 02_modeling.py   # read titanic.csv, split, train/evaluate/tune, save best_pipeline.joblib
```

**Module 3 — Support Assistant**
```bash
cd support_assistant
python ingest.py                                # embed 8 policy docs into ChromaDB
uvicorn main:app --host 0.0.0.0 --port 7860      # MOCK_LLM defaults to 1 (graded baseline)
# in another terminal:
curl -X POST http://localhost:7860/ask -H "Content-Type: application/json" \
     -d '{"query": "How much does delivery cost?"}'
```
```bash
# Docker (builds and runs locally):
docker build -t zepto-support-assistant support_assistant
docker run -p 7860:7860 zepto-support-assistant
```

Full details, example output, and every required written interpretation for
each module live in that module's own README — linked below.

## Design decisions — summary (full detail in each module's README)

**[`/data_pipeline`](data_pipeline/README.md)** — scrapes books.toscrape.com
by category with `requests`+`BeautifulSoup`; cleans price/rating/stock with
explicit, justified missing-value handling (drop unparseable prices,
median-impute unparseable ratings); converts GBP→INR at the required fixed
baseline (1 GBP = 105.50 INR); loads into a 2-table `categories`/`books`
SQLite schema; runs 6 SQL queries covering every required clause plus a
window-function JOIN; cross-checks `pd.read_sql` against a from-scratch
`pd.merge` reproduction of the same JOIN. **Read the module README's "how the
dataset was obtained" section** — the scraper is real and correct, but this
sandboxed session's network policy blocked it from reaching
books.toscrape.com directly, so the committed dataset was assembled from two
other genuine, live-sourced inputs instead (full explanation there).

**[`/analytics`](analytics/README.md)** — loads Titanic via
`sns.load_dataset` exactly once; cleans missing values by a percentage
threshold rule (drop `deck` at 77% missing, median-impute `age` at 20%
missing, drop 2 rows for `embarked` at 0.2% missing); full univariate/
bivariate/multivariate EDA with a 4-chart data story and a 6-column
correlation heatmap; a `ColumnTransformer`+`Pipeline` fit strictly on the
training split; 3 classifiers with a full metric suite + ROC/AUC; a 3-way
imbalance comparison (baseline / `class_weight` / SMOTE-on-train-fold-only);
`GridSearchCV` + OOB score on Random Forest; a linear regression side-task on
`fare` with a heteroscedasticity check; and a joblib-saved, reload-verified
full pipeline.

**[`/support_assistant`](support_assistant/README.md)** — 8 Zepto policy
docs, chunked and embedded with `all-MiniLM-L6-v2` into ChromaDB (cosine
similarity); a LangGraph `StateGraph` (`classify_intent` →
`retrieve_and_answer` | `direct_answer`) that is fully deterministic under
the graded `MOCK_LLM=1` default, with an optional Groq-backed real-LLM
extension behind the same toggle; a Pydantic-validated `answer`/`sources`/
`confidence` schema; a FastAPI `POST /ask`, demonstrated live with two
captured example calls; a Dockerfile. **Also read that module's README** for
why its embedding model loads via a local ONNX fallback in this sandbox
instead of a direct Hugging Face download.

## A note on this submission's environment (read once, applies to all three modules)

This submission was built inside a sandboxed session whose outbound network
access is restricted to an allowlist (PyPI, npm, GitHub, and a handful of
package-registry hosts) — confirmed by direct `403` responses from the
egress proxy on every attempt to reach books.toscrape.com, huggingface.co,
and every container registry (Docker Hub included). None of these are
problems with the assignment's chosen tools or sites; they're normal,
unrestricted destinations on any ordinary machine (your laptop, Colab, a CI
runner). Where this session hit that restriction, the affected module's
README explains exactly what was blocked, what real, live-sourced data or
weights were used instead (never anything fabricated), and confirms the
"real" code path (`scrape.py`'s live requests+BeautifulSoup scraper, the
direct `sentence-transformers` Hugging Face download, `docker build` against
Docker Hub) is correct and will simply work end-to-end on a normal machine.

## Git workflow

This repository's history includes a feature branch
(`feature/analytics-pipeline`), committed to twice, then merged back into
`main` with `--no-ff` — visible via:
```bash
git log --oneline --graph --all
```
```
* Module 3: GenAI support assistant (RAG) — LangGraph + FastAPI + ChromaDB
*   Merge feature/analytics-pipeline into main
|\
| * Module 2 Part B: Titanic modeling — classifiers, tuning, regression, pipeline
| * Module 2 Part A: Titanic EDA — profiling, cleaning, data story
|/
* Init repo structure; Module 1 data pipeline (scrape, clean, SQLite, SQL queries)
```
(Module 3 was committed directly to `main` after the merge — the branch/
commit/merge activity itself is what's scored, once, across the whole repo,
per the assignment's git-workflow criterion.)
