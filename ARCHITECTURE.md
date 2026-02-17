# ARCHITECTURE & DESIGN: PopForecast

## 1. SYSTEM OVERVIEW
* **Goal:** End-to-End Machine Learning Application to predict song popularity.
* **Core Function:** Ingest raw audio features, process via Scikit-Learn pipelines, and serve predictions via XGBoost.
* **Engineering Standard:** Production-grade code structure (Hexagonal Architecture), separating experimental notebooks from production logic.

## 2. DATA STRATEGY
* **Source:** "Spotify Tracks Dataset" (Luckey01 on Kaggle) - ~440k rows.
* **Privacy:** Raw data is NOT committed to version control (`.gitignore`).
* **Ingestion:** Automated download script via Kaggle API (`src/scripts/download_data.py`).
* **Storage:**
    * `data/raw`: Immutable CSV files downloaded from source.
    * `data/processed`: Optimized Parquet files for training.

## 3. TECH STACK
* **Runtime:** Python 3.10+ (Enforced via Pyenv).
* **Dependencies:** Managed via Poetry (`pyproject.toml`).
* **ML Core:** Scikit-Learn (Pipelines), XGBoost, Pandas, Pyarrow.
* **Backend/API:** FastAPI (REST Endpoints).
* **Frontend/UI:** Streamlit (User Dashboard).
* **Containerization:** Docker (Target for Phase 4 deployment).

## 4. DIRECTORY STRUCTURE
The project follows a modified Hexagonal Architecture.

```
.
├── data/                     # Local data storage (Ignored by Git)
│   ├── raw/                  # Original CSVs
│   └── processed/            # Parquet files (Type-safe storage)
├── models/                   # Serialized models & configuration snapshots
│   └── cycle_02/             # Artifacts specific to Cycle 2
├── notebooks/                # Experimental & Storytelling
│   └── [00-99]_*.ipynb       # Numbered sequence (e.g., 01_eda, 02_baseline)
├── src/                      # Production Source Code
│   ├── core/                 # Domain Logic & ML Pipelines
│   │   ├── preprocessing.py  # Cleaning pipelines (Sklearn)
│   │   └── features.py       # Feature Engineering logic (Cycle 2+)
│   ├── api/                  # FastAPI Adapters
│   ├── ui/                   # Streamlit Application
│   └── scripts/              # Automation (e.g., download_data.py)
├── tests/                    # Automated Tests (Pytest)
├── poetry.lock               # Exact versions of dependencies
├── pyproject.toml            # Project configuration
├── WORKFLOW_SAPE.md          # Process & Methodology Guide
└── ARCHITECTURE.md           # This file
```

## 5. DEVELOPMENT GUIDELINES (CONTRIBUTING)
* **Language:** All code, docstrings, and variable names must be in **English**.
* **Typing:** Python Type Hints are mandatory for all function signatures.
* **Notebooks:** Restricted to `/notebooks`. No production logic (.py) should live here.

* **Testing:** All core logic must be covered by unit tests.
