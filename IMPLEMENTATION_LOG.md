# IMPLEMENTATION LOG: PopForecast

This document records the **exact steps that worked** (and the rationale behind key choices) so the project can be reproduced quickly on a new machine and resumed safely in a new chat.

---

## 📅 CYCLE 01 — INFRASTRUCTURE & MVP (EDA + evaluation protocol)

### 1. Strategic definition (artifacts created)

- **`WORKFLOW_SAPE.md`** — workflow defined using SAPE (Output → Process → Input) and Fact–Dimension.
- **`ARCHITECTURE.md`** — hexagonal architecture + tech stack (Python 3.10) + folder layout.

---

### 2. Environment setup (WSL / Ubuntu 22.04)

- **Python version:** Python **3.10.12** (library compatibility for the MVP).
- **Isolation with pyenv:**
  ```bash
  pyenv install 3.10.12
  pyenv virtualenv 3.10.12 popforecast-env
  pyenv local popforecast-env
  pip install --upgrade pip
````

---

### 3. Dependency management (Poetry)

* **Initialization:**

  ```bash
  pip install poetry
  poetry init --no-interaction
  ```

* **Install packages (conflict resolution):**

  * Constraint: `kaggle` and recent `scikit-learn` versions can require newer Python; `streamlit` requires `pandas<3`.
  * MVP decision: pin compatible versions for Python 3.10.

  ```bash
  poetry add "pandas<3.0.0" "scikit-learn<1.8" xgboost streamlit fastapi uvicorn pyarrow "kaggle<1.8"
  ```

* **Status:** dependencies installed and `poetry.lock` generated.

---

### 4. Version control (Git + GitHub)

* **Goal:** reproducible and shareable repository; datasets remain local.

* **Repository initialization:**

  ```bash
  git init
  git status
  ```

* **Ignore rules:**

  * `.gitignore` prevents committing datasets and machine-local artifacts.
  * Keep `data/raw/` and `data/processed/` via `.gitkeep`.

* **Remote setup (GitHub):**

  ```bash
  git remote add origin <REPO_URL>
  git branch -M main
  git push -u origin main
  ```

* **Notes:**

  * Do not commit secrets (Kaggle credentials, tokens, `.env`).

---

### 5. Data ingestion (Kaggle)

* **Goal:** download raw dataset locally (not committed).
* **Credentials:** Kaggle API token at `~/.kaggle/kaggle.json`.
* **Command (works):**

  ```bash
  poetry run python src/scripts/download_data.py --unzip
  ```
* **Output:**

  * Target directory: `data/raw/`
  * Extracted file: `spotify_tracks_metadata.csv`

---

### 6. EDA notebook (Cycle 1)

**Artifact:** `notebooks/EDA.ipynb`

Cycle 1 EDA is intentionally scoped to:

1. validate dataset structure,
2. surface lightweight data-quality issues that affect modeling, and
3. lock MVP decisions (task framing, split strategy, leakage rules).

The goal is **reproducibility and clarity**, not exhaustive auditing.

#### 6.1 Notebook structure adopted (storytelling layout)

0. Title & Purpose (Cycle 1 objective, risks, decisions)
1. Setup (imports, settings, paths)
2. Data Loading (raw data read + sanity checks)
3. Schema Overview (columns, dtypes, memory, cardinality)
4. Data Quality Checks (missingness, duplicates, release-year consistency)
5. Target Understanding (`song_popularity`: distribution, concentration, zero mass)
6. Feature Candidates — quick scan (ranges, anomalies, weak bivariate signals)
7. Leakage & Non-usable Columns (MVP rules)
8. Baseline Evaluation Plan (task, splits, metrics; baseline protocol validation)
9. Decisions & Next Steps (compact list of closed decisions + action items)

#### 6.2 Cleaning workflow (EDA-first, production-later)

* Keep `data_raw` immutable (source of truth).
* Apply cleaning incrementally by overwriting `data_clean` inside the notebook.
* After decisions stabilize, consolidate into a **single production cleaning function/script**.

This preserves EDA speed while still producing a clean, reproducible production implementation later.

---

### 7. Key cleaning decisions implemented (Cycle 1)

#### 7.1 Duplicates by `spotify_id`

* `data_raw.duplicated().sum() == 0` → no fully identical rows.
* `data_raw.duplicated(subset=["spotify_id"]).sum() > 0` → collisions on Spotify IDs with divergent metadata.

**Decision (MVP):** for duplicated `spotify_id`, keep the record with the **maximum `song_popularity`**.
Rationale: the ID should represent the same track entity; popularity varies across updates, and max is a pragmatic, stable tie-breaker for Cycle 1.

#### 7.2 Release year consistency + diagnostic flag

Release-year metadata is imperfect (original releases vs remasters vs re-issues), and rare implausible years exist.

**Decision (MVP):**

* Keep `album_release_year` as the temporal signal for diagnostics.
* Introduce a boolean **diagnostic flag**:

  * `release_year_missing_or_suspect = True` for missing or suspicious years.
* Do not overfit cleaning rules to a few anomalies in Cycle 1; preserve traceability.

#### 7.3 Range diagnostics and anomaly counts (kept, flagged)

Cycle 1 does **not** block modeling on these edge cases, but they are recorded for later cycles:

* `tempo == 0` → likely malformed/missing tempo encoded as zero.
* `time_signature == 0` → likely invalid placeholder.
* very long tracks (`duration_ms > 20 min`) → legitimate long-tail content.
* `loudness > 0` → rare but plausible for some masters.

**Decision (MVP):** keep them for now; revisit only if they harm baseline stability.

#### 7.4 Type tightening (memory + clarity)

**Decision (MVP):** downcast to smaller dtypes where safe to reduce footprint and make intent explicit:

* `song_popularity`, `total_available_markets` → small integers (e.g., `int16`)
* `key`, `mode`, `time_signature` → small integers (e.g., `int8`)
* audio features → `float32`
* `album_release_year` → nullable integer (e.g., `Int16`)
* flags → `bool`

---

### 8. Baseline evaluation plan (no heavy modeling)

This section formalizes how Cycle 1 baseline results are reported. The baseline is intentionally lightweight; priority is an evaluation protocol that is **reproducible** and **interpretable**.

#### 8.1 Task type

`song_popularity` is a bounded discrete score in **[0, 100]**.

**Decision (MVP):** treat as a **regression task**, predicting a numeric score directly (no arbitrary binning).

#### 8.2 Split strategy

Cycle 1 uses a **dual split strategy**, with different roles:

**A) Primary benchmark: random holdout (i.i.d. reference)**

* Purpose: stable baseline under i.i.d. assumptions.
* Implementation: shuffled holdout (train/test).
  *(A train/validation split is optional later for model selection, but not required to validate the baseline protocol.)*

**B) Secondary diagnostic: temporal best-effort**

* Purpose: stress-test generalization to “nominally newer” releases.
* Temporal boundaries used:

  * Train: `album_release_year <= 2019`
  * Validation: `album_release_year == 2020`
  * Test: `album_release_year == 2021`

**Temporal hygiene rule (MVP):**

* Use integer years (no float quantiles).
* Exclude `release_year_missing_or_suspect == True` from the entire temporal diagnostic to avoid timeline noise.

#### 8.3 Metrics

All metrics are computed in “popularity points”.

* **MAE (primary):** robust, easy to interpret.
* **RMSE (secondary):** highlights large misses.
* **R² (context only):** reported for reference.

**Segment-aware diagnostics (because of zero mass):**

* MAE on `y == 0`
* MAE on `y > 0`

This prevents a baseline from looking strong merely by predicting near-zero for most tracks.

#### 8.4 Leakage prevention rule (MVP)

All preprocessing must be split-safe:

* Fit preprocessing **only on the training split**.
* Apply to validation/test via the same fitted pipeline.
* No statistics from validation/test may influence training.

#### 8.5 Baseline protocol validation inside the notebook

The notebook validates the evaluation protocol using a **constant median predictor** per split, producing:

* a consolidated metrics table (random vs temporal),
* a short interpretation of differences (notably the temporal shift in zero mass).

---

### 9. Environment portability and reproducibility notes

#### 9.1 Poetry / pyenv interaction (new machine)

On a clean machine, `poetry install` can fail when `.python-version` points to a Python version (or pyenv env name) not installed locally. Typical symptom: Poetry probes `python` via pyenv shims and exits non-zero.

**Resolution pattern (Cycle 1):**

* Install the required Python version under pyenv locally.
* Point Poetry to the correct interpreter:

  ```bash
  poetry env use <path-to-python-or-python3>
  poetry install
  ```

#### 9.2 Poetry “project could not be installed” error

If dependencies install successfully but Poetry fails installing the root project with:

> “No file/folder found for package popforecast”

**Pragmatic Cycle 1 choice:** use Poetry for dependency management only.
Two valid options:

* `poetry install --no-root`, or
* set `package-mode = false` in `pyproject.toml` under `[tool.poetry]`

(We only need imports within notebooks/scripts in Cycle 1, not a distributable package.)

#### 9.3 Kaggle credentials portability

Kaggle downloads fail on a new machine if credentials exist only on the previous setup.

**Cycle 1 rule:** treat external credentials as machine-local configuration.
Ensure `~/.kaggle/kaggle.json` exists before re-running ingestion steps.

---

### 10. Git hygiene (Cycle 1)

* Track `notebooks/EDA.ipynb` as a first-class artifact (Cycle 1 audit trail).
* Do not commit `notebooks/.ipynb_checkpoints/` (keep it ignored).
* Keep `poetry.lock` in sync with `pyproject.toml`.

Recommended commit structure:

* one commit for environment/dependency changes (pyproject + lock),
* one commit for notebook milestones (EDA structure + major section completions).

---

### 11. Open items carried into the next cycle

* Implement preprocessing inside a scikit-learn `Pipeline` (split-safe).
* Implement baseline training notebook (beyond constant median) using the locked protocol.
* Decide whether anomalies (`tempo == 0`, `time_signature == 0`) should become missing / special category / explicit flags.
* Consolidate final cleaning decisions into a production cleaning function/script (once stable).

