# 🛡️ MY DATA SCIENCE STANDARD (The SAPE Way)
> *Version 2.0 — Strategy First, Code Second*

This workflow is based on the **SAPE methodology**, a Portuguese acronym for  
**Saída → Processo → Entrada** (Output → Process → Input).  
The principle is simple: start from the desired outcome, then design the process, and only then define the inputs.

---

## 🟢 PHASE 1: SAPE PLANNING (Strategy)
*Do not write a single line of code before filling this out. This connects "Step 1" to the rest of the project.*

### 1. Define the OUTPUT
*What will be physically delivered at the end?*
- [ ] **Business Question:** "How to predict song popularity before release?"
- [ ] **The Data Product:**
    - [ ] Is it a Dashboard? (Streamlit)
    - [ ] Is it an API? (FastAPI)
    - [ ] Is it a Report? (PDF/Notebook)
- [ ] **Mental Prototype:** "User inputs 'Tempo' and 'Danceability', and the system returns a score from 0 to 100."

### 2. Plan the PROCESS
*What macro tasks transform Input into Output?*
- [ ] **Step 1:** Secure data collection and storage.
- [ ] **Step 2:** Cleaning and validation (Pipelines).
- [ ] **Step 3:** Model Training (XGBoost).
- [ ] **Step 4:** API Construction.

### 3. Identify the INPUT
*What raw materials are needed?*
- [ ] **Data Sources:** Spotify API / Kaggle Dataset.
- [ ] **Requirements:** Access keys, CSV files.
- [ ] **Tools:** Python, Pandas, Scikit-Learn.

---

## 🟡 PHASE 2: ANALYSIS & FACTS (Analysis)
*We investigate the Input to see if the Process is viable.*

### 4. Dimensional Modeling (Star Schema Concept)
*Think in Facts (Events) and Dimensions (Context).*
- [ ] **Fact Table:** `SongPopularity` (Grain: One row per song).
- [ ] **Dimension Tables:** `Artist`, `Album`, `Genre`, `Time`.

### 5. Exploratory Data Analysis (EDA)
*Validate the quality of Facts and Dimensions.*
- [ ] **Fact Analysis:** Distribution of `popularity` (Target). Any outliers?
- [ ] **Dimension Context:** `key`, `genre` (if available), `explicit`.
- [ ] **Feature Dimension:** `danceability`, `energy`, `loudness`.

### 6. Hypothesis Validation (Macro/Micro)
*Combine Fact and Dimension to generate insights.*
- [ ] **Macro View:** Has average popularity changed over the years? (Line chart).
- [ ] **Micro View:** Are faster songs (`tempo`) more popular? (Scatter/Bar chart).

---

## 🔴 PHASE 3: ENGINEERING & REFINEMENT (Execution)
*Now that we know WHAT to do (SAPE) and WHERE to look (Fact-Dimension), we apply engineering.*

### 7. Environment Hygiene
*Prepare the ground for the Process defined in Step 2.*
- [ ] `pyenv local 3.10`
- [ ] `poetry init`
- [ ] Install libs: `pandas`, `scikit-learn`, `xgboost`, `streamlit`.

### 8. Modularization (The Refactor)
*Transform discovery from Phase 2 into robust software.*
- [ ] Create `src/scripts/download_data.py` (Input Automation).
- [ ] Create `src/core/preprocessing.py` (Dimension Cleaning logic).
- [ ] Create `src/core/train.py` (The engine that predicts the Fact).

---

## 🔵 PHASE 4: DELIVERY (Delivery)
*The Final Output meets the World.*

### 9. Final Validation
- [ ] Does the model answer the Business Question from Step 1?
- [ ] Is the error acceptable (RMSE/MAE)?
- [ ] Is the code reproducible on another machine?

### 10. Deployment
- [ ] Commit to Git.
- [ ] Generate `requirements.txt` / `poetry.lock`.
- [ ] Write `README.md` (Documentation).