
# 🎛️ PopForecast: AI-Powered A&R Simulator

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT) 
[![Python 3.8](https://img.shields.io/badge/python-3.10.12-blue.svg)](https://www.python.org/downloads/release/python-31012/) 
[![GitHub code size in bytes](https://img.shields.io/github/languages/code-size/Daniel-ASG/PopForecast)](https://github.com/Daniel-ASG/PopForecast) 
[![GitHub last commit](https://img.shields.io/github/last-commit/Daniel-ASG/PopForecast)](https://github.com/Daniel-ASG/PopForecast)

> An end-to-end Machine Learning platform leveraging **Mixture of Experts (MoE)** and **Explainable AI (XAI)** to forecast the organic market potential of music. It serves as a resilient, data-driven baseline for A&R (Artists and Repertoire) decision-making.

👉 **[Try the Live App on Streamlit](https://popforecast.streamlit.app/)** 👈

<a id='index'></a>
## Index

- [1. The Business Problem: Why This Exists](#Business_Problem)
    - [1.1. Performance Validation (MAE)](#Performance_Validation)
- [2. Key Features & Resilient Orchestration](#Key_Features)
- [3. Machine Learning Architecture (MoE)](#Architecture)
- [4. Technical Stack](#Tech_Stack)
- [5. Project Structure](#Project_Structure)
- [6. Interactive Case Studies (Try It Yourself)](#Case_Studies)
- [7. Next Steps & SOTA Evolution](#Next_Steps)
- [8. How to Use this Project](#How_to_Use)
- [9. Author](#Author)

---

<a id='Business_Problem'></a>
## 1. The Business Problem: Why This Exists

In the modern music industry, labels process thousands of tracks daily. Historically, A&R executives relied on intuition to predict if a song's acoustic profile matched market demand. However, success is non-linear: the traits of an underground hit differ drastically from mainstream anthems.

**PopForecast** provides a statistically grounded, bias-free baseline to mitigate risk and identify hidden organic potential.

<a id='Performance_Validation'></a>
### 1.1. Performance Validation (MAE)
Validated via a strict **temporal split** (Training < 2021, Testing >= 2021) to ensure real-world reliability and prevent data leakage.

| Model Architecture | Test MAE (Spotify Scale 0-100) | Verdict |
| :--- | :--- | :--- |
| Linear Baseline | 15.63 | Rigid; fails to capture drift |
| High-Capacity XGBoost | 14.39 | Non-linear breakthrough |
| **Mixture of Experts (MoE)** | **8.88** | **Current Champion: Regime Specialization** |

---

<a id='Key_Features'></a>
## 2. Key Features & Resilient Orchestration

* **"What-If" Studio Simulation:** Real-time sliders for acoustic features allowing producers to simulate mix adjustments before release.
* **Entity Triangulation (Cycle 08):** Uses MusicBrainz ISRCs as the source of truth to resolve identities, preventing homonym collisions.
* **Track Rescue Mechanism:** If primary searches fail, the engine triggers a deep catalog scan of the validated artist to rescue the track's DNA directly from its source.
* **Outlier Tag Mitigation:** A toggle to filter noisy, crowd-sourced tags that could artificially suppress a track's score.

---

<a id='Architecture'></a>
## 3. Machine Learning Architecture (MoE)

Determinants of success for indie artists differ mathematically from global superstars. We implemented a **Mixture of Experts** architecture:

* **The Gating Network:** Routes tracks based on the artist's Cultural Authority (Listener volume).
* **Specialized Experts:** Dedicated models for **Underground**, **Tipping Point**, and **Mainstream** regimes.
* **Explainable AI (XAI):** Powered by **SHAP**, every prediction visualizes exactly which features are pushing the score up or down.

---

<a id='Tech_Stack'></a>
## 4. Technical Stack

* **Core Engine:** Python 3.10, XGBoost, SHAP decision manifold.
* **Data Pipeline:** Async requests, MusicBrainz API, YTMusic API (fallback context), ReccoBeats API.
* **Interface:** Streamlit with custom CSS and interactive Plotly Radar Charts.

---

<a id='Project_Structure'></a>
## 5. Project Structure

```text
PopForecast/
├── .streamlit/           # Streamlit configuration and secrets
├── data/                 # Local data storage (ignored by git)
├── models/               # Trained MoE Experts and metadata
├── notebooks/            # Research, EDA, and Model Auditing
├── src/                  # Source code
│   ├── api/              # API Clients (Spotify, Last.fm)
│   ├── data/             # Raw data preprocessing logic
│   ├── features/         # Feature engineering and data contracts
│   ├── models/           # Model training and evaluation routines
│   └── ui/               # Streamlit Web Application
├── pyproject.toml        # Poetry dependency management
└── IMPLEMENTATION_LOG.md # Technical decision history

```

---

<a id='Case_Studies'></a>
## 6. Interactive Case Studies (Try It Yourself)

* **The "Hollywood Sync" Effect (Kate Bush - *Running Up That Hill*):** The model predicts a high baseline for this 80s track; the current 78 popularity reflects the unquantifiable *Stranger Things* sync.
* **The Metadata Rescue (Gilberto Gil - *Sandra*):** Search for this track to see the **Cycle 08 Rescue Operation** in action, bypassing the lack of ISRCs in open databases.
* **The Genre Friction (ANGRA - *Wuthering Heights*):** MoE detects the "Metal" tag and predicts a lower ceiling due to algorithmic genre friction in global playlists.

---

<a id='Next_Steps'></a>

## 7. Next Steps

To further enhance the models and the overall platform, I am considering the following steps:

* **Feature Engineering (Lyrics & Social Media):** Integrate NLP models to quantify the emotional valency of lyrics, and include external features like TikTok trending hashtag frequency to predict the "Marketing Delta" (the difference between the Organic Floor and the Viral Ceiling).
* **Latency–Accuracy Trade-off as a Strategic Choice:** The current system prioritizes sub‑second inference by relying on lightweight tabular acoustic features, but future iterations will intentionally shift this balance. By adopting **Mel‑spectrograms** and **2D CNNs** to extract representations directly from raw audio, the model can capture transient and fine‑grained sonic patterns that external APIs cannot provide. This deep‑audio approach reduces dependency on third‑party acoustic attributes and embraces a deliberate trade‑off: **higher computational cost in exchange for significantly improved predictive accuracy**.
* **Time-Series Decay Modeling:** Enhance the prediction logic to better map the natural popularity decay curve of catalog tracks over decades.
* **Deployment and Containerization:** Containerize the application using Docker and deploy it to a cloud environment (e.g., AWS, GCP) to ensure high availability and continuous monitoring for concurrent A&R users.

---

<a id='How_to_Use'></a>

## 8. How to Use this Project

This section guides you through running the PopForecast Live A&R Simulator on your local machine.

**Prerequisites:**

* This project was developed using [Python](https://www.python.org/) version 3.10.12.
* You need to have [Poetry](https://python-poetry.org/) installed for strict dependency management.

**Core Libraries:**

* `streamlit`
* `xgboost`
* `shap`
* `requests`
* `pandas`

**Instructions:**

Don't want to install dependencies? 
**[Play with the live version on Streamlit Cloud](https://popforecast.streamlit.app/)** instead!

If you prefer to run the environment locally, follow these steps:

1. Clone this repository to your local machine:
```bash
git clone [https://github.com/Daniel-ASG/PopForecast.git](https://github.com/Daniel-ASG/PopForecast.git)
cd PopForecast

```


2. Install the required dependencies via Poetry:
```bash
poetry install

```


3. Set up your API credentials. Create a `.streamlit` folder and a `secrets.toml` file:
```bash
mkdir .streamlit
touch .streamlit/secrets.toml

```


*Inside the `secrets.toml` file, add your Last.fm API Key:*
```toml
LASTFM_API_KEY = "your_lastfm_key_here"

```


4. Run the Streamlit application:
```bash
poetry run streamlit run src/ui/app.py

```

---

<a id='Author'></a>

## 9. Author

Made by Daniel Gomes. Feel free to reach out to discuss music data, machine learning architecture, or A&R analytics!

[![Website](https://img.shields.io/website?url=https%3A%2F%2Fgithub.com&up_message=Portfolio)](https://daniel-asg.github.io/portfolio_projetos/)
[![Email](https://img.shields.io/badge/Email-D14836?logo=gmail&logoColor=white)](mailto:daniel.alexandre.eng@gmail.com)
[![LinkedIn](https://img.shields.io/badge/LinkedIn-%230077B5.svg?logo=linkedin&logoColor=white)](https://www.linkedin.com/in/daniel-asgomes)

