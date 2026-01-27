# IMPLEMENTATION LOG: PopForecast

Este documento registra os passos exatos que funcionaram, servindo como guia de reprodução e treinamento.

## 📅 CICLO 01: INFRAESTRUTURA & MVP

### 1. Definição Estratégica (Arquivos Criados)
* **`WORKFLOW_SAPE.md`**: Definido o fluxo de trabalho baseado no método SAPE (Saída -> Processo -> Entrada) e Fato-Dimensão.
* **`ARCHITECTURE.md`**: Definida a arquitetura Hexagonal, stack tecnológico (Python 3.10) e estrutura de pastas.

### 2. Configuração de Ambiente (WSL/Ubuntu 22.04)
* **Python Version:** Definido uso do Python 3.10.12 para garantir compatibilidade de bibliotecas ML.
* **Isolamento:**
  ```bash
  pyenv install 3.10.12
  pyenv virtualenv 3.10.12 popforecast-env
  pyenv local popforecast-env
  pip install --upgrade pip
  ```

### 3. Gerenciamento de Dependências (Poetry)

* **Inicialização:**
```bash
pip install poetry
poetry init --no-interaction

```


* **Instalação de Pacotes (Solução de Conflitos):**
* *Desafio:* Bibliotecas `kaggle` e `scikit-learn` (v1.8+) exigiam Python 3.11+. `streamlit` exigia `pandas<3`.
* *Solução:* Travamento de versões específicas compatíveis com Python 3.10.


```bash
poetry add "pandas<3.0.0" "scikit-learn<1.8" xgboost streamlit fastapi uvicorn pyarrow "kaggle<1.8"

```


* **Status:** Dependências instaladas e `poetry.lock` gerado com sucesso.



### 4. Version Control (Git + GitHub)

* **Goal:** Make the project reproducible and shareable via a remote repository.
* **Repository initialization:**
  ```bash
  git init
  git status
````

* **Ignore rules:**

  * Added `.gitignore` to prevent committing datasets and local artifacts.
  * Ensured `data/raw/` and `data/processed/` are kept via `.gitkeep` files.

* **First commits (Conventional Commits):**

  * `chore: initial project scaffold`

* **Remote setup (GitHub):**

  ```bash
  git remote add origin <REPO_URL>
  git branch -M main
  git push -u origin main
  ```

* **Notes:**

  * Dataset files must remain local (downloaded via `src/scripts/download_data.py`).
  * Do not commit secrets (Kaggle credentials, tokens, .env).
