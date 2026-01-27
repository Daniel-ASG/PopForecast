# 🛡️ MY DATA SCIENCE STANDARD (The SAPE Way)
> *Version 2.0 - Strategy First, Code Second*

Este fluxo garante que a tecnologia sirva ao negócio, e não o contrário.
Baseado na metodologia SAPE: Saída -> Processo -> Entrada.

---

## 🟢 FASE 1: PLANEJAMENTO SAPE (Strategy)
*Não escreva uma linha de código antes de preencher isso. Isso conecta o "Passo 1" ao resto do projeto.*

### 1. Definir a SAÍDA (Output)
*O que será entregue fisicamente ao final?*
- [ ] **Pergunta de Negócio:** "Como prever a popularidade de uma música antes do lançamento?"
- [ ] **O Produto de Dados:**
    - [ ] É um Dashboard? (Streamlit)
    - [ ] É uma API? (FastAPI)
    - [ ] É um Relatório? (PDF/Notebook)
- [ ] **Protótipo Mental:** "O usuário digita o 'Tempo' e a 'Danceability', e o sistema retorna um score de 0 a 100."

### 2. Planejar o PROCESSO (Process)
*Quais tarefas macro transformam a Entrada na Saída?*
- [ ] **Passo 1:** Coleta e armazenamento seguro dos dados.
- [ ] **Passo 2:** Limpeza e validação (Pipelines).
- [ ] **Passo 3:** Treinamento do Modelo (XGBoost).
- [ ] **Passo 4:** Construção da API/Interface.

### 3. Identificar as ENTRADAS (Input)
*Quais fontes estão disponíveis e acessíveis?*
- [ ] **Fonte:** Kaggle (Spotify Tracks Dataset).
- [ ] **Formato:** CSV Bruto (~440k linhas).
- [ ] **Restrições:** Dados não podem subir para o GitHub (>100MB). Necessário script de download.

---

## 🟡 FASE 2: DESCOBERTA FATO-DIMENSÃO (Analysis)
*Aqui usamos a filosofia "Fato-Dimensão" para não fazer gráficos inúteis.*
*Local: `notebooks/01_exploration.ipynb`*

### 4. Definição do FATO (O Alvo)
*Qual é a métrica numérica central que queremos analisar?*
- [ ] **Fato:** `song_popularity` (0-100).
- [ ] **Objetivo:** Entender o que faz esse número subir ou descer.

### 5. Definição das DIMENSÕES (O Contexto)
*Quais atributos qualitativos ou temporais explicam o fato?*
- [ ] **Dimensão Tempo:** `year` (A música mudou com o tempo?).
- [ ] **Dimensão Produto (Música):** `key`, `genre` (se houver), `explicit`.
- [ ] **Dimensão Característica:** `danceability`, `energy`, `loudness`.

### 6. Validação de Hipóteses (Macro/Micro)
*Combine Fato e Dimensão para gerar insights.*
- [ ] **Visão Macro:** A popularidade média mudou ao longo dos anos? (Linha).
- [ ] **Visão Micro:** Músicas mais rápidas (`tempo`) são mais populares? (Dispersão/Barra).

---

## 🔴 FASE 3: ENGENHARIA & REFINAMENTO (Execution)
*Agora que sabemos O QUE fazer (SAPE) e ONDE olhar (Fato-Dimensão), aplicamos a engenharia.*

### 7. Higiene do Ambiente
*Preparar o terreno para o Processo definido no Passo 2.*
- [ ] `pyenv local 3.10`
- [ ] `poetry init`
- [ ] Instalar libs: `pandas`, `scikit-learn`, `xgboost`, `streamlit`.

### 8. Modularização (The Refactor)
*Transformar a descoberta da Fase 2 em software robusto.*
- [ ] Criar `src/scripts/download_data.py` (Automação da Entrada).
- [ ] Criar `src/core/preprocessing.py` (A lógica de limpeza das Dimensões).
- [ ] Criar `src/core/train.py` (O motor que prevê o Fato).

---

## 🔵 FASE 4: ENTREGA DO PRODUTO (Delivery)
*Materializar a "Saída" definida no Passo 1.*

### 9. Construção da Interface
- [ ] Backend: FastAPI (para servir o modelo).
- [ ] Frontend: Streamlit (para o usuário interagir com as Dimensões e ver o Fato previsto).

### 10. Documentação Final
- [ ] Atualizar `README.md` explicando o problema de negócio e a solução.
- [ ] Garantir que `ARCHITECTURE.md` reflete a estrutura técnica.