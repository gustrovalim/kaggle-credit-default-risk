# Credit Risk ML — Step-by-Step Plan

End-to-end credit default risk system on Kaggle's Home Credit Default Risk dataset,
built with Polars and Python, from raw tables to a monitored model service.
Theory is read in parallel, one chapter at a time, when the project reaches it.

**Book abbreviations**
- **Huyen** — *Designing Machine Learning Systems*, Chip Huyen
- **MML** — *Mathematics for Machine Learning*, Deisenroth, Faisal & Ong (free: mml-book.github.io)
- **ISLP** — *An Introduction to Statistical Learning with Python* (free: statlearning.com)

---

## Ground rules

1. One project. No new side projects until Stage 5 is done.
2. Every stage ends with something that runs.
3. Code that matters lives in a Python package with tests. Notebooks are only for exploring.
4. Polars only for data work. No falling back to SQL.
5. No Airflow, dbt, Docker or feature store unless real pain demands it.
6. Math: a fixed daily slot of about 40 minutes, tied to the current stage.
7. After every session, write 3–5 lines in `docs/learning-log.md`:
   what you did, what broke, what you learned, which decision was wrong.
8. When stuck, ask a narrow question (why is this AUC 0.99? what does this parameter do?).
   Do not go back to reading in general.

---

## Stage 0 — Project setup (1 session)

- [ ] Create the repository and initialize git.
- [ ] Create a virtual environment (uv or `python -m venv`). Python 3.11+.
- [ ] Install only what Stage 1 needs: `polars`, `numpy`, `scikit-learn`, `pytest`, `jupyterlab`.
      Add other libraries when a stage requires them, not before.
- [ ] Create the layout:
      ```
      src/credit_risk/     your package
      tests/
      notebooks/
      docs/learning-log.md
      data/raw/            Kaggle CSVs
      data/processed/      Parquet outputs
      ```
- [ ] Add `data/` to `.gitignore` before the first commit. Never commit the data.
- [ ] Move the extracted Kaggle CSVs into `data/raw/`.
- [ ] Open `HomeCredit_columns_description.csv` and skim it. It explains every column
      in every table; you will come back to it constantly.

**Done when:** the repo has a first commit, the data is in place and ignored by git.

---

## Stage 1 — A working model (1–2 weeks)

Use **only** `application_train.csv`. Ignore all other tables.

### Learn first (hands-on)
- [ ] Kaggle Learn: *Intro to Machine Learning* (a few hours).
- [ ] Kaggle Learn: *Intermediate Machine Learning* — especially missing values,
      categorical variables, and data leakage.

### Explore (notebook)
- [ ] Load `application_train.csv` with Polars. Check shape, column types, null counts.
- [ ] Look at `TARGET`: what fraction of clients defaulted? (It is heavily imbalanced —
      write down the number. It explains why accuracy is a useless metric here.)
- [ ] Look at the `DAYS_*` columns. They are negative numbers relative to the application
      date. Find the strange value in `DAYS_EMPLOYED` and decide what it means.
- [ ] Note: this dataset has **no application date**, so a true time-based split is not
      possible. You will use a stratified random split instead. Write in your log why
      a time-based split would be better in a real bank.

### Build (package code, not notebook)
- [ ] `src/credit_risk/data.py`: a function that loads the application table.
- [ ] Pick 10–15 numeric columns you think are meaningful (the `EXT_SOURCE_*` columns,
      income, credit amount, annuity, age from `DAYS_BIRTH`, employment length).
- [ ] Handle missing values (start simple: median imputation with scikit-learn's
      `SimpleImputer`, fitted on the training split only — understand why).
- [ ] Split into train and validation with `train_test_split(..., stratify=y)`.
- [ ] Train a `LogisticRegression` inside a scikit-learn `Pipeline`
      (imputer → `StandardScaler` → model). Understand why scaling matters for this model.
- [ ] Evaluate with ROC AUC on the validation split.
- [ ] `src/credit_risk/train.py`, runnable as `python -m credit_risk.train`, printing the AUC.
- [ ] One test in `tests/` that checks the loader returns the expected columns and
      that the target only contains 0 and 1.

### Theory alongside
- Huyen: chapter on model development and offline evaluation.
- ISLP: chapter on classification (logistic regression).
- MML: chapter 2 (linear algebra) — vectors, matrices, dot products, shapes.

**Done when:** `python -m credit_risk.train` prints a validation AUC, and you can explain
in your own words what ROC AUC measures and why it is used instead of accuracy.
Expect roughly 0.70–0.75. The number does not matter yet.

---

## Stage 1b — Logistic regression from scratch (1 week)

- [ ] On paper: write the logistic regression model, the sigmoid, and the log-loss
      (cross-entropy). Derive the gradient of the loss with respect to the weights.
- [ ] Write in your log why the loss is cross-entropy rather than squared error.
- [ ] Implement it in NumPy: sigmoid, loss, gradient, a gradient-descent loop.
- [ ] Train it on the same features (already imputed and scaled) and compare its AUC
      with scikit-learn's. They should be close.
- [ ] Plot the loss per iteration. Try a learning rate that is too large and one that is
      too small; describe what happens.

### Theory alongside
- MML: chapter 5 (vector calculus — gradients, chain rule).
- MML: chapter 6 (probability and distributions).

**Done when:** your NumPy model reaches nearly the same AUC as scikit-learn, and you can
derive the gradient without looking.

---

## Stage 2 — Features from the other tables (2–3 weeks)

### The tables and their keys
| Table | Grain | Joins to |
|---|---|---|
| `application_train` | one row per client application | `SK_ID_CURR` |
| `bureau` | credits at other institutions | `SK_ID_CURR` |
| `bureau_balance` | monthly status of each bureau credit | `SK_ID_BUREAU` → `bureau` |
| `previous_application` | previous Home Credit applications | `SK_ID_CURR` |
| `POS_CASH_balance` | monthly snapshots of previous POS/cash loans | `SK_ID_PREV` |
| `installments_payments` | payment history of previous loans | `SK_ID_PREV` |
| `credit_card_balance` | monthly card snapshots | `SK_ID_PREV` |

### Learn first
- [ ] Kaggle Learn: *Feature Engineering*.
- [ ] Polars user guide: expressions, `group_by().agg()`, joins, lazy API (`scan_csv`).

### Build
- [ ] Convert raw CSVs to Parquet once (`data/processed/`), so later runs are fast.
- [ ] `src/credit_risk/features.py`: one function per source table that returns one row
      per `SK_ID_CURR` with aggregated features (count, mean, max, min, sum of relevant
      columns). Start with `bureau`, then `previous_application`, then `installments_payments`.
- [ ] Generate repetitive aggregations with loops over expressions instead of writing
      them by hand — this is where the Python fluency practice is.
- [ ] Build domain features you understand from credit risk: e.g. share of installments
      paid late, average days late, ratio of amount paid to amount due, number of active
      bureau credits, ratio of credit to income.
- [ ] Join everything onto the application table with left joins. Check the row count
      did not change.
- [ ] After each new table, retrain and record the AUC change in your log.
- [ ] Install `lightgbm` and train a LightGBM model on the same features. Compare with
      logistic regression.
- [ ] Switch evaluation to stratified 5-fold cross-validation and report mean and
      standard deviation of AUC.
- [ ] Tests: each feature function returns one row per client; no duplicated
      `SK_ID_CURR`; a hand-made tiny input produces the expected aggregate values.

### Leakage checklist (write a note answering each)
- [ ] Does any feature use information the bank would not have at application time?
- [ ] Is any imputer, scaler or encoder fitted on data that includes the validation split?
- [ ] Is any feature suspiciously predictive on its own? Investigate before celebrating.

### Theory alongside
- Huyen: chapters on data engineering fundamentals, training data, feature engineering.
- ISLP: chapter on resampling methods (cross-validation); chapter on tree-based methods.

**Done when:** a feature pipeline module with tests produces one Parquet training table;
LightGBM beats the Stage 1 baseline; you have a written leakage note.

---

## Stage 3 — Experiment tracking (1 week)

- [ ] Install `mlflow`. Do the official MLflow tracking tutorial.
- [ ] Log every training run: parameters, feature set version, CV metrics, the model artifact.
- [ ] Run a small hyperparameter search on LightGBM (a handful of runs, not a huge grid).
- [ ] Compare runs in the MLflow UI.
- [ ] Register the best model in the MLflow model registry.

### Theory alongside
- Huyen: experiment tracking and model selection sections.

**Done when:** you can pick any past result and reproduce it from its MLflow run.

---

## Stage 4 — Serving (1–2 weeks)

- [ ] Install `fastapi` and `uvicorn`. Do the first chapters of the FastAPI tutorial.
- [ ] Endpoint `POST /predict`: receives a client id (or raw client records), computes
      features **using the same functions from `features.py`**, loads the registered model,
      returns the default probability.
- [ ] Define request and response schemas with Pydantic.
- [ ] Test: for a sample of clients, features computed by the serving path equal the
      features in the training table (this is the training/serving skew check).
- [ ] Test: the endpoint returns a probability between 0 and 1 and handles a missing client.

### Theory alongside
- Huyen: chapter on model deployment.

**Done when:** the service runs locally, answers predictions, and the skew test passes.

---

## Stage 5 — Monitoring (1 week)

- [ ] Install `evidently`.
- [ ] Use `application_test.csv` as the "new incoming data" (it has no labels — like real
      production traffic).
- [ ] Generate a drift report comparing its features with the training distribution.
- [ ] Simulate drift on purpose (e.g. shift incomes, change the mix of loan types) and
      confirm the report catches it.
- [ ] Wrap it as a script you could schedule: `python -m credit_risk.monitor`.

### Theory alongside
- Huyen: chapter on data distribution shifts and monitoring.

**Done when:** a script produces a drift report, and you can explain which drift metric it
used and what it measures.

---

## Wrap-up

- [ ] Write the README as a case study: problem, data, architecture, results, and the
      decisions that turned out wrong (from the learning log).
- [ ] Push to GitHub.
- [ ] Only now decide the next project (the natural sequel is real-time fraud detection).

Realistic horizon: 6–8 weeks to Stage 4, Stage 5 shortly after.
