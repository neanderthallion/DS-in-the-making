# DS-in-the-making

A collection of my practice on correlation and prediction modelling using publicly available datasets (primarily health-related), done with the support of AI. Each folder is a self-contained mini-project with its own data, script and (where applicable) outputs.

| Project | Task | Model(s) | Headline result |
|---|---|---|---|
| [`Obesity_dataset`](#1-obesity-risk-classification) | Multi-class classification | Random Forest | 5-fold CV accuracy, Kaggle-style submission |
| [`Sleep_LifeStyle_Dataset`](#2-sleep-health--lifestyle-sleep-disorder-classification) | Multi-class classification + explainability | Tuned Random Forest, SHAP | Ranked drivers of insomnia / sleep apnea |
| [`SleepDebt_ScreenTime_Dataset`](#3-bedtime-screen-time--sleep-debt) | Correlation analysis + regression | Linear, Random Forest, XGBoost | Test R² ≈ 0.83 predicting next-day fatigue |

---

## 1. Obesity Risk Classification

**Folder:** `Obesity_dataset/` · **Script:** `solution.py`

**Goal:** predict an individual's obesity level (`NObeyesdad`, 7 classes from *Insufficient_Weight* to *Obesity_Type_III*) from eating habits, physical condition and demographics. Follows a Kaggle-playground-style setup (`train.csv` ≈ 20.7k rows, `test.csv` ≈ 13.8k rows, `sample_submission.csv`).

**Approach**
- Feature engineering: `BMI = Weight / Height²`.
- Encoding: binary yes/no → 0/1, gender → 0/1, ordinal frequency columns (`CAEC`, `CALC`) → 0–3, transport mode (`MTRANS`) → one-hot, with train/test column alignment.
- Model: `RandomForestClassifier` (500 trees), evaluated with 5-fold stratified CV, then refit on all training data.
- Output: `submission.csv` with predicted class labels for the test set.

**Run:** `cd Obesity_dataset && python solution.py`

---

## 2. Sleep Health & Lifestyle: Sleep Disorder Classification

**Folder:** `Sleep_LifeStyle_Dataset/` · **Script:** `main.py`

**Goal:** classify whether a person has *no sleep disorder*, *insomnia* or *sleep apnea* from lifestyle and health measures (age, occupation, sleep duration/quality, physical activity, stress, BMI category, blood pressure, heart rate, daily steps), and identify which factors matter most.

**Dataset:** the public *Sleep Health and Lifestyle* dataset (373 rows, collapsing to far fewer unique profiles after de-duplication).

**Approach**
- Cleaning: missing `Sleep Disorder` → "None", harmonised BMI labels, blood pressure split into systolic/diastolic.
- Duplicate profiles removed *before* splitting to avoid train/test leakage.
- `RandomForestClassifier` tuned with `RandomizedSearchCV` (40 iterations, macro-F1), 75/25 stratified split, plus repeated stratified 5×5 CV for a more stable estimate on a small dataset.
- Explainability with **SHAP** (`TreeExplainer`): global importance per class and beeswarm plots for each class.

**Outputs** (`outputs/`): `shap_global_importance.png`, `shap_beeswarm_{none,insomnia,sleep_apnea}.png`, `shap_feature_ranking.csv`, and the trained model `rf_sleep_model.joblib`.

**Key finding:** blood pressure (systolic/diastolic) and BMI category rank as the strongest predictors, followed by sleep duration, physical activity and age.

**Run:** `cd Sleep_LifeStyle_Dataset && python main.py`

---

## 3. Bedtime Screen Time & Sleep Debt

**Folder:** `SleepDebt_ScreenTime_Dataset/` · **Script:** `main.py` · **Docs:** `About`

**Dataset:** a *simulated* cross-sectional dataset of 8,500 people (evening phone use, brightness, blue-light filter, caffeine, chronotype, app genre, plus sleep metrics and fatigue scores). Because the data is synthetic and cross-sectional, results describe associations, not causal effects.

**Questions and methods**
1. **Doomscroll penalty:** how much REM sleep is lost per 30 minutes of bedtime scrolling? OLS regressions (simple vs. adjusted for confounders) with 95% CIs, for REM minutes, REM %, and total sleep.
2. **Mitigation efficacy:** does a blue-light filter offset the fatigue cost of ~100% brightness? Welch t-test, Mann-Whitney, bootstrap CI, Cohen's d, a brightness × filter interaction model and a chi-square test against sleep-debt category.
3. **Predictive modelling:** predict `next_day_fatigue_score` from *behavioural features only* (physiological outputs such as sleep hours/REM are excluded as leakage). 80/20 split, `RandomizedSearchCV` tuning of Random Forest and XGBoost, compared against mean and linear baselines, with predictions clipped to the valid 1–10 range, residual diagnostics by score bin, and permutation importance.

**Results (hold-out test set)**

| Model | R² | RMSE | MAE |
|---|---|---|---|
| Mean baseline | -0.001 | 2.731 | 2.285 |
| Linear regression | 0.819 | 1.160 | 0.868 |
| Random Forest | 0.815 | 1.174 | 0.886 |
| **XGBoost** (selected by CV RMSE) | **0.831** | **1.124** | **0.838** |

Most important behavioural features: bedtime phone minutes (by far), chronotype, occupation type and late caffeine. Residual analysis shows the models under-predict at the extremes (fatigue is capped at 10).

**Outputs** (`outputs/`): `1_doomscroll_rem.png`, `2_filter_vs_brightness.png`, `3_model_evaluation.png`, `metrics.csv`, `feature_importance.csv`, `residual_by_bin.csv`, `results.csv`; `run.log` holds the console output.

**Run:** `cd SleepDebt_ScreenTime_Dataset && python main.py`

---

## Setup

Python 3 with: `pandas`, `numpy`, `scikit-learn`, `scipy`, `statsmodels`, `xgboost`, `shap`, `matplotlib`, `seaborn`, `joblib`.

```bash
pip install pandas numpy scikit-learn scipy statsmodels xgboost shap matplotlib seaborn joblib
```

## Notes

- Projects were built as learning exercises with AI assistance; findings are exploratory, not clinical advice.
- Datasets are public or simulated; see each folder for the source files.
