from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import statsmodels.formula.api as smf
from scipy import stats
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, RandomizedSearchCV, cross_validate, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from xgboost import XGBRegressor

BASE = Path(__file__).parent
OUT = BASE / "outputs"
OUT.mkdir(exist_ok=True)
SEED = 42

df = pd.read_csv(BASE / "bedtime_screentime_sleep_debt.csv")
assert df.isna().sum().sum() == 0, "unexpected missing values"
assert not df.duplicated().any() and df["user_id"].is_unique, "duplicates found"
assert df["screen_brightness_pct"].between(10, 100).all()
assert df["blue_light_filter_active"].isin([0, 1]).all()
print(f"Loaded {df.shape[0]} rows, {df.shape[1]} columns; no nulls or duplicates.\n")

### 1. The Doomscroll Penalty
print("=" * 60, "\n1. DOOMSCROLL PENALTY\n", "=" * 60)
# Transforming REM
df["rem_minutes"] = df["rem_sleep_pct"] / 100 * df["total_sleep_hours"] * 60
df["scroll_30min"] = df["bedtime_phone_minutes"] / 30

controls = ("caffeine_post_5pm_mg + age + C(chronotype) + C(primary_bedtime_app)"
            " + screen_brightness_pct + blue_light_filter_active")
results = {}
for label, y in [("REM minutes", "rem_minutes"), ("REM % of sleep", "rem_sleep_pct"),
                 ("Total sleep hours", "total_sleep_hours")]:
    for tag, f in [("simple", f"{y} ~ scroll_30min"), ("adjusted", f"{y} ~ scroll_30min + {controls}")]:
        m = smf.ols(f, df).fit()
        b, (lo, hi), p = m.params["scroll_30min"], m.conf_int().loc["scroll_30min"], m.pvalues["scroll_30min"]
        results[(label, tag)] = m
        print(f"{label:18s} {tag:9s} per 30 min: {b:+.3f} (95% CI {lo:+.3f}, {hi:+.3f}) p={p:.3g}  R2={m.rsquared:.3f}")

# Visualize a regression line
fig, ax = plt.subplots(figsize=(8, 5))
ax.hexbin(df["bedtime_phone_minutes"], df["rem_minutes"], gridsize=40, cmap="Blues", mincnt=1)
sns.regplot(x="bedtime_phone_minutes", y="rem_minutes", data=df, scatter=False, ci=95,
            line_kws={"color": "crimson"}, ax=ax)
bins = pd.cut(df["bedtime_phone_minutes"], 12)
bm = df.groupby(bins, observed=True).agg(x=("bedtime_phone_minutes", "mean"), y=("rem_minutes", "mean"))
ax.plot(bm["x"], bm["y"], "o-", color="black", ms=4, label="binned mean")
ax.set(xlabel="Bedtime phone minutes", ylabel="REM sleep (minutes)", title="Doomscroll penalty on REM sleep")
ax.legend()
fig.tight_layout(); fig.savefig(OUT / "1_doomscroll_rem.png", dpi=150); plt.close(fig)

### 2. Mitigation Efficacy
print("\n" + "=" * 60, "\n2. MITIGATION EFFICACY\n", "=" * 60)
hi_b = df[df["screen_brightness_pct"] >= 90]
off = hi_b.loc[hi_b["blue_light_filter_active"] == 0, "next_day_fatigue_score"]
on = hi_b.loc[hi_b["blue_light_filter_active"] == 1, "next_day_fatigue_score"]
t, p = stats.ttest_ind(on, off, equal_var=False)
u, pu = stats.mannwhitneyu(on, off)
sp = np.sqrt(((len(on) - 1) * on.var() + (len(off) - 1) * off.var()) / (len(on) + len(off) - 2))
rng = np.random.default_rng(SEED)
boot = [rng.choice(on, len(on)).mean() - rng.choice(off, len(off)).mean() for _ in range(5000)]
print(f"Brightness >= 90%: filter off n={len(off)} mean={off.mean():.2f}; on n={len(on)} mean={on.mean():.2f}")
print(f"Diff (on-off)={on.mean() - off.mean():+.3f}, bootstrap 95% CI {np.percentile(boot, [2.5, 97.5]).round(3)}, "
      f"Welch p={p:.3g}, Mann-Whitney p={pu:.3g}, Cohen d={(on.mean() - off.mean()) / sp:.2f}")

m2 = smf.ols("next_day_fatigue_score ~ screen_brightness_pct * blue_light_filter_active"
             " + bedtime_phone_minutes + C(primary_bedtime_app)", df).fit()
print(m2.summary().tables[1])
ref = {"bedtime_phone_minutes": df["bedtime_phone_minutes"].median(),
       "primary_bedtime_app": df["primary_bedtime_app"].mode()[0]}
pred = lambda b, f: float(m2.predict(pd.DataFrame([{**ref, "screen_brightness_pct": b, "blue_light_filter_active": f}]))[0])
print(f"Predicted fatigue @100% + filter: {pred(100, 1):.2f} | @100% no filter: {pred(100, 0):.2f} "
      f"| @10% no filter: {pred(10, 0):.2f}")
print("Filter fully offsets the 100%-brightness penalty" if pred(100, 1) <= pred(10, 0)
      else "Filter does NOT fully offset the 100%-brightness penalty")

# Secondary categorical view: filter x sleep_debt_category
ct = pd.crosstab(df["blue_light_filter_active"], df["sleep_debt_category"])
chi2, pc, dof, _ = stats.chi2_contingency(ct)
print(f"\nChi-square filter x sleep_debt_category: chi2={chi2:.2f}, dof={dof}, p={pc:.3g}, "
      f"Cramer's V={np.sqrt(chi2 / (ct.values.sum() * (min(ct.shape) - 1))):.3f}")

fig, ax = plt.subplots(figsize=(8, 5))
d2 = df.assign(b=(df["screen_brightness_pct"] // 10 * 10).clip(upper=90),
               Filter=df["blue_light_filter_active"].map({0: "off", 1: "on"}))
sns.lineplot(data=d2, x="b", y="next_day_fatigue_score", hue="Filter", errorbar=("ci", 95), marker="o", ax=ax)
ax.set(xlabel="Screen brightness (10% bins)", ylabel="Mean next-day fatigue", title="Fatigue vs brightness by blue-light filter")
fig.tight_layout(); fig.savefig(OUT / "2_filter_vs_brightness.png", dpi=150); plt.close(fig)

### 3. Predictive Modeling
print("\n" + "=" * 60, "\n3. PREDICTIVE MODELING (behavioral features)\n", "=" * 60)
# 3.2 Data cleaning
num = ["age", "bedtime_phone_minutes", "screen_brightness_pct", "blue_light_filter_active",
       "caffeine_post_5pm_mg", "physical_activity_min"]
cat = ["gender", "occupation_type", "chronotype", "primary_bedtime_app"]
leak = ["sleep_latency_min", "total_sleep_hours", "deep_sleep_pct", "rem_sleep_pct",
        "morning_alarm_snoozes", "sleep_debt_category", "user_id"]
assert not set(num + cat) & set(leak)
target = "next_day_fatigue_score"
X, y = df[num + cat], df[target]

# 3.3 Train: 80/20 split, pipeline
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=SEED)
prep = ColumnTransformer([("cat", OneHotEncoder(handle_unknown="ignore"), cat)], remainder="passthrough")
mk = lambda m: Pipeline([("prep", prep), ("model", m)])
LO, HI = 1.0, 10.0

# Hyperparameter search: 5-fold CV on the training set only
cv5 = KFold(5, shuffle=True, random_state=SEED)
searches = {
    "Random Forest": RandomizedSearchCV(
        mk(RandomForestRegressor(n_estimators=300, n_jobs=-1, random_state=SEED)),
        {"model__min_samples_leaf": [2, 3, 5, 10], "model__max_features": [0.5, 0.8, 1.0]},
        n_iter=12, cv=cv5, scoring="neg_root_mean_squared_error", random_state=SEED),
    "XGBoost": RandomizedSearchCV(
        mk(XGBRegressor(objective="reg:squarederror", tree_method="hist", n_jobs=-1, random_state=SEED)),
        {"model__n_estimators": [200, 400, 600, 900],
         "model__learning_rate": [0.02, 0.03, 0.05, 0.1],
         "model__max_depth": [2, 3, 4, 5],
         "model__min_child_weight": [1, 5, 10, 20],
         "model__subsample": [0.7, 0.8, 1.0],
         "model__colsample_bytree": [0.7, 0.8, 1.0],
         "model__reg_lambda": [1, 3, 5, 10]},
        n_iter=30, cv=cv5, scoring="neg_root_mean_squared_error", random_state=SEED, n_jobs=1),
}
for k, s_ in searches.items():
    s_.fit(X_tr, y_tr)
    print(f"{k} best CV RMSE={-s_.best_score_:.4f} params={ {p.split('__')[1]: v for p, v in s_.best_params_.items()} }")

pipes = {"Mean baseline": mk(DummyRegressor()), "Linear regression": mk(LinearRegression()),
         **{k: s_.best_estimator_ for k, s_ in searches.items()}}

print("\n5-fold CV on training set (re-scored, tuned params):")
cv_rmse = {}
for k, pipe in pipes.items():
    cv = cross_validate(pipe, X_tr, y_tr, cv=cv5, scoring=("r2", "neg_root_mean_squared_error"))
    cv_rmse[k] = -cv["test_neg_root_mean_squared_error"].mean()
    print(f"  {k:18s} R2={cv['test_r2'].mean():.3f}±{cv['test_r2'].std():.3f}  RMSE={cv_rmse[k]:.3f}")

# 3.4 Evaluate once on the held-out test set; predictions clipped to the valid score range
print("\nHold-out test set (predictions clipped to [1, 10]):")
metrics, preds = [], {}
for k, pipe in pipes.items():
    pipe.fit(X_tr, y_tr)
    preds[k] = np.clip(pipe.predict(X_te), LO, HI)
    metrics.append({"model": k, "CV_RMSE": cv_rmse[k], "R2": r2_score(y_te, preds[k]),
                    "RMSE": np.sqrt(mean_squared_error(y_te, preds[k])),
                    "MAE": mean_absolute_error(y_te, preds[k])})
mdf = pd.DataFrame(metrics).round(4)
print(mdf.to_string(index=False))

# Final model chosen by CV RMSE (not test score) among the tuned tree models
best = min(searches, key=cv_rmse.get)
print(f"\nSelected model (lowest CV RMSE): {best}")
best_pipe, pred_te = pipes[best], preds[best]

# Extreme-value diagnostics: error by ACTUAL and by PREDICTED score bin
edges = [0, 2, 4, 6, 8, 10.01]
diag = []
for k in searches:
    r = y_te.values - preds[k]
    for by, v in [("actual", y_te.values), ("predicted", preds[k])]:
        g = pd.DataFrame({"bin": pd.cut(v, edges), "r": r}).groupby("bin", observed=True)["r"]
        d_ = g.agg(n="count", bias="mean", rmse=lambda x: np.sqrt((x ** 2).mean())).reset_index()
        diag.append(d_.assign(model=k, binned_by=by))
diag = pd.concat(diag)
print("\nResidual bias / RMSE by score bin:\n",
      diag.pivot_table(index=["binned_by", "bin"], columns="model", values=["bias", "rmse"], observed=True).round(3).to_string())

imp = permutation_importance(best_pipe, X_te, y_te, n_repeats=10, random_state=SEED, n_jobs=-1)
fi = pd.DataFrame({"feature": X_te.columns, "importance": imp.importances_mean,
                   "std": imp.importances_std}).sort_values("importance", ascending=False)
print(f"\nPermutation importance ({best}, test):\n", fi.round(4).to_string(index=False))

fig, axes = plt.subplots(2, 3, figsize=(16, 9))
for j, k in enumerate(searches):
    axes[0, j].scatter(y_te, preds[k], s=6, alpha=0.4); axes[0, j].plot([1, 10], [1, 10], "r--")
    axes[0, j].set(xlabel="Actual", ylabel="Predicted", title=f"{k}: actual vs predicted")
    axes[1, j].scatter(preds[k], y_te - preds[k], s=6, alpha=0.4); axes[1, j].axhline(0, color="r", ls="--")
    axes[1, j].set(xlabel="Predicted", ylabel="Residual", title=f"{k}: residuals vs predicted")
axes[0, 2].barh(fi["feature"][::-1], fi["importance"][::-1]); axes[0, 2].set(title=f"{best}: permutation importance (ΔR²)")
rb = diag[diag.binned_by == "actual"].pivot(index="bin", columns="model", values="bias")
rb.plot.bar(ax=axes[1, 2], rot=30); axes[1, 2].axhline(0, color="k", lw=0.5)
axes[1, 2].set(title="Residual bias by ACTUAL score bin", xlabel="", ylabel="mean(actual - predicted)")
fig.tight_layout(); fig.savefig(OUT / "3_model_evaluation.png", dpi=150); plt.close(fig)

# 3.5 Export
pd.DataFrame({"row_index": X_te.index, "actual": y_te.values, "predicted": pred_te.round(3),
              "residual": (y_te.values - pred_te).round(3), "model": best}).to_csv(BASE / "results.csv", index=False)
fi.to_csv(OUT / "feature_importance.csv", index=False)
mdf.to_csv(OUT / "metrics.csv", index=False)
diag.to_csv(OUT / "residual_by_bin.csv", index=False)
print(f"\nExported results.csv ({len(X_te)} rows, model={best}), outputs/feature_importance.csv, "
      "outputs/metrics.csv, outputs/residual_by_bin.csv, 3 figures.")
print("Note: data is simulated/cross-sectional (associations, not causal); fatigue is capped at 10.")
