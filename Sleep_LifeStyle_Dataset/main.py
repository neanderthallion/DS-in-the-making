from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, f1_score
from sklearn.model_selection import (RandomizedSearchCV, RepeatedStratifiedKFold,
                                     StratifiedKFold, cross_val_score, train_test_split)
from sklearn.preprocessing import LabelEncoder

RANDOM_STATE = 42
OUT_DIR = Path("outputs")


def load_and_clean(path="Sleep_health_and_lifestyle_dataset.csv"):
    df = pd.read_csv(path)
    df["Sleep Disorder"] = df["Sleep Disorder"].fillna("None") 
    df["BMI Category"] = df["BMI Category"].replace({"Normal Weight": "Normal"})
    df[["Systolic_BP", "Diastolic_BP"]] = df["Blood Pressure"].str.split("/", expand=True).astype(int)
    return df.drop(columns=["Person ID", "Blood Pressure"])


def encode_features(data):
    X = data.drop(columns="Sleep Disorder").copy()
    X["BMI Category"] = X["BMI Category"].map({"Normal": 0, "Overweight": 1, "Obese": 2})
    X["Gender_Male"] = (X.pop("Gender") == "Male").astype(int)
    X = pd.get_dummies(X, columns=["Occupation"], prefix="Occ", dtype=int)
    return X.astype(float)


def main():
    OUT_DIR.mkdir(exist_ok=True)
    df = load_and_clean()
    le = LabelEncoder()
    y_all = le.fit_transform(df["Sleep Disorder"])
    class_names = list(le.classes_)

    data = encode_features(df).assign(target=y_all).drop_duplicates()
    X, y = data.drop(columns="target"), data["target"].to_numpy()
    print(f"{len(df)} rows -> {len(X)} unique profiles")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, stratify=y, random_state=RANDOM_STATE)

    search = RandomizedSearchCV(
        RandomForestClassifier(random_state=RANDOM_STATE),
        {"n_estimators": [200, 300, 500, 800], "max_depth": [None, 3, 5, 8, 12],
         "min_samples_leaf": [1, 2, 3, 5], "max_features": ["sqrt", "log2", 0.5],
         "class_weight": ["balanced", "balanced_subsample", None]},
        n_iter=40, scoring="f1_macro", random_state=RANDOM_STATE, n_jobs=-1,
        cv=StratifiedKFold(5, shuffle=True, random_state=RANDOM_STATE))
    search.fit(X_train, y_train)
    model = search.best_estimator_
    pred = model.predict(X_test)
    print("Best params:", search.best_params_)
    print(classification_report(y_test, pred, target_names=class_names, digits=3))

    cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=5, random_state=RANDOM_STATE)
    scores = cross_val_score(RandomForestClassifier(**search.best_params_, random_state=RANDOM_STATE),
                             X, y, cv=cv, scoring="f1_macro", n_jobs=-1)
    print(f"Test macro-F1: {f1_score(y_test, pred, average='macro'):.3f}   "
          f"Repeated-CV macro-F1: {scores.mean():.3f} ± {scores.std():.3f}")

    sv = shap.TreeExplainer(model)(X)
    rank = pd.DataFrame(np.abs(sv.values).mean(axis=0), index=X.columns, columns=class_names)
    rank["Total"] = rank.sum(axis=1)
    rank = rank.sort_values("Total", ascending=False)
    print("\nTop factors (mean |SHAP|):")
    print(rank.head(10).round(3).to_string())

    rank.head(12).iloc[::-1][class_names].plot.barh(stacked=True, figsize=(9, 6))
    plt.title("Global SHAP importance by class")
    plt.xlabel("Mean |SHAP value|")
    plt.tight_layout()
    plt.savefig(OUT_DIR / "shap_global_importance.png", dpi=150)
    plt.close()
    for k, cls in enumerate(class_names):
        shap.plots.beeswarm(sv[:, :, k], max_display=10, show=False)
        plt.title(f"SHAP beeswarm: {cls}")
        plt.tight_layout()
        plt.savefig(OUT_DIR / f"shap_beeswarm_{cls.replace(' ', '_').lower()}.png", dpi=150)
        plt.close()

    rank.to_csv(OUT_DIR / "shap_feature_ranking.csv")
    joblib.dump({"model": model, "label_encoder": le, "features": list(X.columns)},
                OUT_DIR / "rf_sleep_model.joblib")
    print(f"\nSaved plots, ranking and model to {OUT_DIR.resolve()}")


if __name__ == "__main__":
    main()
