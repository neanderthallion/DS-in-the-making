import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder

RANDOM_STATE = 42
DATA_DIR = "."


# 1) Load data
train = pd.read_csv(f"{DATA_DIR}/train.csv")
test = pd.read_csv(f"{DATA_DIR}/test.csv")
sample_submission = pd.read_csv(f"{DATA_DIR}/sample_submission.csv")

print("train shape:", train.shape)
print("test shape:", test.shape)
print(train.head())
print(train.info())
print("\nMissing values per column:\n", train.isna().sum())
print("\nTarget class balance:\n", train["NObeyesdad"].value_counts())


# 2) Feature engineering
for df in (train, test):
    df["BMI"] = df["Weight"] / (df["Height"] ** 2)


# 3) Encode categorical features
# 3a) Binary yes/no columns 
binary_cols = ["family_history_with_overweight", "FAVC", "SMOKE", "SCC"]
for col in binary_cols:
    mapping = {"no": 0, "yes": 1}
    train[col] = train[col].map(mapping)
    test[col] = test[col].map(mapping)

# 3b) Gender
gender_map = {"Female": 0, "Male": 1}
train["Gender"] = train["Gender"].map(gender_map)
test["Gender"] = test["Gender"].map(gender_map)

# 3c) Ordinal columns --> ordinal encode 
ordinal_map = {"no": 0, "Sometimes": 1, "Frequently": 2, "Always": 3}
for col in ["CAEC", "CALC"]:
    train[col] = train[col].map(ordinal_map)
    test[col] = test[col].map(ordinal_map)

# 3d) MTRANS (transport mode) --> one-hot encode
train = pd.get_dummies(train, columns=["MTRANS"], prefix="MTRANS")
test = pd.get_dummies(test, columns=["MTRANS"], prefix="MTRANS")
# get_dummies can produce mismatched columns between train/test if a category
# is missing from one split -- align them so both frames have identical
# feature columns in the same order before modeling.
train, test = train.align(test, join="left", axis=1, fill_value=0)
test = test.drop(columns=["NObeyesdad"], errors="ignore")


# 4) Encode target
target_encoder = LabelEncoder()
y = target_encoder.fit_transform(train["NObeyesdad"])

feature_cols = [c for c in train.columns if c not in ("id", "NObeyesdad")]
X = train[feature_cols]
X_test = test[feature_cols]


# 5) Cross-validate 
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
fold_scores = []

for fold, (train_idx, val_idx) in enumerate(skf.split(X, y), start=1):
    X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
    y_train, y_val = y[train_idx], y[val_idx]

    model = RandomForestClassifier(
        n_estimators=500,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)

    val_preds = model.predict(X_val)
    score = accuracy_score(y_val, val_preds)
    fold_scores.append(score)
    print(f"Fold {fold} accuracy: {score:.4f}")

print(f"\nMean CV accuracy: {np.mean(fold_scores):.4f} (+/- {np.std(fold_scores):.4f})")


# 6) Fit final model on all training data
final_model = RandomForestClassifier(
    n_estimators=500,
    random_state=RANDOM_STATE,
    n_jobs=-1,
)
final_model.fit(X, y)


# 7) Predict on test set and write submission
test_preds = final_model.predict(X_test)
test_labels = target_encoder.inverse_transform(test_preds)

submission = pd.DataFrame({
    "id": test["id"],
    "NObeyesdad": test_labels,
})
submission = submission[sample_submission.columns]
submission.to_csv(f"{DATA_DIR}/submission.csv", index=False)

print("\nWrote submission.csv with shape:", submission.shape)
print(submission.head())
