import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from imblearn.over_sampling import SMOTE

# === Load and prepare dataset ===
df_full = pd.read_csv("dataset.csv")
X = df_full[["feat_A", "feat_B", "feat_C"]]
y = df_full["label"]

# Fill missing values (required before SMOTE)
X = X.fillna(X.mean())

# === Split dataset ===
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=17)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=17)

# === Apply SMOTE ===
smote = SMOTE(random_state=17)
X_train_res, y_train_res = smote.fit_resample(X_train, y_train)

# === Define models ===
models = {
    "KNN": KNeighborsClassifier(n_neighbors=3),
    "Decision Tree": DecisionTreeClassifier(max_depth=None, random_state=17),
    "Random Forest": RandomForestClassifier(max_depth=None, random_state=17)
}

# === Plot ROC Curves ===
plt.figure(figsize=(8, 6))

for name, model in models.items():
    model.fit(X_train_res, y_train_res)
    y_probs = model.predict_proba(X_val)[:, 1]
    fpr, tpr, _ = roc_curve(y_val, y_probs)
    auc_score = roc_auc_score(y_val, y_probs)
    plt.plot(fpr, tpr, label=f"{name} (AUC = {auc_score:.2f})")

# Reference diagonal line
plt.plot([0, 1], [0, 1], linestyle="--", color="gray")

# Formatting
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curves for All Classifiers")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()


