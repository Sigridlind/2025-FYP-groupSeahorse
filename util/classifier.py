import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, confusion_matrix, roc_auc_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import KFold, GridSearchCV, train_test_split, learning_curve
from imblearn.over_sampling import SMOTE

def tune_models(x_train, y_train, x_val, y_val):
    results = {}

    cv = KFold(n_splits=5, shuffle=True, random_state=17) # Apply seeding for cross validation
    # KNN
    knn_params = {'n_neighbors': [3, 5, 7, 9]}
    knn = GridSearchCV(KNeighborsClassifier(), param_grid=knn_params, scoring='recall', cv=cv)
    knn.fit(x_train, y_train)
    knn_best = knn.best_estimator_
    results["KNN"] = evaluate_model(knn_best, x_val, y_val)
    results["KNN"]["params"] = knn.best_params_

    # Decision Tree
    dt_params = {'max_depth': [3, 5, 10, None]}
    dt = GridSearchCV(DecisionTreeClassifier(random_state=17), dt_params, scoring='recall', cv=cv)
    dt.fit(x_train, y_train)
    dt_best = dt.best_estimator_
    results["DecisionTree"] = evaluate_model(dt_best, x_val, y_val)
    results["DecisionTree"]["params"] = dt.best_params_

    # Random Forest
    rf_params = {'max_depth': [3, 5, 10, None]}
    rf = GridSearchCV(RandomForestClassifier(random_state=17), rf_params, scoring='recall', cv=cv)
    rf.fit(x_train, y_train)
    rf_best = rf.best_estimator_
    results["RandomForest"] = evaluate_model(rf_best, x_val, y_val)
    results["RandomForest"]["params"] = rf.best_params_

    return results

def evaluate_model(model, x, y):
    y_pred = model.predict(x)
    y_prob = model.predict_proba(x)[:, 1]
    acc = accuracy_score(y, y_pred)
    recall = recall_score(y, y_pred)
    precision = precision_score(y, y_pred)
    f1 = f1_score(y, y_pred)
    auc = roc_auc_score(y, y_prob)
    cm = confusion_matrix(y, y_pred)
    return {"model": model,"acc": acc,"recall": recall, "precision": precision,"f1": f1, "auc": auc, "cm": cm, "y_pred": y_pred, "y_prob": y_prob}

def classification(df, results_path, baseline= True):
    
    # select only the baseline features.
    if baseline:
        features = ["feat_A", "feat_B", "feat_C"] # [col for col in df.columns if col.startswith("feat_")]
    else: features = ["feat_A", "feat_B", "feat_C", "feat_D", "feat_E"]
    
    x_all = df[features]
    y_all = df["label"]

    # split the dataset into training and testing sets.
    x_train, x_temp, y_train, y_temp = train_test_split(x_all, y_all, test_size=0.3, random_state=17)

    # split testing sets into 15% val, 15% test
    x_val, x_test, y_val, y_test = train_test_split(x_temp, y_temp, test_size=0.5, random_state=17)

    # Apply SMOTE to training set
    smote = SMOTE(random_state=17)
    x_train_res, y_train_res = smote.fit_resample(x_train, y_train)
    
    # Tune models using validation set
    tuned_results = tune_models(x_train_res, y_train_res, x_val, y_val)
    
    final_results = {}
    for name, res in tuned_results.items():
        print(f"\n{name} Validation; F1: {res['f1']:.3f}, AUC: {res['auc']:.3f}, Accuracy: {res['acc']:.3f}, Precision: {res['precision']:.3f}, Recall: {res['recall']:.3f}")
        print(f"{name} Best Hyperparameters: {res['params']}")
            
    best_model_name = max(tuned_results, key=lambda name: tuned_results[name]["recall"])
    best_model = tuned_results[best_model_name]["model"]
    print(f"Best model is (Based on validation performance): {best_model_name}")
    
    final_result = evaluate_model(best_model, x_test, y_test)
    print(f"\n{best_model_name} Test; F1: {final_result['f1']:.3f}, AUC: {final_result['auc']:.3f}, Accuracy: {final_result['acc']:.3f}, Precision: {final_result['precision']:.3f}, Recall: {final_result['recall']:.3f}")
    print("Confusion Matrix:\n", final_result["cm"])
    
    # write test results to CSV.
    df_out = df.loc[x_test.index, ["img_id"]].copy()
    df_out["true_label"] = y_test.values
    df_out["predicted_label"] = final_result["y_pred"]
    df_out["melanoma_probability"] = final_result["y_prob"]
    df_out.to_csv(results_path, index=False)