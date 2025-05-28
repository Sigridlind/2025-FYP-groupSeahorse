import numpy as np
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, confusion_matrix, roc_auc_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV

def tune_models(x_train, y_train, x_val, y_val):
    results = {}

    # KNN
    knn_params = {'n_neighbors': [3, 5, 7, 9]}
    knn = GridSearchCV(KNeighborsClassifier(), param_grid=knn_params, scoring='f1')
    knn.fit(x_train, y_train)
    knn_best = knn.best_estimator_
    results["KNN"] = evaluate_model(knn_best, x_val, y_val)
    results["KNN"]["params"] = knn.best_params_

    # Decision Tree
    dt_params = {'max_depth': [3, 5, 10, None]}
    dt = GridSearchCV(DecisionTreeClassifier(random_state=17), dt_params, scoring='f1')
    dt.fit(x_train, y_train)
    dt_best = dt.best_estimator_
    results["DecisionTree"] = evaluate_model(dt_best, x_val, y_val)
    results["DecisionTree"]["params"] = dt.best_params_

    # Random Forest
    rf_params = {'max_depth': [3, 5, 10, None]}
    rf = GridSearchCV(RandomForestClassifier(random_state=17), rf_params, scoring='f1')
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
