"""
This code helps compare three types of models (KNN, Decision Tree, Random Forest)
to see which one does the best job at predicting something based on features in a dataset.

It does the following:
- Picks which features to use (simple or extended set)
- Splits the data into parts for training and testing
- Balances the data using SMOTE so both classes are fairly represented
- Tries different model settings to find the best ones
- Checks how well each model works using accuracy, recall, F1, etc.
- Shows charts of model performance
- Saves the final predictions to a CSV file

"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, confusion_matrix, roc_auc_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import KFold, GridSearchCV, train_test_split
from imblearn.over_sampling import SMOTE

def plot_confusion_matrix(cm, class_names=["Not MEL", "MEL"], title="Confusion Matrix"):
    """
    Plots a labeled confusion matrix.

    Parameters:
        cm (array-like): 2x2 confusion matrix.
        class_names (list): Labels for the classes.
        title (str): Title of the plot.
    """
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=class_names, yticklabels=class_names)

    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title(title)
    plt.tight_layout()
    plt.show()


def tune_models(x_train, y_train, x_val, y_val):
    """
    Tunes KNN, Decision Tree, and Random Forest classifiers using GridSearchCV
    and evaluates them on the validation set.
    
    Returns:
        - results: Dict with performance metrics and parameters.
        - grid_searches: Dict with GridSearchCV objects for each model.
    """
    results = {}
    # Creates cross-validation with 5 folds 
    cv = KFold(n_splits=5, shuffle=True, random_state=57) # Apply seeding for cross validation
    
    # KNN
    # Try multiple values for n_neighbors and select best based on recall 
    knn_params = {'n_neighbors': [3, 5, 7, 9]}
    knn = GridSearchCV(KNeighborsClassifier(), param_grid=knn_params, scoring='recall', cv=cv)
    knn.fit(x_train, y_train)
    knn.X_, knn.y_ = x_train, y_train
    knn_best = knn.best_estimator_
    results["KNN"] = evaluate_model(knn_best, x_val, y_val)
    results["KNN"]["params"] = knn.best_params_

    # Decision Tree
    # Tune max_depth and check with which hyperparameter it performs the best 
    dt_params = {'max_depth': [3, 5, 10, None]}     # A dictionary of possible depth values to test in the grid search
    dt = GridSearchCV(DecisionTreeClassifier(random_state=57), dt_params, scoring='recall', cv=cv)
    dt.fit(x_train, y_train)  # It trains and evaluates 4 models across 5 cross-validation folds 
    dt.X_, dt.y_ = x_train, y_train
    dt_best = dt.best_estimator_
    results["DecisionTree"] = evaluate_model(dt_best, x_val, y_val)   # Evaluate the best model on validation set, chooses the one with highest recall store
    results["DecisionTree"]["params"] = dt.best_params_

    # Random Forest
    # Try multiple values for max depth and select best based on recall
    rf_params = {'max_depth': [3, 5, 10, None]}  # A dictionary of possible depth values to test in the grid search
    rf = GridSearchCV(RandomForestClassifier(random_state=57), rf_params, scoring='recall', cv=cv) # using grid search to find the best max_depth
    rf.fit(x_train, y_train)
    rf.X_, rf.y_ = x_train, y_train
    rf_best = rf.best_estimator_
    results["RandomForest"] = evaluate_model(rf_best, x_val, y_val)  # This line evaluates the best model on an unseen validation set 
    results["RandomForest"]["params"] = rf.best_params_     # This stores the best hyperparameters in the results dictionary for reference

    return results, {"KNN": knn, "DecisionTree": dt, "RandomForest": rf}

# Evaluating a classification model on given data using multiple performance metrics
def evaluate_model(model, x, y):
    """
    Evaluates a trained classifier on given data and returns performance metrics.
    
    Returns:
        A dictionary containing accuracy, precision, recall, F1 score, AUC,
        predictions, probabilities, and the confusion matrix.
    """
    y_pred = model.predict(x)
    y_prob = model.predict_proba(x)[:, 1]
    acc = accuracy_score(y, y_pred)
    recall = recall_score(y, y_pred)
    precision = precision_score(y, y_pred)
    f1 = f1_score(y, y_pred)
    auc = roc_auc_score(y, y_prob)
    cm = confusion_matrix(y, y_pred)  # Compute confusion matrix - shows TP, TN, FP, FN 
    return {"model": model,"acc": acc,"recall": recall, "precision": precision,"f1": f1, "auc": auc, "cm": cm, "y_pred": y_pred, "y_prob": y_prob}     # Return a dictionary with all metrics, predictions and probabilities

def classification(df, results_path, baseline= True):
    """
    Main classification pipeline:
        - Select features
        - Split data
        - Balance with SMOTE
        - Train and tune models
        - Evaluate best model
        - Output results and save classification report & predictions
    """
    # select only the baseline features.
    if baseline:
        features = ["feat_A", "feat_B", "feat_C"] # [col for col in df.columns if col.startswith("feat_")]
    else: features = ["feat_A", "feat_B", "feat_C", "feat_D", "feat_E"]
    
    x_all = df[features]
    y_all = df["label"]

    # split the dataset into training and testing sets.
    x_train, x_temp, y_train, y_temp = train_test_split(x_all, y_all, test_size=0.3, random_state=57)

    # split testing sets into 23% val, 23% test
    x_val, x_test, y_val, y_test = train_test_split(x_temp, y_temp, test_size=0.5, random_state=57)

    # Apply SMOTE to training set
    smote = SMOTE(random_state=57)
    x_train_res, y_train_res = smote.fit_resample(x_train, y_train)
    
    # Tune models using validation set
    # tuned_results = tune_models(x_train_res, y_train_res, x_val, y_val)
    tuned_results, grid_searches = tune_models(x_train_res, y_train_res, x_val, y_val)
    
    print(f"\nValidation Performance per Model")
    for name, res in tuned_results.items():
        print(f"\n{name}")
        print("-" * len(name))
        print(f"  F1 Score       : {res['f1']:.3f}")
        print(f"  AUC            : {res['auc']:.3f}")
        print(f"  Accuracy       : {res['acc']:.3f}")
        print(f"  Precision      : {res['precision']:.3f}")
        print(f"  Recall         : {res['recall']:.3f}")
        print(f"  Best Parameters: {res['params']}")

    best_model_name = max(tuned_results, key=lambda name: tuned_results[name]["recall"])
    best_model = tuned_results[best_model_name]["model"]
    print(f"\nBest model based on validation recall: {best_model_name}")

    final_result = evaluate_model(best_model, x_test, y_test)
    print(f"\nTest Performance: {best_model_name} ")
    print(f"  F1 Score   : {final_result['f1']:.3f}")
    print(f"  AUC        : {final_result['auc']:.3f}")
    print(f"  Accuracy   : {final_result['acc']:.3f}")
    print(f"  Precision  : {final_result['precision']:.3f}")
    print(f"  Recall     : {final_result['recall']:.3f}")
    print("  Confusion Matrix:")
    print(final_result["cm"])
    
    
    # Generate classification report as dictionary
    from sklearn.metrics import classification_report
    import pandas as pd

    report_dict = classification_report(
        y_test,
        final_result["y_pred"],
        target_names=["Melanoma", "Non-Melanoma"],
        output_dict=True)

    # Convert to DataFrame
    report_df = pd.DataFrame(report_dict).T
    latex_table = report_df.to_latex(float_format="%.2f")
    with open("classification_report.tex", "w") as f:
        f.write(latex_table)
    
    

    
    # write test results to CSV.
    df_out = df.loc[x_test.index, ["img_id"]].copy()
    df_out["true_label"] = y_test.values
    df_out["predicted_label"] = final_result["y_pred"]
    df_out["melanoma_probability"] = final_result["y_prob"]
    df_out.to_csv(results_path, index=False)
    plot_confusion_matrix(final_result["cm"], title=f"{best_model_name} - Test Confusion Matrix")