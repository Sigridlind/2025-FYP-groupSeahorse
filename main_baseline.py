import sys
import os

import numpy as np
import pandas as pd
import util.feature_A
import util.feature_B
import util.feature_C
import util.classifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
from util.img_util import readImageFile, saveImageFile
from util.inpaint_util import removeHair
from imblearn.over_sampling import SMOTE


def main(csv_path, mask_path, img_path, save_path):
    # load dataset CSV file
    df = pd.read_csv(csv_path)
    
    feat_A_values = []
    feat_B_values = []
    feat_C_values = []

    for img_id in df["img_id"]:
        mask_filename = img_id.replace(".png", "_mask.png")
        full_mask_path = os.path.join(mask_path, mask_filename)
        mask = util.feature_A.preprocess_image(full_mask_path)
        full_lesion_path = os.path.join(img_path, img_id)

        try:
            # Feature A
            asymmetry_score = util.feature_A.mean_asymmetry(mask)

            # Feature B
            border_score = util.feature_B.compactness_score(full_mask_path)

            # Feature C
            color_score = util.feature_C.color_score(full_lesion_path, full_mask_path)

        except Exception as e:
            print(f"Error with {img_id}: {e}")
            asymmetry_score = np.nan
            border_score = np.nan
            color_score = np.nan

        feat_A_values.append(asymmetry_score)
        feat_B_values.append(border_score)
        feat_C_values.append(color_score)

    df["feat_A"] = feat_A_values
    df["feat_B"] = feat_B_values
    df["feat_C"] = feat_C_values
    df["feat_D"] = int(round((df["diameter_1"] + df["diameter_2"]) / 2, 0))
    df["feat_E"] = df["grew"] == "True" or df["changed"] == "True"
    df['label'] = df["diagnostic"] == "MEL"
     
    # Gem udvalgt data
    df_out = df[["img_id", "feat_A", "feat_B","feat_C", "feat_D", "feat_E", "label"]]
    df_out.to_csv(save_path, index=False)

def classification(feature_csv_path, results_path):
    df = pd.read_csv(feature_csv_path)
    
    # select only the baseline features.
    baseline_feats = [col for col in df.columns if col.startswith("feat_")]
    x_all = df[baseline_feats]
    y_all = df["label"]

    # split the dataset into training and testing sets.
    x_train, x_temp, y_train, y_temp = train_test_split(x_all, y_all, test_size=0.3, random_state=17)

    # split testing sets into 15% val, 15% test
    x_val, x_test, y_val, y_test = train_test_split(x_temp, y_temp, test_size=0.5, random_state=17)

    # Apply SMOTE to training set
    smote = SMOTE(random_state=17)
    x_train_res, y_train_res = smote.fit_resample(x_train, y_train)
    
    # Tune models using validation set
    tuned_results = util.classifier.tune_models(x_train_res, y_train_res, x_val, y_val)
    
    final_results = {}
    for name, res in tuned_results.items():
        print(f"\n{name} Validation F1: {res['f1']:.3f}, AUC: {res['auc']:.3f}, Accuracy: {res['acc']:.3f}, Precision: {res['precision']:.3f}, Recall: {res['recall']:.3f}")
        
    best_model_name = max(tuned_results, key=lambda name: tuned_results[name]["f1"])
    best_model = tuned_results[best_model_name]["model"]
    print(f"Best model is (Based on validation performance): {best_model_name}")
    
    final_result = util.classifier.evaluate_model(best_model, x_test, y_test)
    print(f"\n{best_model_name} Test F1: {final_result['f1']:.3f}, AUC: {final_result['auc']:.3f}, Accuracy: {final_result['acc']:.3f}, Precision: {final_result['precision']:.3f}, Recall: {final_result['recall']:.3f}")
    print("Confusion Matrix:\n", final_result["cm"])

    # write test results to CSV.
    df_out = df.loc[x_test.index, ["img_id"]].copy()
    df_out["true_label"] = y_test.values
    df_out["predicted_label"] = final_result["y_pred"]
    df_out["melanoma_probability"] = final_result["y_prob"]
    df_out.to_csv(results_path, index=False)


if __name__ == "__main__":
    csv_path = "./dataset.csv"
    # add mask_path
    # add img_path
    save_path = "./result/result_baseline.csv"
    

    main(csv_path, mask_path, img_path, save_path)
