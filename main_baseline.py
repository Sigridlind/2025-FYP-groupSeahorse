import sys
import os
import numpy as np
import pandas as pd
import util.feature_A
import util.feature_B
import util.feature_C
import util.classifier
from skimage.io import imread
from skimage.color import rgb2gray
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
from util.img_util import readImageFile, saveImageFile
from util.inpaint_util import removeHair
from imblearn.over_sampling import SMOTE

import os
import pandas as pd
import numpy as np
from skimage.io import imread
from skimage.color import rgb2gray

def remove_rows_csv(csv_path, mask_dir, min_lesion_pixels=10, binarization_threshold=0.05):
    """
    Removes rows where:
    - the corresponding mask does not exist
    - or the mask has fewer than `min_lesion_pixels` with intensity above `binarization_threshold`

    Parameters:
        csv_path (str): Path to your local metadata CSV
        mask_dir (str): Path to the local folder with lesion masks
        min_lesion_pixels (int): Minimum lesion pixels required to keep the row
        binarization_threshold (float): Threshold to binarize the mask (default: 0.05)

    Returns:
        None: Updates the CSV in-place
    """
    df = pd.read_csv(csv_path)
    valid_rows = []

    for idx, row in df.iterrows():
        img_id = row["img_id"]
        mask_name = img_id.replace(".png", "_mask.png")
        mask_path = os.path.normpath(os.path.join(mask_dir, mask_name))

        if not os.path.exists(mask_path):
            print(f"Missing mask: {img_id}")
            continue

        try:
            mask = imread(mask_path)

            if mask.ndim == 3:
                mask = rgb2gray(mask)

            binary_mask = mask > binarization_threshold
            lesion_pixels = np.sum(binary_mask)

            if lesion_pixels < min_lesion_pixels:
                print(f"Too few lesion pixels ({lesion_pixels:.0f}) in: {img_id}")
                continue

        except Exception as e:
            print(f"Error reading {img_id}: {e}")
            continue

        valid_rows.append(row)

    # Save filtered CSV
    filtered_df = pd.DataFrame(valid_rows)
    print(f"Kept {len(filtered_df)} rows out of {len(df)}.")
    filtered_df.to_csv(csv_path, index=False)
    

def main(csv_path, mask_path, img_path, save_path):
    # load dataset CSV file
    df = pd.read_csv(csv_path)
    
    feat_A_values = []
    feat_B_values = []
    feat_C_values = []

    for img_id in df["img_id"]:
        mask_filename = img_id.replace(".png", "_mask.png")
        full_mask_path = os.path.normpath(os.path.join(mask_path, mask_filename))
        full_lesion_path = os.path.normpath(os.path.join(img_path, img_id))
        
        try:
            # Feature A
            asymmetry_score = util.feature_A.asymmetry_score(full_mask_path)

            # Feature B
            border_score = util.feature_B.border_irregularity(full_mask_path)

            # Feature C
            color_score = util.feature_C.color_score(full_lesion_path, full_mask_path)

        except Exception as e:
            print(f"Error processing {img_id}: {e}")
            asymmetry_score = np.nan
            border_score = np.nan
            color_score = np.nan

        feat_A_values.append(asymmetry_score)
        feat_B_values.append(border_score)
        feat_C_values.append(color_score)
        if (len(feat_C_values) % 100) == 0:
            print(f"Processed {len(feat_C_values)} images...")

    df["feat_A"] = feat_A_values
    df["feat_B"] = feat_B_values
    df["feat_C"] = feat_C_values
    df["feat_D"] = ((df["diameter_1"] + df["diameter_2"]) / 2)
    df["feat_E"] = (df["grew"] == "True") | (df["changed"] == "True")
    df['label'] = df["diagnostic"] == "MEL"
    
    # Gem udvalgt data
    df_out = df[["img_id", "feat_A", "feat_B","feat_C", "feat_D", "feat_E", "label"]]
    df_out.to_csv(save_path, index=False)
    

def classification(feature_csv_path, results_path):
    df = pd.read_csv(feature_csv_path)
    
    # select only the baseline features.
    
    baseline_feats = ["feat_A", "feat_B","feat_C"] # [col for col in df.columns if col.startswith("feat_")]
    
    df = df.dropna(subset=baseline_feats)
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
        print(f"{name} Best Hyperparameters: {res['params']}")
            
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
    csv_path = "C:/Users/Lenovo/OneDrive - ITU/Uni/2. Semester/Projects DS/2025-FYP-groupSeahorse/metadata.csv"
    mask_path = "C:/Users/Lenovo/OneDrive - ITU/Uni/2. Semester/Projects DS/Project/lesion_masks"
    img_path = "C:/Users/Lenovo/OneDrive - ITU/Uni/2. Semester/Projects DS/Project/EDA/imgs"
    save_path = "C:/Users/Lenovo/OneDrive - ITU/Uni/2. Semester/Projects DS/2025-FYP-groupSeahorse/result/result.csv"
    
    # remove_rows_csv(csv_path, mask_path)
    # main(csv_path, mask_path, img_path, save_path)
    classification("result/result.csv", "result/result2.csv")

