import sys
import os

import numpy as np
import pandas as pd
import util.feature_A
import util.feature_B
import util.feature_C
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
from util.img_util import readImageFile, saveImageFile
from util.inpaint_util import removeHair


def main(csv_path, mask_path, img_path, save_path):
    # load dataset CSV file
    df = pd.read_csv(csv_path)
    
    feat_A_values = []
    feat_B_values = []
    feat_C_values = []

    for img_id in df["img_id"]:
        mask_filename = img_id.replace(".png", "_mask.png")
        full_mask_path = os.path.join(mask_path, mask_filename)
        full_lesion_path = os.path.join(img_path, img_id)

        try:
            # Feature A
            asymmetry_score = util.feature_A.mean_asymmetry(full_mask_path)

            # Feature B
            border_score = util.feature_B.compactness_score(full_mask_path)

            # Feature C
            color_score = util.feature_C.color_score(full_lesion_path)

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

    # # Gem udvalgt data
    # df_out = df[["img_id", "feat_A", "feat_B", "label"]]
    # df_out.to_csv(save_path, index=False)
    # print(f"Features gemt i: {save_path}")

    df['label'] = df["diagnostic"] == "MEL"

    # select only the baseline features.
    baseline_feats = [col for col in df.columns if col.startswith("feat_")]
    x_all = df[baseline_feats]
    y_all = df["label"]

    # split the dataset into training and testing sets.
    x_train, x_temp, y_train, y_temp = train_test_split(x_all, y_all, test_size=0.3, random_state=17)

    # split testing sets into 15% val, 15% test
    x_val, x_test, y_val, y_test = train_test_split(x_temp, y_temp, test_size=0.5, random_state=17)

    # # train the classifier (using logistic regression as an example)
    # clf = LogisticRegression(max_iter=1000, verbose=1)
    # clf.fit(x_train, y_train)

    # # test the trained classifier
    # y_pred = clf.predict(x_test)
    # acc = accuracy_score(y_test, y_pred)
    # cm = confusion_matrix(y_test, y_pred)
    # print("Test Accuracy:", acc)
    # print("Confusion Matrix:\n", cm)

    # # write test results to CSV.
    # result_df = data.loc[X_test.index, ["filename"]].copy()
    # result_df['true_label'] = y_test.values
    # result_df['predicted_label'] = y_pred
    # result_df.to_csv(save_path, index=False)
    # print("Results saved to:", save_path)


if __name__ == "__main__":
    csv_path = "./dataset.csv"
    # add mask_path
    # add img_path
    save_path = "./result/result_baseline.csv"
    

    main(csv_path, mask_path, img_path, save_path)
