import os
import pandas as pd
import numpy as np
import util.feature_A
import util.feature_B
import util.feature_C

def feature_extraction(df, mask_dir, img_dir):
    
    feat_A_values = []
    feat_B_values = []
    feat_C_values = []

    for img_id in df["img_id"]:
        mask_filename = img_id.replace(".png", "_mask.png")
        mask_path = os.path.normpath(os.path.join(mask_dir, mask_filename))
        lesion_path = os.path.normpath(os.path.join(img_dir, img_id))
        
        try:
            # Feature A
            asymmetry_score = util.feature_A.asymmetry_score(mask_path)

            # Feature B
            border_score = util.feature_B.border_irregularity(mask_path)

            # Feature C
            color_score = util.feature_C.color_score(lesion_path, mask_path)

        except Exception as e:
            print(f"Error processing {img_id}: {e}")
            asymmetry_score = np.nan
            border_score = np.nan
            color_score = np.nan

        feat_A_values.append(asymmetry_score)
        feat_B_values.append(border_score)
        feat_C_values.append(color_score)
        # if (len(feat_C_values) % 100) == 0:
        #     print(f"Processed {len(feat_C_values)} images...")

    df["feat_A"] = feat_A_values
    df["feat_B"] = feat_B_values
    df["feat_C"] = feat_C_values
    df["feat_D"] = ((df["diameter_1"] + df["diameter_2"]) / 2)
    df["feat_E"] = (df["grew"] == "True") | (df["changed"] == "True")
    df['label'] = df["diagnostic"] == "MEL"
    
    return df[["img_id", "feat_A", "feat_B", "feat_C", "feat_D", "feat_E", "label"]]
    