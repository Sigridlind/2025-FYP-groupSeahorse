import numpy as np
import pandas as pd
import util.classifier
import util.feature_extraction
import util.clean_data
    

def main_baseline(metadata_path, mask_dir, img_dir, results_baseline_path):

    # load the metadata
    df = pd.read_csv(metadata_path)

    # clean the data
    df = util.clean_data.clean_data(df, mask_dir)

    # extract features
    df = util.feature_extraction.feature_extraction(df, mask_dir, img_dir)
    df.to_csv("dataset.csv")

    # fill NaN with column means so KNN works
    df_numeric = df.select_dtypes(include=[np.number])
    df[df_numeric.columns] = df_numeric.fillna(df_numeric.mean())

    # classifiers and save results
    util.classifier.classification(df, results_baseline_path)


if __name__ == "__main__":
    csv_path = "C:/Users/Lenovo/OneDrive - ITU/Uni/2. Semester/Projects DS/2025-FYP-groupSeahorse/metadata.csv"
    mask_path = "C:/Users/Lenovo/OneDrive - ITU/Uni/2. Semester/Projects DS/Project/lesion_masks"
    img_path = "C:/Users/Lenovo/OneDrive - ITU/Uni/2. Semester/Projects DS/Project/EDA/imgs"
    save_path = "C:/Users/Lenovo/OneDrive - ITU/Uni/2. Semester/Projects DS/2025-FYP-groupSeahorse/result/result.csv"
    
    # remove_rows_csv(csv_path, mask_path)
    main_baseline(csv_path, mask_path, img_path, "C:/Users/Lenovo/OneDrive - ITU/Uni/2. Semester/Projects DS/2025-FYP-groupSeahorse/result/result_baseline.csv")
    # classification("result/result.csv", "result/result2.csv")

