import numpy as np
import pandas as pd
import util.classifier
import util.feature_extraction
import util.clean_data
from util.classifier import plot_confusion_matrix

def main_baseline(metadata_path, mask_dir, img_dir, results_baseline_path):

    # load the metadata
    df = pd.read_csv(metadata_path)

    # clean the data
    df = util.clean_data.clean_data(df, mask_dir)

    # extract features
    df = util.feature_extraction.feature_extraction(df, mask_dir, img_dir)
    
    # save features in dataset.csv
    df.to_csv("dataset.csv")

    # fill NaN with column means so KNN works
    df_numeric = df.select_dtypes(include=[np.number])
    df[df_numeric.columns] = df_numeric.fillna(df_numeric.mean())

    # use classifiers and save results
    util.classifier.classification(df, results_baseline_path)