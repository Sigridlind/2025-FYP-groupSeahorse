import pandas as pd
import numpy as np
import util.classifier
import util.feature_extraction
import util.clean_data
    
def main_extended(metadata_path, mask_dir, img_dir, results_extended_path):

    # load the metadata
    df = pd.read_csv(metadata_path)

    # clean the data
    df = util.clean_data.clean_data(df, mask_dir)

    # extract features
    df = util.feature_extraction.feature_extraction(df, mask_dir, img_dir)

    # save to csv
    df.to_csv("dataset.csv")
    
    # fill NaN with column means so KNN works
    df_numeric = df.select_dtypes(include=[np.number])
    df[df_numeric.columns] = df_numeric.fillna(df_numeric.mean())
    
    # use classifiers and save results
    util.classifier.classification(df, results_extended_path, baseline= False)