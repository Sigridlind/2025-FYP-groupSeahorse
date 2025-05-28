import pandas as pd
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

    # fill NaN with column means so KNN works
    df = df.fillna(df.mean())

    # 
    util.classifier.classification(df, results_extended_path, baseline= False)