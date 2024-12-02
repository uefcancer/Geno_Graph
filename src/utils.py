import os
import glob
from tqdm import tqdm
import pandas as pd
from sklearn.impute import SimpleImputer
import time
import numpy as np
from scipy.stats import chi2_contingency
from sklearn.preprocessing import MinMaxScaler
from imblearn.over_sampling import ADASYN, RandomOverSampler, BorderlineSMOTE, SMOTE, SMOTEN, KMeansSMOTE, SVMSMOTE
from imblearn.combine import SMOTEENN, SMOTETomek

"""
def create_genotype_dataframe(data_path, file_id, raw=None, labels_data_path):
   
    #Create a dataset by merging multiple data frames based on 'SampleID' column.
    #Args:
    #    data_path (str): Path to the data files.
    #    imputed (bool): Flag indicating whether to process imputed data.
    #    raw (bool): Flag indicating whether to process raw data.
    #    file_id (str): Identifier for the specific files to be processed.
    #    labels_data (str): Path to the labels data file.
    #Returns:
    #    df (DataFrame): Merged data frame containing the combined dataset.
    
    if not raw:
        imputed_data_files = glob.glob(os.path.join(data_path, 'processed_imputed', file_id, '*.csv'))
        print('\033[91m[INFO]\033[0m Number of imputed files uploaded: {}'.format(len(imputed_data_files))) #red color info
 
        # Create an empty list to store the data frames
        dfs = []

        # Read each imputed data file into a data frame
        for df_path in tqdm(imputed_data_files):
            df = pd.read_csv(df_path)
            df.rename(columns={"Unnamed: 0": "SampleID"}, inplace=True)
            dfs.append(df)
        print('\033[92m[INFO]\033[0m Combining all the sub files into single dataframe...')
        # Merge the imputed data frames based on the 'SampleID' column
        combined_df = dfs[0]  # Initialize the combined dataframe with the first dataframe
        for df in dfs[1:]:
            combined_df = pd.merge(combined_df, df, on='SampleID', how='outer')

    if raw:
        raw_data_files = glob.glob(os.path.join(data_path, 'processed_raw', file_id, '*.csv'))
        print('\033[91m[INFO]\033[0m Number of raw files uploaded: {}'.format(len(raw_data_files)))

        # Create an empty list to store the data frames
        dfs = []

        # Read each raw data file into a data frame
        for df_path in tqdm(raw_data_files):
            df = pd.read_csv(df_path)
            df.rename(columns={"Unnamed: 0": "SampleID"}, inplace=True)
            dfs.append(df)

        print('\033[92m[INFO]\033[0m Combining all the sub files into single dataframe...')
        # Concatenate the raw data frames into a single data frame
        combined_df = pd.concat(dfs, sort=False)
        # Impute missing values using SimpleImputer
        imputer = SimpleImputer(strategy='median')  # You can change the strategy as needed
        combined_df.iloc[:, 1:] = imputer.fit_transform(combined_df.iloc[:, 1:])  # Assuming first column is 'SampleID'

        combined_df.drop_duplicates(inplace=True)
        combined_df.reset_index(drop=True, inplace=True)
    
    
    print('\033[91m[INFO]\033[0m Loading labels dataframe....')
    labels_df = pd.read_csv(labels_data_path)
    labels_df = labels_df.rename(columns={'PatientID': 'SampleID'})

    # Merge the combined data frame with the labels data frame based on 'SampleID'
    df = combined_df.merge(labels_df, on='SampleID')
    print('\033[92m[INFO]\033[0m Genotype dataframe created successfully....')

    return combined_df
"""


def variants_filter(gt_df, labels_path, gwas_threshold=5e-8, n_significant_variants=100):
    # Load the labels dataframe
    labels_df = pd.read_csv(labels_path)
    labels_df = labels_df.rename(columns={'PatientID': 'SampleID'})
    # Merge the genotype variants dataframe with labels data
    print('\033[92m[INFO]\033[0m Combining genotype dataframe with the labels data...')
    df = gt_df.merge(labels_df, on='SampleID')

    X = df.drop(columns=['SampleID', 'CaseControl']).values
    y = df['CaseControl'].values
    feature_names = df.drop(columns=['SampleID', 'CaseControl']).columns.tolist()

    scaler = MinMaxScaler()
    X = scaler.fit_transform(X)
    print('\033[91m[INFO]\033[0m Computing the chi-square to filter GWAS significant variants...')
    start_time = time.time()  # Record start time
    p_values = []
    for feature in X.T:
        contingency_table = pd.crosstab(feature, y)
        chi2, p, _, _ = chi2_contingency(contingency_table)
        p_values.append(p)
    
    end_time = time.time()  # Record end time
    chi_square_computation_time = end_time - start_time
    print(f'\033[92m[INFO]\033[0m Chi-square computation done in {chi_square_computation_time:.2f} seconds.')

    negative_log10_p_values = [-np.log10(p) for p in p_values]
    # Filter variants based on the gwas_threshold
    significant_variants = [feature_names[i] for i, p_value in enumerate(negative_log10_p_values) if p_value <= -np.log10(gwas_threshold)]
    # Sort the filtered variants based on negative_log10_p_values in increasing order
    sorted_variants = sorted(significant_variants, key=lambda x: negative_log10_p_values[feature_names.index(x)])
    # Take the top n_significant_variants from the sorted list
    top_n_significant_variants = sorted_variants[:n_significant_variants]
    return top_n_significant_variants

def class_balancing(X, y, type, method, random_state=42):
    """
    Apply class imbalance handling techniques to balance the dataset.

    Parameters:
        X (array-like): The feature matrix.
        y (array-like): The target variable.
        type (str): The class balance type ('over_sampling' or 'combined').
        method (str): The imbalance handling method to be applied.
        random_state (int, RandomState instance or None, optional): Controls the random seed for reproducibility.
                                                                   Defaults to 42.

    Returns:
        X_sampled (array-like): The balanced feature matrix.
        y_sampled (array-like): The balanced target variable.
    """

    if type == 'over_sampling':
        if method == 'ADASYN':
            oversampler = ADASYN(random_state=random_state)
        elif method == 'RandomOverSampler':
            oversampler = RandomOverSampler(random_state=random_state)
        elif method == 'BorderlineSMOTE':
            oversampler = BorderlineSMOTE(random_state=random_state)
        elif method == 'SMOTE':
            oversampler = SMOTE(random_state=random_state)
        elif method == 'SMOTEN':
            oversampler = SMOTEN(random_state=random_state)
        elif method == 'KMeansSMOTE':
            oversampler = KMeansSMOTE(random_state=random_state)
        elif method == 'SVMSMOTE':
            oversampler = SVMSMOTE(random_state=random_state)
        else:
            raise ValueError('Invalid over-sampling method specified.')

        X_sampled, y_sampled = oversampler.fit_resample(X, y)

    elif type == 'combined':
        if method == 'SMOTEENN':
            combiner = SMOTEENN(random_state=random_state)
        elif method == 'SMOTETomek':
            combiner = SMOTETomek(random_state=random_state)
        else:
            raise ValueError('Invalid combination method specified.')

        X_sampled, y_sampled = combiner.fit_resample(X, y)

    else:
        raise ValueError('Invalid type specified.')

    return X_sampled, y_sampled


"""
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
import numpy as np

X = raw_df.drop(columns=['SampleID', 'CaseControl']).values
scaler = StandardScaler()
X = scaler.fit_transform(X)  # Scaling the X values
y = raw_df['CaseControl'].values
feature_names = raw_df.columns[2:].tolist()

# Function to normalize feature importance
def normalize_importance(importance):
    return importance / np.sum(importance)

xgb_model = xgb.XGBClassifier().fit(X,y)
xgb_importance = normalize_importance(xgb_model.feature_importances_)

# Display feature importance for each model and aggregated evidence
feature_importances = pd.DataFrame({
    'Feature': feature_names,
    'XGBoost': xgb_importance,
})

# Selecting top features based on aggregated evidence
num_top_features = 200
top_features = list(feature_importances.nlargest(num_top_features, 'XGBoost')['Feature'])

"""
