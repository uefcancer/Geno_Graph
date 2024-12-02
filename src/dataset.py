"""
PrepareDataset Class for Data Preprocessing

Developer: Raju Gudhe (raju.gudhe@uef.fi)
Copyright: This code is for research purposes only.

Description:
This module provides a class to load, preprocess, handle class imbalance, and split
the dataset for further use in machine learning models.
"""

import os
import pickle
import pandas as pd
from datetime import datetime
import yaml
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler,LabelEncoder
from utils import class_balancing
import warnings
# Ignore FutureWarnings
warnings.simplefilter(action='ignore', category=FutureWarning)


class PrepareDataset:
    def __init__(self, config):
        """
        Initialize the dataset preparation class with configuration parameters.
        """
        self.config = config


    def load_data(self):
        """
        Load the raw genotype data and convert variant identifiers to reference SNP identifiers.
        """
        print(f'\033[91m[INFO]\033[0m Preparing Dataset and Pre-procesing ')

        self.df = pd.read_pickle(os.path.join(self.config['dataset']['path'], self.config['dataset']['file_name']))
        #self.df.set_index('SampleID', inplace=True)

        # Load the rsIds mapping dictionary from the saved file
        with open(r'data\Biobank\raw\raw_snp_rsid_mapping.pkl', 'rb') as file:
            snps_to_rsid = pickle.load(file)
        self.df = self.df.rename(columns=snps_to_rsid)

    def process_data(self):
        """
        Preprocess the data by normalizing and encoding labels.
        """
        print(f'\033[92m[{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}]\033[0m Data loaded successfully.')
        self.raw_df = self.df.drop(columns=['CaseControl'])

        self.snps = self.raw_df.columns.to_list()
 
        print(f'\033[92m[{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}]\033[0m Data preprocessing: Normalizing the data...')
        #scaler = MinMaxScaler()
        scaler = StandardScaler()
        self.X = self.raw_df.values
        self.X = scaler.fit_transform(self.X)
       
        y = self.df['CaseControl'].values
        label_encoder = LabelEncoder()
        self.y = label_encoder.fit_transform(y)
        self.y = self.y.astype(np.float64)
    
    def handle_imbalance(self):
        """
        Handle class imbalance using the specified class balancing method.
        """
        print(f'\033[92m[{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}]\033[0m Data preprocessing: Handling class imbalance...')
        self.X, self.y = class_balancing(self.X, self.y, 
                                         type=self.config['class_balancing']['type'], 
                                         method=self.config['class_balancing']['method'], 
                                         random_state=42)
    def split_data(self):
        """
        Split the data into training and test sets.
        """
        print(f'\033[92m[{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}]\033[0m Splitting data into train and test sets...')
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y, test_size=self.config['split_percentage'], random_state=self.config['random_seed'], stratify=self.y
        )
    
    def prepare(self):
        """
        Execute the data preparation pipeline.
        """
        self.load_data()
        self.process_data()
        self.handle_imbalance()
        self.split_data()
        return self.snps, self.X_train, self.X_test, self.y_train, self.y_test
