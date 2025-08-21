import pandas as pd
import numpy as np
import os
from src.utils.logger import logger

def cleaning_data(df:pd.DataFrame):
    try: 
        logger.info('Cleaning Data ...')
        df = df.drop('fnlwgt',axis=1)
        df = df.replace('?',np.nan)

        nan = df.isnull().mean()*100
        nan_cols = nan[nan>0].index.tolist()
        def impute_nan(series):
            return series.fillna(series.value_counts().index[0])

        df[nan_cols] = df[nan_cols].apply(impute_nan)

        age_0 = df[df['income']=='<=50K'].age
        age_1 = df[df['income']=='>50K'].age

        def detect_outliers(lt):
            iqr= lt.quantile(0.75)-lt.quantile(0.25)
            ub = lt.quantile(0.75)+1.5*iqr
            lb=lt.quantile(0.25)-1.5*iqr
            upper_array = lt.loc[lt > ub]
            lower_array = lt.loc[lt < lb]
            return upper_array,lower_array

        age_0_upper,age_0_lower = detect_outliers(age_0)
        age_1_upper,age_1_lower = detect_outliers(age_1)
        df = df.drop(age_0_upper.index)
        df = df.drop(age_0_lower.index)
        df = df.drop(age_1_upper.index)
        df = df.drop(age_1_lower.index)

        hours_0 = df.loc[df['income']=='<=50K','hours-per-week']
        hours_1 = df.loc[df['income']=='>50K','hours-per-week']
        hours_1_upper,hours_1_lower = detect_outliers(hours_1)
        hours_0_upper,hours_0_lower = detect_outliers(hours_0)
        df = df.drop(hours_0_upper.index)
        df = df.drop(hours_0_lower.index)
        df = df.drop(hours_1_upper.index)
        df = df.drop(hours_1_lower.index)

        logger.success('Data cleaning Completed')
        logger.info(f'Data Shape after preprocessing : {df.shape}')
        return df
    
    except KeyError as e:
        logger.error(f'Missing column in dataframe : {e}')
        raise
    
    except Exception as e:
        logger.error(f'An unexpected error occured during preprocessing : {e}') 
        raise

def save_cleaned_data(df:pd.DataFrame,destination_path:str):
    try:
        cleaned_data_path = os.path.join(destination_path,'cleaned')
        os.makedirs(cleaned_data_path,exist_ok=True)
        logger.info(f'Saving cleaned Data to {cleaned_data_path}')
        df.to_csv(os.path.join(cleaned_data_path,'cleaned_data.csv'),index=False)
        logger.success(f'Raw Processed data saved to {cleaned_data_path}')

    except Exception as e:
        logger.error(f'An unexpected error occured while saving cleaned data : {e}')
        raise
