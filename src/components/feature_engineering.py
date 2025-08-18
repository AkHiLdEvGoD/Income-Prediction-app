import pandas as pd
import os
from src.utils.config import CLEANED_DATA_PATH,DATA_DIR
from src.utils.logger import logger

def feature_engineering(df:pd.DataFrame):
    try : 
        logger.info('Starting Feature Engineering ...')

        df["work_type"] = df["workclass"].replace({
        "Self-emp-not-inc": "Self-Employed",
        "Self-emp-inc": "Self-Employed",
        "Local-gov" : "Gov_employee",
        "State_gov" : "Gov_employee",
        "Without-pay": "Other",
        "Never-worked": "Other"
        })

        df["is_married"] = df["marital-status"].apply(lambda x: 1 if "Married" in x else 0)

        def group_country(country):
            if country in ['United-States','Mexico','Canada']:
                return 'North America'
            elif country in ['India', 'China', 'Philippines', 'Japan','Taiwan','Iran']:
                return 'Asia'
            elif country in ['Germany', 'England', 'France', 'Italy','Poland']:
                return 'Europe'
            else:
                return 'Other'

        df['region'] = df['native-country'].apply(group_country)
        df["race_gender"] = df["race"] + "_" + df["gender"]

        df["has_capital_gain"] = (df["capital-gain"] > 0).astype(int)
        df["has_capital_loss"] = (df["capital-loss"] > 0).astype(int)
        
        df = df.drop(columns=['race','gender','native-country','marital-status','workclass'],axis=1)
        df = df.drop('education',axis=1)

        logger.info(f'Old features removed. New features generated. Shape of df {df.shape}')
        return df
    
    except Exception as e:
        logger.error(f'An unexpected error occured during preprocessing : {e}') 
        raise

def save_df(df,destination_path):
    try:
        featured_data_path = os.path.join(destination_path,'featured')
        os.makedirs(featured_data_path,exist_ok=True)
        logger.info(f'Saving cleaned Data to {featured_data_path}')
        df.to_csv(os.path.join(featured_data_path,'featured_data.csv'),index=False)
        logger.success(f'Raw Processed data saved to {featured_data_path}')

    except Exception as e:
        logger.error(f'An unexpected error occured while saving cleaned data : {e}')
        raise

def main():
    try:
        cleaned_df = pd.read_csv(CLEANED_DATA_PATH)
        featured_df = feature_engineering(cleaned_df)
        save_df(featured_df,DATA_DIR)
        logger.success('Feature Engineering Completed')

    except Exception as e:
        logger.error(f'Failed to complete feature engineerirng process : {e}')

if __name__ == '__main__':
    main()