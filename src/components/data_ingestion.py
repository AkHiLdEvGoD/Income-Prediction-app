import pandas as pd
import numpy as np
import os
from src.utils.logger import logger
from src.utils.config import DATA_DIR,DATA_URL

def load_data(data_url:str):
    logger.info(f'Reading data from {data_url}')
    try:
        df = pd.read_csv(data_url)
        logger.info(f'Data loaded and saved as Dataframe. Dataset shape : {df.shape}')
        return df
    except pd.errors.ParserError as e:
        logger.error(f'Failed to parse the CSV file : {e}')
        raise
    except Exception as e:
        logger.error(f'An unexpected error occured while loading data {e}')
        raise


def save_data(df:pd.DataFrame,destination_path):
    try:
        raw_data_path=os.path.join(destination_path,'raw')
        os.makedirs(raw_data_path,exist_ok=True)
        logger.info(f'Saving Data to {raw_data_path}')
        df.to_csv(os.path.join(raw_data_path,'raw_data.csv'),index=False)
        logger.success(f'Raw Processed data saved to {raw_data_path}')

    except Exception as e:
        logger.error(f'An unexpected error occured while saving the data : {e}')
        raise

def main():
    try:
        df = load_data(DATA_URL)
        save_data(df,destination_path=DATA_DIR)
        logger.success('Data Ingestion Completed')
    
    except Exception as e:
        logger.error(f'Failed to complete data ingestion process : {e}')
        print('error',e)

if __name__ == '__main__':
    main()