import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from src.components.data_cleaning import cleaning_data,save_cleaned_data
from src.utils.logger import logger
from src.utils.config import RAW_DATA_PATH,DATA_DIR
import pandas as pd

def main():
    try:
        raw_df = pd.read_csv(RAW_DATA_PATH)
        cleaned_df = cleaning_data(raw_df)
        save_cleaned_data(cleaned_df,DATA_DIR)
        logger.success('Data Cleaning Completed')
    
    except Exception as e:
        logger.error(f'Failed to complete data cleaning process : {e}')

if __name__ == '__main__':
    main()
