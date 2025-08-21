import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from src.components.feature_engineering import feature_engineering,save_df
from src.utils.config import CLEANED_DATA_PATH,DATA_DIR
from src.utils.logger import logger
import pandas as pd

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
