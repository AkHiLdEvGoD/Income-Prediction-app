import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from src.components.data_ingestion import load_data,save_data
from src.utils.logger import logger
from src.utils.config import DATA_URL,DATA_DIR

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