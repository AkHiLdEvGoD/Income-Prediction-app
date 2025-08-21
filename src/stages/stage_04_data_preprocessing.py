import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from src.components.data_preprocessing import preprocess,save_preprocessed_data,load_params
from src.utils.logger import logger
from src.utils.config import FEATURED_DATA_PATH,DATA_DIR
import pandas as pd



def main():
    try:
        params = load_params(params_path='./params.yaml')
        test_size = params['data_preprocessing']['test_size']
        df = pd.read_csv(FEATURED_DATA_PATH)
        X_train,X_test,y_train,y_test = preprocess(df,DATA_DIR,test_size)
        save_preprocessed_data(X_train,X_test,y_train,y_test,destination_path=DATA_DIR)
        logger.success('Data Preprocessing Completed')

    except Exception as e:
        logger.error(f'Failed to complete data Preprocessing process : {e}')

if __name__ == '__main__':
    main()
