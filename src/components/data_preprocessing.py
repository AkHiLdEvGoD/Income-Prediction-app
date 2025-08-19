import pandas as pd
from src.utils.logger import logger
from src.utils.config import FEATURED_DATA_PATH,DATA_DIR
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder,StandardScaler
from sklearn.compose import ColumnTransformer
from imblearn.over_sampling import SMOTE
import joblib
import os

def preprocess(df:pd.DataFrame,save_dir):
    try:
        logger.info('Starting Preprocessing data ...')
        df['income'] = df['income'].map({'>50K':1, '<=50K':0})
        X = df.drop('income',axis=1)
        y = df.loc[:,'income']
        X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.15,random_state=42)
        logger.info('Data Splitted into training and test set')

        df_dtypes = X_train.dtypes
        obj_cols = df_dtypes[df_dtypes == 'object'].index.tolist()
        num_cols = df_dtypes[df_dtypes != 'object'].index.tolist()

        oe = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
        preprocessor = ColumnTransformer(
            transformers=[
                ('scaler',StandardScaler(),num_cols),
                ('cat',oe,obj_cols),
            ]
        )

        X_train_processed = preprocessor.fit_transform(X_train)
        X_test_processed = preprocessor.transform(X_test)
        logger.info('Scaling and Encoding done')

        smote = SMOTE(random_state=42)
        X_train_resampled, y_train_resampled = smote.fit_resample(X_train_processed, y_train)   
        artifacts_path = os.path.join(save_dir,'artifacts')
        os.makedirs(artifacts_path,exist_ok=True)
        joblib.dump(preprocessor,os.path.join(artifacts_path,'preprocessor.pkl'))  
        logger.info(f'Preprocessor saved to path : {os.path.join(artifacts_path,"preprocessor.pkl")}')
      
        logger.success('Preprocessing Done')
        return X_train_resampled, X_test_processed, y_train_resampled, y_test
    
    except KeyError as e:
        logger.error(f'Missing column in dataframe : {e}')
        raise

    except Exception as e:
        logger.error(f'An unexpected error occured while saving cleaned data : {e}')
        raise

def save_preprocessed_data(X_train,X_test,y_train,y_test,destination_path):
    try:
        X_train_df = pd.DataFrame(X_train).reset_index(drop=True)
        y_train_series = pd.Series(y_train, name='target').reset_index(drop=True)
        train_df = pd.concat([X_train_df, y_train_series], axis=1)

        X_test_df = pd.DataFrame(X_test).reset_index(drop=True)
        y_test_series = pd.Series(y_test, name='target').reset_index(drop=True)
        test_df = pd.concat([X_test_df, y_test_series], axis=1)

        processed_data_path = os.path.join(destination_path,'processed')
        os.makedirs(processed_data_path,exist_ok=True)
        train_df.to_csv(os.path.join(processed_data_path,'train.csv'),index=False)
        test_df.to_csv(os.path.join(processed_data_path,'test.csv'),index=False)
        logger.success(f'Train and test data saved to path : {processed_data_path}')
        logger.debug(f"Train shape: {train_df.shape} | Test shape: {test_df.shape}")

    except Exception as e:
        logger.error(f'Unexpected error occured while saving preprocessed data : {e}')
        raise

def main():
    try:
        df = pd.read_csv(FEATURED_DATA_PATH)
        X_train,X_test,y_train,y_test = preprocess(df,DATA_DIR)
        save_preprocessed_data(X_train,X_test,y_train,y_test,destination_path=DATA_DIR)
        logger.success('Data Preprocessing Completed')

    except Exception as e:
        logger.error(f'Failed to complete data Preprocessing process : {e}')

if __name__ == '__main__':
    main()