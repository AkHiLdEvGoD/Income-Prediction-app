import os
import io
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score,recall_score,f1_score,precision_score,classification_report,confusion_matrix
import joblib
import pandas as pd
from src.utils.logger import logger
import json
import yaml

def load_params(params_path:str):
    try:
        with open(params_path,'r') as f:
            params = yaml.safe_load(f)
        logger.info(f'Parameter retrieved from {params_path}')
        return params
    except FileNotFoundError:
        logger.error('File not found: %s', params_path)
        raise
    except yaml.YAMLError as e:
        logger.error('YAML error: %s', e)
        raise
    except Exception as e:
        logger.error('Unexpected error: %s', e)
        raise

def load_model(model_path):
    try:
        model = joblib.load(model_path)
        logger.info('Model Loaded for evaluation')
        return model
    
    except FileNotFoundError:
        logger.error(f'Model not found at {model_path}')
        raise

    except Exception as e:
        logger.error(f'Unexpected error occured while loading the model {e}')
        raise

def load_data(data_path:str):
    try:
        df = pd.read_csv(data_path)
        logger.info(f'Data loaded from path {data_path}')
        return df
    
    except pd.errors.ParserError as e:
        logger.error(f'Failed to parse the CSV file: {e}')
        raise
    
    except Exception as e:
        logger.error(f'Unexpected error occurred while loading the data: {e}')
        raise

def evaluate_model(model,df:pd.DataFrame):
    try:
        X = df.drop(columns = ['target'])
        y = df['target']

        y_pred = model.predict(X)

        accuracy = accuracy_score(y,y_pred)
        precision = precision_score(y,y_pred)
        recall = recall_score(y,y_pred)
        f1 = f1_score(y,y_pred)

        metric_dict = {
            'accuracy' : accuracy,
            'precision' : precision,
            'recall' : recall,
            'f1_score' : f1
        }
        cm = confusion_matrix(y,y_pred)
        plt.figure(figsize=(6, 4))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
        plt.title("Confusion Matrix")
        plt.xlabel("Predicted")
        plt.ylabel("Actual")

        cm_buf = io.BytesIO()
        plt.savefig(cm_buf, format="png")
        cm_buf.seek(0)
        plt.close()

        clf_report = classification_report(y, y_pred)


        logger.info('All metrics of model evaluated')
        return metric_dict,cm_buf,clf_report
    
    except Exception as e:
        logger.error(f'Unexpected error occured while evaluating model {e}')
        raise

def save_metrics(metric_dict,cm,clf_report,save_path):
    try:
        metric_save_path = os.path.join(save_path,'metrics')
        os.makedirs(metric_save_path,exist_ok=True)
        with open(os.path.join(metric_save_path,'metrics.json'),'w') as f:
            json.dump(metric_dict,f,indent=4)
        logger.info(f'Metrics saved at {os.path.join(metric_save_path,'metrics.json')}')

        with open(os.path.join(metric_save_path,"classification_report.json"), "w") as f:
            json.dump(clf_report,f,indent=4)
        logger.info(f'Metrics saved at {os.path.join(metric_save_path,"classification_report.txt")}')

        cm_path = os.path.join(metric_save_path, "confusion_matrix.png")
        with open(cm_path, "wb") as f:
            f.write(cm.getvalue())
        logger.info(f'Confusion matrix saved at {cm_path}')

    except Exception as e:
        logger.error(f'Unexpected error occured while saving metrics : {e}')
        raise

def save_model_info(run_id,model_path,file_path):
    try:
        os.makedirs(file_path,exist_ok=True)
        info_file_path = os.path.join(file_path,'model_info.json')
        model_info = {'run_id': run_id, 'model_path': model_path}
        with open(info_file_path, 'w') as file:
            json.dump(model_info, file, indent=4)
        logger.info(f'Model info saved to {file_path}')
    except Exception as e:
        logger.error(f'Error occurred while saving the model info: {e}')
        raise           