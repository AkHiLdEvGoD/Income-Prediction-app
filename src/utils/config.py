from dotenv import load_dotenv
import os

load_dotenv()
DATA_DIR = os.getenv('DATA_DIR')
DATA_URL = os.getenv('DATA_URL')
RAW_DATA_PATH = os.getenv('RAW_DATA_PATH')
CLEANED_DATA_PATH = os.getenv('CLEANED_DATA_PATH')
FEATURED_DATA_PATH = os.getenv('FEATURED_DATA_PATH')
PROCESSED_DATA_PATH = os.getenv('PROCESSED_DATA_PATH')
ARTIFACTS_DIR = os.getenv('ARTIFACTS_DIR')