from loguru import logger
import os
import sys
from datetime import datetime

LOG_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),'logs')
os.makedirs(LOG_DIR,exist_ok=True)
LOG_FILE_PATH = os.path.join(LOG_DIR,f'{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.log')

logger.remove()

logger.add(
    LOG_FILE_PATH,
    rotation='500 KB',
    level='DEBUG',
    backtrace=True,         
    diagnose=True  
)

logger.add(
    sys.stdout,
    level='DEBUG',
    colorize=True,
    format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{module}</cyan>:<cyan>{function}</cyan> - <level>{message}</level>"
)

__all__ = ['logger']