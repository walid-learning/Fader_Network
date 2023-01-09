import logging
import pathlib
from datetime import datetime as date
import os

################## Logging #################
PARENT_PATH = os.path.dirname(os.path.abspath(__file__))
PATH = PARENT_PATH[:-3]
logging_path = PARENT_PATH[:-3] + "log"
logger = logging.getLogger()
info = logger.info
debug = logger.debug
warning = logger.warning

def log_config(name) : 
    logging.basicConfig(
        level = logging.DEBUG,
        format = " {levelname:<8} {asctime} {message}",
        style='{',
        filename=logging_path + f'\\{name}_{date.today().strftime("%d-%m-%Y_%Hh%M")}.log',
        filemode='w'
    )



def save_loss(msg):
    with open(f'{PATH}utils\\loss\\loss_{date.today().strftime("%d-%m-%Y")}.txt', "w") as f:
        f.write(msg + '\n')
