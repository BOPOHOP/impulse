# thread_module.py
import threading
import logging

from shproto.dispatcher import process_01
from shproto.dispatcher import start

logger = logging.getLogger(__name__)
logger.setLevel(logging.ERROR)

data_thread     = None

def start_data_thread(filename):

    data_thread = threading.Thread(target=process_01, args=(filename,))
    data_thread.start()
    data_thread.join()

    logger.debug(f'thread_module.start_data_tread({filename})')

    return data_thread
