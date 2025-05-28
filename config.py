import torch
import warnings
from transformers import logging

class Config:
    # Model settings
    SEED = 42
    N_SPLITS = 5
    MAX_LEN = 180
    N_CLASSES = 3
    CLASS_NAMES = ['Tiêu cực', 'Trung bình', 'Tích cực']
    DEVICE = torch.device('cpu')
    MODEL_NAME = "vinai/phobert-base"
    
    # Server settings
    HOST = '127.0.0.1'
    PORT = 5000
    DEBUG = True
    
    # Analysis settings
    DEFAULT_PAGE_LIMIT = 20
    MAX_REPRESENTATIVE_COMMENTS = 3
    
    @classmethod
    def setup_warnings(cls):
        warnings.filterwarnings("ignore")
        logging.set_verbosity_error()
