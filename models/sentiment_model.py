import os
import torch
import torch.nn as nn
import numpy as np
from transformers import AutoTokenizer, AutoModel
from pyvi import ViTokenizer
from gensim.utils import simple_preprocess
from config import Config

class SentimentClassifier(nn.Module):
    def __init__(self, n_classes=Config.N_CLASSES):
        super(SentimentClassifier, self).__init__()
        self.bert = AutoModel.from_pretrained(Config.MODEL_NAME)
        self.drop = nn.Dropout(p=0.3)
        self.fc = nn.Linear(self.bert.config.hidden_size, n_classes)
        nn.init.normal_(self.fc.weight, std=0.02)
        nn.init.normal_(self.fc.bias, 0)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True
        )
        pooler_output = outputs.pooler_output
        x = self.drop(pooler_output)
        x = self.fc(x)
        return x

class ModelLoader:
    def __init__(self):
        self.tokenizer = None
        self.models = []
        self._load_tokenizer()
        self._load_models()
    
    def _load_tokenizer(self):
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(Config.MODEL_NAME, use_fast=False)
            print("Tokenizer loaded successfully.")
        except Exception as e:
            print(f"FATAL: Error loading tokenizer: {e}")
            raise
    
    def _load_models(self):
        for fold in range(Config.N_SPLITS):
            model_path = f'phobert_sentiment_fold{fold+1}.pth'
            if os.path.exists(model_path):
                try:
                    model = SentimentClassifier(n_classes=Config.N_CLASSES)
                    model.load_state_dict(torch.load(model_path, map_location=Config.DEVICE))
                    model.to(Config.DEVICE)
                    model.eval()
                    self.models.append(model)
                    print(f"Successfully loaded model from: {model_path}")
                except Exception as e:
                    print(f"ERROR: Could not load model from {model_path}. Error: {e}")
            else:
                print(f"WARNING: Model file not found for fold {fold+1} at {model_path}")
        
        if not self.models:
            raise RuntimeError("ERROR: No models were loaded. The application cannot predict.")