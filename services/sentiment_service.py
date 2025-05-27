import torch
from typing import Tuple, Dict, List
from config import Config
from gensim.utils import simple_preprocess
from pyvi import ViTokenizer

class SentimentService:
    def __init__(self, model_loader):
        self.tokenizer = model_loader.tokenizer
        self.models = model_loader.models
    
    def predict_sentiment(self, text: str) -> Tuple[str, Dict]:
        if not self.models or self.tokenizer is None:
            return "Lỗi hệ thống: Mô hình hoặc tokenizer chưa sẵn sàng.", {}

        try:
            # Preprocess text
            processed_text = ' '.join(simple_preprocess(text))
            tokenized_text = ViTokenizer.tokenize(processed_text)
            
            # Encode text
            encoding = self.tokenizer.encode_plus(
                tokenized_text,
                max_length=Config.MAX_LEN,
                truncation=True,
                add_special_tokens=True,
                padding='max_length',
                return_attention_mask=True,
                return_token_type_ids=False,
                return_tensors='pt',
            )
            
            input_ids = encoding['input_ids'].to(Config.DEVICE)
            attention_mask = encoding['attention_mask'].to(Config.DEVICE)
            
            # Ensemble prediction
            all_outputs = []
            with torch.no_grad():
                for model in self.models:
                    outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                    all_outputs.append(outputs)
            
            # Average results
            stacked_outputs = torch.stack(all_outputs)
            mean_outputs = torch.mean(stacked_outputs, dim=0)
            probabilities = torch.softmax(mean_outputs, dim=1)[0]
            final_pred_index = torch.argmax(probabilities).item()
            
            predicted_label = Config.CLASS_NAMES[final_pred_index]
            prob_dict = {name: prob.item() for name, prob in zip(Config.CLASS_NAMES, probabilities.cpu())}
            
            return predicted_label, prob_dict
            
        except Exception as e:
            print(f"Error in prediction: {e}")
            return f"Lỗi dự đoán: {e}", {}