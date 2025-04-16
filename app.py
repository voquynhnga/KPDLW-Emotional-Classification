import os
import torch
import torch.nn as nn
import numpy as np
from flask import Flask, request, jsonify, render_template
from transformers import AutoTokenizer, AutoModel, logging
from pyvi import ViTokenizer
from gensim.utils import simple_preprocess
import traceback
import warnings

# --- Configuration ---
warnings.filterwarnings("ignore")
logging.set_verbosity_error() # Chỉ log lỗi từ transformers

# --- Constants ---
SEED = 42
N_SPLITS = 5
MAX_LEN = 180 # Điều chỉnh nếu bạn dùng giá trị khác khi huấn luyện
N_CLASSES = 3
# QUAN TRỌNG: Đảm bảo thứ tự này khớp với cách bạn huấn luyện và file index.html
CLASS_NAMES = ['Tiêu cực', 'Trung bình', 'Tích cực']
DEVICE = torch.device('cpu')
print(f"Using device: {DEVICE}")
# DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {DEVICE}")

# --- Reproducibility ---
def seed_everything(seed_value):
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

seed_everything(SEED)

# --- Model Definition (Copy từ notebook) ---
class SentimentClassifier(nn.Module):
    def __init__(self, n_classes=N_CLASSES):
        super(SentimentClassifier, self).__init__()
        try:
            self.bert = AutoModel.from_pretrained("vinai/phobert-base")
        except Exception as e:
            print(f"FATAL: Error loading PhoBERT base model: {e}")
            raise # Ngưng nếu không load được base model
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

# --- Load Tokenizer and Models (Load một lần khi khởi động) ---
print("Loading tokenizer...")
try:
    TOKENIZER = AutoTokenizer.from_pretrained("vinai/phobert-base", use_fast=False)
    print("Tokenizer loaded successfully.")
except Exception as e:
    print(f"FATAL: Error loading tokenizer: {e}")
    TOKENIZER = None # Sẽ gây lỗi sau nếu không load được

print("Loading trained models...")
MODELS = []
if TOKENIZER:
    for fold in range(N_SPLITS):
        model_path = f'phobert_sentiment_fold{fold+1}.pth'
        if os.path.exists(model_path):
            try:
                model = SentimentClassifier(n_classes=N_CLASSES)
                model.load_state_dict(torch.load(model_path, map_location=DEVICE))
                model.to(DEVICE)
                model.eval() # Quan trọng: Chuyển sang chế độ đánh giá
                MODELS.append(model)
                print(f"Successfully loaded model from: {model_path}")
            except Exception as e:
                 print(f"ERROR: Could not load model from {model_path}. Error: {e}")
                 print(traceback.format_exc())
        else:
            print(f"WARNING: Model file not found for fold {fold+1} at {model_path}")
else:
     print("WARNING: Skipping model loading because tokenizer failed to load.")

if not MODELS:
    print("\n!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
    print("ERROR: No models were loaded. The application cannot predict.")
    print("Please ensure model files exist and tokenizer loaded correctly.")
    print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n")

# --- Flask App Initialization ---
app = Flask(__name__)

# --- Prediction Function ---
def predict_sentiment(text):
    """
    Performs sentiment prediction using the loaded ensemble models.
    Returns: (predicted_label_str, probability_dict) or (error_message_str, {})
    """
    if not MODELS or TOKENIZER is None:
        return "Lỗi hệ thống: Mô hình hoặc tokenizer chưa sẵn sàng.", {}

    # 1. Preprocess Text
    try:
        processed_text = ' '.join(simple_preprocess(text))
        tokenized_text = ViTokenizer.tokenize(processed_text)
    except Exception as e_pre:
        print(f"Error preprocessing text: '{text[:50]}...'. Error: {e_pre}")
        return f"Lỗi tiền xử lý: {e_pre}", {}

    # 2. Encode Text
    try:
        encoding = TOKENIZER.encode_plus(
            tokenized_text,
            max_length=MAX_LEN,
            truncation=True,
            add_special_tokens=True,
            padding='max_length',
            return_attention_mask=True,
            return_token_type_ids=False,
            return_tensors='pt',
        )
        input_ids = encoding['input_ids'].to(DEVICE)
        attention_mask = encoding['attention_mask'].to(DEVICE)
    except Exception as e_enc:
        print(f"Error encoding text: '{tokenized_text[:50]}...'. Error: {e_enc}")
        return f"Lỗi mã hóa: {e_enc}", {}

    # 3. Ensemble Prediction
    all_outputs = []
    with torch.no_grad():
        for i, model in enumerate(MODELS):
            try:
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                all_outputs.append(outputs)
            except Exception as e_inf:
                print(f"Error during inference with model {i+1}: {e_inf}")
                print(traceback.format_exc())
                # Có thể tiếp tục với các model khác hoặc báo lỗi ngay
                return f"Lỗi dự đoán với mô hình {i+1}: {e_inf}", {}

    if not all_outputs:
         print("Error: No outputs generated from models.")
         return "Lỗi: Không có kết quả từ mô hình.", {}

    # 4. Average results & Get Final Prediction
    try:
        stacked_outputs = torch.stack(all_outputs)
        mean_outputs = torch.mean(stacked_outputs, dim=0)
        probabilities = torch.softmax(mean_outputs, dim=1)[0] # Chỉ có 1 sample input
        final_pred_index = torch.argmax(probabilities).item() # argmax trên tensor xác suất

        predicted_label = CLASS_NAMES[final_pred_index]
        prob_dict = {name: prob.item() for name, prob in zip(CLASS_NAMES, probabilities.cpu())}

        print(f"Prediction for '{text[:50]}...': Label={predicted_label}, Probs={prob_dict}")
        return predicted_label, prob_dict
    except Exception as e_post:
        print(f"Error processing model outputs: {e_post}")
        print(traceback.format_exc())
        return f"Lỗi xử lý kết quả: {e_post}", {}

# --- Flask Routes ---
@app.route('/')
def home():
    """Renders the main HTML page."""
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict_api():
    """API endpoint to handle prediction requests."""
    if not request.is_json:
        return jsonify({"error": "Yêu cầu phải ở định dạng JSON"}), 400

    data = request.get_json()
    if 'text' not in data:
        return jsonify({"error": "Thiếu trường 'text' trong yêu cầu"}), 400

    input_text = data['text']
    if not isinstance(input_text, str) or not input_text.strip():
         return jsonify({"error": "Trường 'text' phải là một chuỗi không rỗng"}), 400

    if not MODELS: # Kiểm tra lại nếu model không load được
         print("Attempted prediction with no models loaded.")
         return jsonify({"error": "Lỗi hệ thống: Mô hình chưa sẵn sàng."}), 503 # Service Unavailable

    print(f"Received prediction request for: '{input_text[:100]}...'")
    label, probabilities = predict_sentiment(input_text)

    # Kiểm tra nếu predict_sentiment trả về lỗi
    if not isinstance(probabilities, dict):
         # Label lúc này sẽ chứa thông báo lỗi
         return jsonify({"error": label}), 500 # Internal Server Error

    return jsonify({
        'predicted_label': label,
        'probabilities': probabilities
    })

# --- Run the App ---
if __name__ == '__main__':
    print("Starting Flask server...")
    # Sử dụng debug=False khi deploy hoặc chia sẻ
    # host='0.0.0.0' cho phép truy cập từ các máy khác trong mạng
    app.run(host='0.0.0.0', port=5000, debug=False)