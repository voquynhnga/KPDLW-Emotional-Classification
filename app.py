import os
import csv
import torch
import torch.nn as nn
import numpy as np
from flask import Flask, request, jsonify, render_template
from transformers import AutoTokenizer, AutoModel, logging
from pyvi import ViTokenizer
from gensim.utils import simple_preprocess
import traceback
import warnings
from livereload import Server


from urllib.parse import urlparse, parse_qs
from concurrent.futures import ThreadPoolExecutor
from crawler.DMXCrawler import DMXCrawler
from crawler.TIKICrawler import TIKICrawler

# --- Configuration ---
warnings.filterwarnings("ignore")
logging.set_verbosity_error()

# --- Constants ---
SEED = 42
N_SPLITS = 5
MAX_LEN = 180
N_CLASSES = 3
CLASS_NAMES = ['Tiêu cực', 'Trung bình', 'Tích cực']
DEVICE = torch.device('cpu')


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

# --- Model Definition ---
class SentimentClassifier(nn.Module):
    def __init__(self, n_classes=N_CLASSES):
        super(SentimentClassifier, self).__init__()
        try:
            self.bert = AutoModel.from_pretrained("vinai/phobert-base")
        except Exception as e:
            print(f"FATAL: Error loading PhoBERT base model: {e}")
            raise
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

# --- Load Tokenizer and Models ---
try:
    TOKENIZER = AutoTokenizer.from_pretrained("vinai/phobert-base", use_fast=False)
    print("Tokenizer loaded successfully.")
except Exception as e:
    print(f"FATAL: Error loading tokenizer: {e}")
    TOKENIZER = None

MODELS = []
if TOKENIZER:
    for fold in range(N_SPLITS):
        model_path = f'phobert_sentiment_fold{fold+1}.pth'
        if os.path.exists(model_path):
            try:
                model = SentimentClassifier(n_classes=N_CLASSES)
                model.load_state_dict(torch.load(model_path, map_location=DEVICE))
                model.to(DEVICE)
                model.eval()
                MODELS.append(model)
                print(f"Successfully loaded model from: {model_path}")
            except Exception as e:
                print(f"ERROR: Could not load model from {model_path}. Error: {e}")
        else:
            print(f"WARNING: Model file not found for fold {fold+1} at {model_path}")

if not MODELS:
    print("ERROR: No models were loaded. The application cannot predict.")

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')


def predict_sentiment(text):
    if not MODELS or TOKENIZER is None:
        return "Lỗi hệ thống: Mô hình hoặc tokenizer chưa sẵn sàng.", {}

    try:
        # Preprocess text
        processed_text = ' '.join(simple_preprocess(text))
        tokenized_text = ViTokenizer.tokenize(processed_text)
        
        # Encode text
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
        
        # Ensemble prediction
        all_outputs = []
        with torch.no_grad():
            for model in MODELS:
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                all_outputs.append(outputs)
        
        # Average results
        stacked_outputs = torch.stack(all_outputs)
        mean_outputs = torch.mean(stacked_outputs, dim=0)
        probabilities = torch.softmax(mean_outputs, dim=1)[0]
        final_pred_index = torch.argmax(probabilities).item()
        
        predicted_label = CLASS_NAMES[final_pred_index]
        prob_dict = {name: prob.item() for name, prob in zip(CLASS_NAMES, probabilities.cpu())}
        
        return predicted_label, prob_dict
        
    except Exception as e:
        print(f"Error in prediction: {e}")
        return f"Lỗi dự đoán: {e}", {}

def analyze_comments(comments):
    if not comments:
        return {
            "total": 0,
            "sentiment_stats": {"Tích cực": 0, "Trung bình": 0, "Tiêu cực": 0},
            "representative_comments": {"positive": [], "neutral": [], "negative": []},
            "sentiment_scores": []
        }
    
    results = []
    sentiment_counts = {"Tích cực": 0, "Trung bình": 0, "Tiêu cực": 0}
    comments_by_sentiment = {"positive": [], "neutral": [], "negative": []}
    
    print(f"Analyzing {len(comments)} comments...")
    
    for i, comment in enumerate(comments):
        if i % 10 == 0:
            print(f"Processed {i}/{len(comments)} comments")
            
        label, probs = predict_sentiment(comment)
        
        if isinstance(probs, dict) and probs:
            sentiment_counts[label] += 1
            
            # Store comment with sentiment info
            comment_data = {
                "text": comment,
                "label": label,
                "probabilities": probs,
                "confidence": max(probs.values())
            }
            results.append(comment_data)
            
            # Categorize for representative comments
            if label == "Tích cực":
                comments_by_sentiment["positive"].append(comment)
            elif label == "Tiêu cực":
                comments_by_sentiment["negative"].append(comment)
            else:
                comments_by_sentiment["neutral"].append(comment)
    
    # Select representative comments (top 5 by confidence for each category)
    representative = {}
    for key in ["positive", "neutral", "negative"]:
        sentiment_comments = [r for r in results if 
                            (key == "positive" and r["label"] == "Tích cực") or
                            (key == "negative" and r["label"] == "Tiêu cực") or
                            (key == "neutral" and r["label"] == "Trung bình")]
        
        # Sort by confidence and take top 5
        sentiment_comments.sort(key=lambda x: x["confidence"], reverse=True)
        representative[key] = [c["text"] for c in sentiment_comments[:5]]
    
    return {
        "total": len(comments),
        "sentiment_stats": sentiment_counts,
        "representative_comments": representative,
        "sentiment_scores": results
    }

def generate_recommendation(analysis_result):
    """Generate purchase recommendation based on sentiment analysis"""
    stats = analysis_result["sentiment_stats"]
    total = analysis_result["total"]
    
    if total == 0:
        return {
            "decision": "consider",
            "reason": "Không có đủ đánh giá để đưa ra khuyến nghị.",
            "confidence": 0
        }
    
    positive_ratio = stats["Tích cực"] / total
    negative_ratio = stats["Tiêu cực"] / total
    neutral_ratio = stats["Trung bình"] / total
    
    # Decision logic
    if positive_ratio >= 0.6:
        decision = "buy"
        reason = f"Sản phẩm có {positive_ratio:.1%} đánh giá tích cực. Khách hàng đánh giá cao về chất lượng và hiệu suất."
        confidence = positive_ratio
    elif negative_ratio >= 0.4:
        decision = "avoid"
        reason = f"Sản phẩm có {negative_ratio:.1%} đánh giá tiêu cực. Nhiều khách hàng không hài lòng về sản phẩm."
        confidence = negative_ratio
    elif positive_ratio >= 0.4 and negative_ratio <= 0.3:
        decision = "buy"
        reason = f"Sản phẩm có {positive_ratio:.1%} đánh giá tích cực và chỉ {negative_ratio:.1%} đánh giá tiêu cực. Tổng thể khá tốt."
        confidence = positive_ratio - negative_ratio
    else:
        decision = "consider"
        reason = f"Sản phẩm có ý kiến trái chiều: {positive_ratio:.1%} tích cực, {negative_ratio:.1%} tiêu cực. Nên cân nhắc kỹ trước khi mua."
        confidence = abs(positive_ratio - negative_ratio)
    
    return {
        "decision": decision,
        "reason": reason,
        "confidence": confidence,
        "stats_summary": f"Tích cực: {stats['Tích cực']}, Trung bình: {stats['Trung bình']}, Tiêu cực: {stats['Tiêu cực']}"
    }

# Single product analysis
@app.route('/analyze-product', methods=['POST'])
def analyze_single_product():
    if not request.is_json:
        return jsonify({"error": "Yêu cầu phải ở định dạng JSON"}), 400
    
    data = request.get_json()
    if 'url' not in data:
        return jsonify({"error": "Thiếu trường 'url' trong yêu cầu"}), 400
    
    url = data['url'].strip()

    if url.startswith('https://www.dienmayxanh.com/'):
        crawler = DMXCrawler()
    elif url.startswith('https://tiki.vn/'):
        crawler = TIKICrawler()
    if not url or ('dienmayxanh.com' not in url and 'tiki.vn' not in url):
        return jsonify({"error": "URL không hợp lệ"}), 400

    
    if not MODELS:
        return jsonify({"error": "Lỗi hệ thống: Mô hình chưa sẵn sàng."}), 503
    
    try:

        comments = crawler.get_comments(url, 20)  # Limit to 10 pages for faster analysis
        
        if not comments:
            return jsonify({
                "error": "Không tìm thấy bình luận nào cho sản phẩm này. Có thể sản phẩm chưa có đánh giá."
            }), 404
        

        analysis = analyze_comments(comments)
        
        # Generate recommendation
        recommendation = generate_recommendation(analysis)
        
        return jsonify({
            "product_url": url,
            "total_reviews": len(comments),
            "sentiment_stats": analysis["sentiment_stats"],
            "representative_comments": analysis["representative_comments"],
            "recommendation": recommendation
        })
        
    except Exception as e:
        print(f"Error analyzing product: {e}")
        print(traceback.format_exc())
        return jsonify({"error": f"Lỗi phân tích sản phẩm: {str(e)}"}), 500

# Original prediction endpoint (for backward compatibility)
@app.route('/predict', methods=['POST'])
def predict_api():
    if not request.is_json:
        return jsonify({"error": "Yêu cầu phải ở định dạng JSON"}), 400

    data = request.get_json()
    if 'text' not in data:
        return jsonify({"error": "Thiếu trường 'text' trong yêu cầu"}), 400

    input_text = data['text']
    if not isinstance(input_text, str) or not input_text.strip():
        return jsonify({"error": "Trường 'text' phải là một chuỗi không rỗng"}), 400

    if not MODELS:
        return jsonify({"error": "Lỗi hệ thống: Mô hình chưa sẵn sàng."}), 503

    label, probabilities = predict_sentiment(input_text)

    if not isinstance(probabilities, dict):
        return jsonify({"error": label}), 500

    return jsonify({
        'predicted_label': label,
        'probabilities': probabilities
    })


if __name__ == '__main__':
    print("Starting Flask server...")
    print(f"Models loaded: {len(MODELS)}")
    print("Server ready for product analysis!")


    server = Server(app.wsgi_app)
    server.watch('templates/')
    server.watch('static/')
    server.watch('app.py')
    server.serve(host='127.0.0.1', port=5000, debug=True)