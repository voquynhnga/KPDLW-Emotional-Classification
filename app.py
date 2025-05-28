import os
import csv
import torch
import torch.nn as nn
import numpy as np
from flask import Flask, request, jsonify, render_template
from transformers import AutoTokenizer, AutoModel, logging
import traceback
from flask import Flask, request, jsonify, render_template
from livereload import Server

from config import Config
from models.sentiment_model import ModelLoader
from services.sentiment_service import SentimentService
from services.analysis_service import AnalysisService
from services.recommendation_service import RecommendationService
from utils.crawler_factory import CrawlerFactory
from utils.validators import Validators
from utils.seed_utils import seed_everything

class SentimentAnalysisApp:
    def __init__(self):
        Config.setup_warnings()
        seed_everything(Config.SEED)
        
        self.app = Flask(__name__)
        self._initialize_services()
        self._register_routes()
    
    def _initialize_services(self):
        try:
            model_loader = ModelLoader()
            self.sentiment_service = SentimentService(model_loader)
            self.analysis_service = AnalysisService(self.sentiment_service)
            self.recommendation_service = RecommendationService()
            print(f"Models loaded: {len(model_loader.models)}")
        except Exception as e:
            print(f"Failed to initialize services: {e}")
            raise
    
    def _register_routes(self):
        self.app.add_url_rule('/', 'home', self.home)
        self.app.add_url_rule('/analyze-product', 'analyze_product', self.analyze_product, methods=['POST'])
        self.app.add_url_rule('/predict', 'predict', self.predict_api, methods=['POST'])
    
    def home(self):
        return render_template('index.html')
    
    def analyze_product(self):
        if not request.is_json:
            return jsonify({"error": "Yêu cầu phải ở định dạng JSON"}), 400
        
        data = request.get_json()
        if 'url' not in data:
            return jsonify({"error": "Thiếu trường 'url' trong yêu cầu"}), 400
        
        url = data['url'].strip()
        
        if not Validators.validate_url(url):
            return jsonify({"error": "URL không hợp lệ"}), 400
        
        try:
            crawler = CrawlerFactory.get_crawler(url)
            comments = crawler.get_comments(url, Config.DEFAULT_PAGE_LIMIT)
            
            if not comments:
                return jsonify({
                    "error": "Không tìm thấy bình luận nào cho sản phẩm này. Có thể sản phẩm chưa có đánh giá."
                }), 404
            
            analysis = self.analysis_service.analyze_comments(comments)
            recommendation = self.recommendation_service.generate_recommendation(analysis)
            
            return jsonify({
                "product_url": url,
                "total_reviews": len(comments),
                "sentiment_stats": analysis["sentiment_stats"],
                "representative_comments": analysis["representative_comments"],
                "recommendation": recommendation
            })
            
        except ValueError as e:
            return jsonify({"error": str(e)}), 400
        except Exception as e:
            print(f"Error analyzing product: {e}")
            print(traceback.format_exc())
            return jsonify({"error": f"Lỗi phân tích sản phẩm: {str(e)}"}), 500
    
    def predict_api(self):
        if not request.is_json:
            return jsonify({"error": "Yêu cầu phải ở định dạng JSON"}), 400

        data = request.get_json()
        if 'text' not in data:
            return jsonify({"error": "Thiếu trường 'text' trong yêu cầu"}), 400

        input_text = data['text']
        if not Validators.validate_text(input_text):
            return jsonify({"error": "Trường 'text' phải là một chuỗi không rỗng"}), 400

        label, probabilities = self.sentiment_service.predict_sentiment(input_text)

        if not isinstance(probabilities, dict):
            return jsonify({"error": label}), 500

        return jsonify({
            'predicted_label': label,
            'probabilities': probabilities
        })
    
    def run(self):
        print("Starting Flask server...")
        print("Server ready for product analysis!")
        
        if Config.DEBUG:
            server = Server(self.app.wsgi_app)
            server.watch('templates/')
            server.watch('static/')
            server.watch('app.py')
            server.serve(host=Config.HOST, port=Config.PORT, debug=Config.DEBUG)
        else:
            self.app.run(host=Config.HOST, port=Config.PORT, debug=Config.DEBUG)

if __name__ == '__main__':
    app = SentimentAnalysisApp()
    app.run()
