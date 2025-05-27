from typing import List, Dict
from config import Config

class AnalysisService:
    def __init__(self, sentiment_service):
        self.sentiment_service = sentiment_service
    
    def analyze_comments(self, comments: List[str]) -> Dict:
        if not comments:
            return self._empty_analysis()
        
        results = []
        sentiment_counts = {"Tích cực": 0, "Trung bình": 0, "Tiêu cực": 0}
        
        print(f"Analyzing {len(comments)} comments...")
        
        for i, comment in enumerate(comments):
            if i % 10 == 0:
                print(f"Processed {i}/{len(comments)} comments")
                
            label, probs = self.sentiment_service.predict_sentiment(comment)
            
            if isinstance(probs, dict) and probs:
                sentiment_counts[label] += 1
                
                comment_data = {
                    "text": comment,
                    "label": label,
                    "probabilities": probs,
                    "confidence": max(probs.values())
                }
                results.append(comment_data)
        
        representative_comments = self._get_representative_comments(results)
        
        return {
            "total": len(comments),
            "sentiment_stats": sentiment_counts,
            "representative_comments": representative_comments,
            "sentiment_scores": results
        }
    
    def _empty_analysis(self) -> Dict:
        return {
            "total": 0,
            "sentiment_stats": {"Tích cực": 0, "Trung bình": 0, "Tiêu cực": 0},
            "representative_comments": {"positive": [], "neutral": [], "negative": []},
            "sentiment_scores": []
        }
    
    def _get_representative_comments(self, results: List[Dict]) -> Dict:
        representative = {}
        sentiment_mapping = {
            "positive": "Tích cực",
            "negative": "Tiêu cực", 
            "neutral": "Trung bình"
        }
        
        for key, label in sentiment_mapping.items():
            sentiment_comments = [r for r in results if r["label"] == label]
            sentiment_comments.sort(key=lambda x: x["confidence"], reverse=True)
            representative[key] = [c["text"] for c in sentiment_comments[:Config.MAX_REPRESENTATIVE_COMMENTS]]
        
        return representative