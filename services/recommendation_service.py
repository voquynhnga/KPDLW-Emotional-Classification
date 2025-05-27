from typing import Dict

class RecommendationService:
    @staticmethod
    def generate_recommendation(analysis_result: Dict) -> Dict:
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