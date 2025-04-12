import logging
import os
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import numpy as np

# Environment setup
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class SentimentAnalyzer:
    def __init__(self):
        """Initialize class for sentiment analysis"""
        try:
            # Load default sentiment analysis model (Hugging Face transformers)
            model_name = "distilbert-base-uncased-finetuned-sst-2-english"  # Default sentiment analysis model
            
            logger.info(f"Loading sentiment analysis model: {model_name}")
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
            
            # Additional negative emotion keywords - words related to suicidal thoughts
            self.negative_keywords = [
                'suicide', 'kill myself', 'end my life', 'want to die', 'better off dead',
                'no reason to live', 'can\'t go on', 'hopeless', 'worthless', 'burden',
                'painful', 'miserable', 'trapped', 'can\'t take it anymore', 'no future',
                'alone', 'lonely', 'abandoned', 'depressed', 'despair',
                'overdose', 'self-harm', 'cut myself', 'pills', 'gun',
                'plan to', 'goodbye', 'note', 'last wish', 'never wake up'
            ]
            
            logger.info("Sentiment analyzer initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize sentiment analyzer: {e}")
            raise
    
    def predict_sentiment(self, text):
        """Predict text sentiment (positive/negative)"""
        try:
            # Prepare model input
            inputs = self.tokenizer(text, return_tensors="pt", truncation=True, padding=True)
            
            # Prediction
            with torch.no_grad():
                outputs = self.model(**inputs)
                
            # Process results
            scores = outputs.logits.softmax(dim=1).numpy()[0]
            
            # Model results are in [negative, positive] format
            negative_score = float(scores[0])
            positive_score = float(scores[1])
            
            return {
                'negative_score': negative_score,
                'positive_score': positive_score
            }
        
        except Exception as e:
            logger.error(f"Error in sentiment prediction: {e}")
            return {
                'negative_score': 0.5,
                'positive_score': 0.5
            }
    
    def analyze_suicide_sentiment(self, text):
        """Analyze suicide tendency related emotions in text"""
        try:
            # Basic sentiment analysis
            sentiment = self.predict_sentiment(text)
            negative_score = sentiment['negative_score']
            
            # Detect suicide-related keywords
            keyword_count = 0
            keyword_matches = []
            
            for keyword in self.negative_keywords:
                if keyword.lower() in text.lower():
                    keyword_count += 1
                    keyword_matches.append(keyword)
            
            # Suicide-related keyword weight (max 0.3)
            keyword_weight = min(0.3, keyword_count * 0.02)
            
            # Calculate final risk score
            # Add keyword weight to basic sentiment analysis result (negative_score)
            risk_score_raw = negative_score + keyword_weight
            
            # Adjust score to 1-100 range
            risk_score = min(100, int(risk_score_raw * 100))
            
            # Classify risk level
            risk_level = "Low Risk"
            if risk_score >= 75:
                risk_level = "Very High Risk"
            elif risk_score >= 50:
                risk_level = "High Risk"
            elif risk_score >= 25:
                risk_level = "Medium Risk"
                
            return {
                'risk_score': risk_score,
                'risk_level': risk_level,
                'negative_sentiment': negative_score,
                'keyword_matches': keyword_matches,
                'keyword_count': keyword_count
            }
            
        except Exception as e:
            logger.error(f"Error in suicide sentiment analysis: {e}")
            return {
                'risk_score': 50,  # Default value
                'risk_level': "Medium Risk",
                'negative_sentiment': 0.5,
                'keyword_matches': [],
                'keyword_count': 0
            }

# Test code
if __name__ == "__main__":
    analyzer = SentimentAnalyzer()
    
    # Test samples
    test_samples = [
        "I'm feeling good today, everything is working out well.",
        "I don't know if I can continue living like this anymore. I feel hopeless.",
        "Sometimes I think about ending it all. I have the pills ready.",
        "I had a great day at work and enjoyed time with my family."
    ]
    
    for sample in test_samples:
        result = analyzer.analyze_suicide_sentiment(sample)
        print(f"\nText: {sample}")
        print(f"Risk score: {result['risk_score']}")
        print(f"Risk level: {result['risk_level']}")
        print(f"Negative sentiment: {result['negative_sentiment']:.4f}")
        print(f"Keywords: {result['keyword_matches']}") 