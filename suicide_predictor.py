import os
import pandas as pd
import numpy as np
import logging
import joblib
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

# 환경 설정
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# NLTK 리소스 다운로드
try:
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)
    nltk.download('wordnet', quiet=True)
except Exception as e:
    logger.warning(f"NLTK download warning: {e}")

class SuicidePredictor:
    def __init__(self, model_path='models/suicide_model.joblib'):
        self.model_path = model_path
        self.model = None
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('english'))
        
        # 모델 로드 또는 생성
        if os.path.exists(model_path):
            try:
                self.model = joblib.load(model_path)
                logger.info(f"Model loaded from {model_path}")
            except Exception as e:
                logger.error(f"Failed to load model: {e}")
                self.model = None
    
    def preprocess_text(self, text):
        """텍스트 전처리 함수"""
        # 소문자 변환
        text = text.lower()
        # 토큰화
        word_tokens = word_tokenize(text)
        # 불용어 제거 및 lemmatization
        filtered_tokens = [
            self.lemmatizer.lemmatize(word) for word in word_tokens 
            if word.isalnum() and word not in self.stop_words
        ]
        return ' '.join(filtered_tokens)
    
    def train(self, csv_path, test_size=0.2, random_state=42):
        """CSV 파일로부터 자살 예측 모델 학습"""
        try:
            # 데이터 로드
            logger.info(f"Loading data from {csv_path}")
            df = pd.read_csv(csv_path)
            
            # 필요한 열만 선택 (텍스트와 라벨)
            # 주의: CSV 파일의 실제 열 이름에 맞게 조정 필요
            text_column = 'text'  # 실제 텍스트 열 이름으로 변경
            label_column = 'class'  # 실제 라벨 열 이름으로 변경
            
            if text_column not in df.columns or label_column not in df.columns:
                # 열 이름 추측
                if len(df.columns) >= 2:
                    text_column = df.columns[0]
                    label_column = df.columns[1]
                    logger.warning(f"Column names not found, using {text_column} and {label_column} instead")
                else:
                    raise ValueError("CSV file does not have enough columns")
            
            logger.info("Preprocessing data...")
            
            # 텍스트 전처리
            df['processed_text'] = df[text_column].fillna('').apply(self.preprocess_text)
            
            # 라벨 변환 (필요한 경우)
            # 예: 'suicide' -> 1, 'non-suicide' -> 0
            if df[label_column].dtype == 'object':
                unique_labels = df[label_column].unique()
                if len(unique_labels) == 2:
                    # 라벨이 이미 바이너리인 경우
                    if all(label in [0, 1] for label in unique_labels):
                        pass
                    else:
                        # 문자열 라벨을 숫자로 변환
                        # 'suicide'와 관련된 라벨을 1로, 나머지를 0으로
                        suicide_keywords = ['suicide', 'suicidal', 'yes', 'positive', '1', 'true']
                        df['label'] = df[label_column].apply(
                            lambda x: 1 if str(x).lower() in suicide_keywords else 0
                        )
                        label_column = 'label'
                        logger.info(f"Converted labels: {df[label_column].value_counts().to_dict()}")
                else:
                    logger.warning(f"Found {len(unique_labels)} unique labels, expected 2")
            
            # 데이터 분할
            X_train, X_test, y_train, y_test = train_test_split(
                df['processed_text'], 
                df[label_column],
                test_size=test_size,
                random_state=random_state,
                stratify=df[label_column]
            )
            
            # 파이프라인 생성
            logger.info("Building and training model...")
            pipeline = Pipeline([
                ('tfidf', TfidfVectorizer(max_features=10000)),
                ('clf', LogisticRegression(C=1.0, class_weight='balanced', max_iter=1000, random_state=random_state))
            ])
            
            # 모델 학습
            pipeline.fit(X_train, y_train)
            
            # 성능 평가
            y_pred = pipeline.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            report = classification_report(y_test, y_pred)
            
            logger.info(f"Model accuracy: {accuracy:.4f}")
            logger.info(f"Classification report:\n{report}")
            
            # 모델 저장
            os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
            joblib.dump(pipeline, self.model_path)
            logger.info(f"Model saved to {self.model_path}")
            
            self.model = pipeline
            return {
                'accuracy': accuracy,
                'report': report
            }
            
        except Exception as e:
            logger.error(f"Error in training model: {e}")
            raise
    
    def predict(self, text):
        """텍스트에서 자살 위험도 예측"""
        if self.model is None:
            raise ValueError("Model not loaded. Please train or load a model first.")
            
        # 텍스트 전처리
        processed_text = self.preprocess_text(text)
        
        # 예측 확률 반환
        proba = self.model.predict_proba([processed_text])[0]
        
        # 1이 자살 위험이 있는 클래스라고 가정
        suicide_prob = proba[1]
        
        # 위험도 점수 계산 (1-100 범위)
        risk_score = int(suicide_prob * 100)
        
        return {
            'risk_score': risk_score,
            'is_suicide_risk': risk_score > 50,
            'raw_probability': float(suicide_prob)
        }

# 테스트 코드
if __name__ == "__main__":
    predictor = SuicidePredictor()
    
    # 모델이 없으면 학습
    if predictor.model is None:
        csv_path = 'data/Suicide_Detection.csv'  # 실제 CSV 파일 경로로 변경
        predictor.train(csv_path)
        
    # 테스트 텍스트로 예측
    test_text = "I don't want to live anymore. Everything feels so hopeless."
    result = predictor.predict(test_text)
    print(f"Risk score: {result['risk_score']}")
    print(f"Is suicide risk: {result['is_suicide_risk']}")
    print(f"Raw probability: {result['raw_probability']:.4f}") 