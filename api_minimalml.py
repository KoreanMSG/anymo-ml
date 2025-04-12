import os
import logging
import uvicorn
import pandas as pd
import numpy as np
import re
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
from enum import Enum
from datetime import datetime

# Basic logging configuration
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Path settings
DATA_DIR = os.path.join(os.path.dirname(__file__), 'data')
MODELS_DIR = os.path.join(os.path.dirname(__file__), 'models')
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)

# Global variables
ml_model = None
ml_model_loaded = False

# Initialize FastAPI app
app = FastAPI(
    title="Suicide Risk Analysis API - Lightweight ML Version",
    description="API for analyzing suicide risk in text conversations (memory-optimized ML version)",
    version="1.0.0-minimalml"
)

# CORS settings
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Data model definitions
class TextInput(BaseModel):
    text: str = Field(..., description="Text to analyze")

class RiskLevel(str, Enum):
    NO_RISK = "No Risk"
    LOW_RISK = "Low Risk"
    MEDIUM_RISK = "Medium Risk"
    HIGH_RISK = "High Risk"
    SEVERE_RISK = "Severe Risk"

class AnalysisResult(BaseModel):
    risk_score: int = Field(..., description="Risk score (1-100)")
    risk_level: RiskLevel = Field(..., description="Risk level")
    ml_confidence: float = Field(..., description="ML model confidence")
    keywords_found: List[str] = Field(default=[], description="Found risk keywords")
    analysis_time: str = Field(..., description="Analysis time")
    method: str = Field(..., description="Analysis method (ML/keyword)")

# Suicide-related keywords (Korean and English) - for keyword matching
SUICIDE_KEYWORDS = {
    "high": [],
    "medium": [],
    "low": []
}

# Function to extract keywords from CSV
def extract_keywords_from_csv():
    """Function to extract suicide-related keywords from CSV data"""
    global SUICIDE_KEYWORDS
    
    csv_path = os.getenv("CSV_PATH", os.path.join(DATA_DIR, "Suicide_Detection_sample.csv"))
    if not os.path.exists(csv_path):
        logger.warning(f"No CSV file for keyword extraction: {csv_path}")
        # Set default keywords
        SUICIDE_KEYWORDS = {
            "high": [
                "자살", "죽고 싶", "목숨", "목맬", "목을 매", "죽어버리", "없어지고 싶", "사라지고 싶", 
                "suicide", "kill myself", "end my life", "take my life"
            ],
            "medium": [
                "우울", "희망이 없", "삶이 의미", "살아갈 이유", "혼자", "외롭", "고통", 
                "hopeless", "worthless", "alone", "lonely", "pain", "suffering", "depressed"
            ],
            "low": [
                "슬프", "힘들", "지쳤", "피곤", "실패", "포기", "도움", 
                "sad", "tired", "exhausted", "failed", "give up", "help me"
            ]
        }
        return
    
    try:
        # Read CSV
        df = pd.read_csv(csv_path)
        
        # Check data structure
        if len(df.columns) >= 2:
            text_column = df.columns[0] if df.columns[0] != '' else df.columns[1]
            label_column = df.columns[1]
            
            # Separate suicide-related texts
            suicide_texts = []
            non_suicide_texts = []
            
            if df[label_column].dtype == 'object':
                suicide_keywords = ['suicide', 'suicidal', 'yes', 'positive', '1', 'true']
                suicide_mask = df[label_column].apply(
                    lambda x: 1 if str(x).lower() in suicide_keywords else 0
                ).astype(bool)
            else:
                suicide_mask = df[label_column].astype(bool)
            
            suicide_texts = df.loc[suicide_mask, text_column].tolist()
            non_suicide_texts = df.loc[~suicide_mask, text_column].tolist()
            
            # Use TF-IDF to extract major keywords
            tfidf = TfidfVectorizer(max_features=300, stop_words='english')
            tfidf_matrix = tfidf.fit_transform(suicide_texts)
            
            # Extract key words
            feature_names = tfidf.get_feature_names_out()
            
            # Sort by score and select top keywords
            tfidf_sum = tfidf_matrix.sum(axis=0).A1
            word_scores = [(word, score) for word, score in zip(feature_names, tfidf_sum)]
            word_scores.sort(key=lambda x: x[1], reverse=True)
            
            # Classify keywords based on score
            total_words = len(word_scores)
            high_threshold = int(total_words * 0.2)  # Top 20%
            medium_threshold = int(total_words * 0.5)  # Top 50%
            
            SUICIDE_KEYWORDS["high"] = [word for word, _ in word_scores[:high_threshold] if len(word) > 2]
            SUICIDE_KEYWORDS["medium"] = [word for word, _ in word_scores[high_threshold:medium_threshold] if len(word) > 2]
            SUICIDE_KEYWORDS["low"] = [word for word, _ in word_scores[medium_threshold:] if len(word) > 2]
            
            logger.info(f"Keywords extracted from CSV: {len(SUICIDE_KEYWORDS['high'])} high, {len(SUICIDE_KEYWORDS['medium'])} medium, {len(SUICIDE_KEYWORDS['low'])} low")
            
            # Add default important keywords (including other languages)
            SUICIDE_KEYWORDS["high"].extend(["자살", "죽고 싶", "suicide", "kill myself"])
            SUICIDE_KEYWORDS["medium"].extend(["우울", "희망이 없", "혼자", "외롭", "hopeless", "worthless"])
            SUICIDE_KEYWORDS["low"].extend(["슬프", "힘들", "지쳤", "sad", "tired"])
        else:
            logger.error(f"Not enough columns in CSV: {csv_path}")
    except Exception as e:
        logger.error(f"Keyword extraction failed: {e}")

# Text preprocessing function
def preprocess_text(text):
    """Text preprocessing function"""
    # Convert to lowercase
    text = text.lower()
    # Remove special characters
    text = re.sub(r'[^\w\s]', '', text)
    return text

# Keyword-based analysis function
def analyze_text_keywords(text: str) -> Dict[str, Any]:
    """Keyword-based suicide risk analysis"""
    text_lower = text.lower()
    
    # Search for keywords
    found_keywords = {
        "high": [kw for kw in SUICIDE_KEYWORDS["high"] if kw.lower() in text_lower],
        "medium": [kw for kw in SUICIDE_KEYWORDS["medium"] if kw.lower() in text_lower],
        "low": [kw for kw in SUICIDE_KEYWORDS["low"] if kw.lower() in text_lower]
    }
    
    # Calculate risk score
    high_count = len(found_keywords["high"]) * 15
    medium_count = len(found_keywords["medium"]) * 7
    low_count = len(found_keywords["low"]) * 3
    
    # Repetition weight
    repetition_multiplier = 1.0
    for level in ["high", "medium", "low"]:
        for kw in found_keywords[level]:
            count = len(re.findall(re.escape(kw.lower()), text_lower))
            if count > 1:
                repetition_multiplier += 0.1 * (count - 1)
    
    # Calculate total score (max 100)
    score = min(100, int((high_count + medium_count + low_count) * repetition_multiplier))
    
    # Determine risk level
    if score >= 80:
        risk_level = RiskLevel.SEVERE_RISK
    elif score >= 60:
        risk_level = RiskLevel.HIGH_RISK
    elif score >= 40:
        risk_level = RiskLevel.MEDIUM_RISK
    elif score >= 20:
        risk_level = RiskLevel.LOW_RISK
    else:
        risk_level = RiskLevel.NO_RISK
    
    # Combine all keywords into one list
    all_keywords = found_keywords["high"] + found_keywords["medium"] + found_keywords["low"]
    
    return {
        "risk_score": score,
        "risk_level": risk_level,
        "ml_confidence": 0.0,  # Since it's keyword-based analysis, ML confidence is 0
        "keywords_found": all_keywords,
        "analysis_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "method": "keyword"
    }

# ML 모델 로드 함수
def load_ml_model():
    """저장된 ML 모델을 로드하거나 CSV에서 학습"""
    global ml_model, ml_model_loaded
    
    if ml_model is not None:
        return True
    
    model_path = os.path.join(MODELS_DIR, 'suicide_model_minimal.joblib')
    
    # 저장된 모델이 있으면 로드
    if os.path.exists(model_path):
        try:
            ml_model = joblib.load(model_path)
            ml_model_loaded = True
            logger.info(f"ML 모델 로드됨: {model_path}")
            
            # 키워드도 함께 추출
            extract_keywords_from_csv()
            return True
        except Exception as e:
            logger.error(f"모델 로드 실패: {e}")
    
    # 저장된 모델이 없으면 CSV에서 학습
    csv_path = os.getenv("CSV_PATH", os.path.join(DATA_DIR, "Suicide_Detection_sample.csv"))
    if os.path.exists(csv_path):
        try:
            # 데이터 로드
            logger.info(f"CSV 데이터에서 모델 학습 시작: {csv_path}")
            df = pd.read_csv(csv_path)
            
            # 키워드 추출
            extract_keywords_from_csv()
            
            # 데이터 구조 추측 (첫 두 열을 텍스트와 라벨로 가정)
            if len(df.columns) >= 2:
                text_column = df.columns[0] if df.columns[0] != '' else df.columns[1]
                label_column = df.columns[1]
                
                # 데이터 준비
                X = df[text_column].fillna('').apply(preprocess_text)
                
                # 라벨 변환 (필요한 경우)
                if df[label_column].dtype == 'object':
                    suicide_keywords = ['suicide', 'suicidal', 'yes', 'positive', '1', 'true']
                    y = df[label_column].apply(
                        lambda x: 1 if str(x).lower() in suicide_keywords else 0
                    )
                else:
                    y = df[label_column]
                
                # 경량 ML 모델 생성 (메모리 효율적인 설정)
                pipeline = Pipeline([
                    ('tfidf', TfidfVectorizer(max_features=5000)),  # 적은 특성 사용
                    ('clf', LogisticRegression(C=1.0, solver='liblinear', max_iter=100))  # 빠른 솔버 사용
                ])
                
                # 모델 학습
                pipeline.fit(X, y)
                
                # 모델 저장
                joblib.dump(pipeline, model_path)
                ml_model = pipeline
                ml_model_loaded = True
                logger.info(f"모델 학습 및 저장 완료: {model_path}")
                return True
            else:
                logger.error(f"CSV에 충분한 열이 없음: {csv_path}")
        except Exception as e:
            logger.error(f"모델 학습 실패: {e}")
    else:
        logger.warning(f"CSV 파일을 찾을 수 없음: {csv_path}")
    
    return False

# ML 기반 분석 함수
def analyze_text_ml(text: str) -> Dict[str, Any]:
    """ML 모델을 사용한 자살 위험도 분석"""
    if not ml_model_loaded:
        if not load_ml_model():
            # ML 모델 로드 실패 시 키워드 분석으로 폴백
            result = analyze_text_keywords(text)
            return result
    
    try:
        # 텍스트 전처리
        processed_text = preprocess_text(text)
        
        # 예측 확률 반환
        proba = ml_model.predict_proba([processed_text])[0]
        
        # 1이 자살 위험이 있는 클래스라고 가정
        suicide_prob = proba[1]
        
        # 위험도 점수 계산 (1-100 범위)
        risk_score = int(suicide_prob * 100)
        
        # 위험도 레벨 결정
        if risk_score >= 80:
            risk_level = RiskLevel.SEVERE_RISK
        elif risk_score >= 60:
            risk_level = RiskLevel.HIGH_RISK
        elif risk_score >= 40:
            risk_level = RiskLevel.MEDIUM_RISK
        elif risk_score >= 20:
            risk_level = RiskLevel.LOW_RISK
        else:
            risk_level = RiskLevel.NO_RISK
        
        # 키워드 분석도 함께 진행
        keyword_result = analyze_text_keywords(text)
        
        # 키워드 및 ML 결과를 결합
        return {
            "risk_score": risk_score,
            "risk_level": risk_level,
            "ml_confidence": float(suicide_prob),
            "keywords_found": keyword_result["keywords_found"],
            "analysis_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "method": "ml"
        }
    except Exception as e:
        logger.error(f"ML 분석 실패: {e}, 키워드 분석으로 폴백")
        # 실패 시 키워드 분석으로 폴백
        return analyze_text_keywords(text)

# 루트 엔드포인트 
@app.get("/")
def read_root():
    return {"message": "자살 위험도 분석 API - 경량 ML 버전", "status": "online"}

# 상태 확인 엔드포인트
@app.get("/status")
def check_status():
    global ml_model_loaded
    return {
        "status": "running",
        "port": os.getenv("PORT", 8000),
        "version": "minimalml",
        "environment": os.getenv("ENVIRONMENT", "development"),
        "memory_usage": "medium",
        "ml_model_loaded": ml_model_loaded,
        "features": ["ml_analysis", "keyword_detection"]
    }

# ML 모델 사전 로드 엔드포인트
@app.post("/load-model")
def load_model():
    """ML 모델 명시적 로드 엔드포인트"""
    success = load_ml_model()
    return {"success": success, "model_loaded": ml_model_loaded}

# 통합 분석 엔드포인트
@app.post("/analyze", response_model=AnalysisResult)
def analyze_text(text_input: TextInput):
    try:
        if not text_input.text or len(text_input.text.strip()) < 3:
            raise HTTPException(status_code=400, detail="Text too short for analysis")
        
        # ML 기반 분석 시도 (실패 시 키워드 분석으로 폴백)
        result = analyze_text_ml(text_input.text)
        return result
    except Exception as e:
        logger.error(f"분석 오류: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# 에러 처리 미들웨어
@app.middleware("http")
async def errors_handling(request: Request, call_next):
    try:
        return await call_next(request)
    except Exception as exc:
        logger.error(f"요청 오류: {exc}")
        return JSONResponse(
            status_code=500,
            content={"detail": str(exc)},
        )

# 서버 시작 함수
def start_server():
    """API 서버 실행"""
    port = int(os.getenv("PORT", 8000))
    logger.info(f"서버 시작: 포트 {port}...")
    
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=port,
        log_level="info"
    )

if __name__ == "__main__":
    logger.info("MinimalML API 시작")
    start_server() 