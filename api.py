import os
import logging
import uvicorn
from fastapi import FastAPI, HTTPException, Body, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
from dotenv import load_dotenv
import traceback

# 환경 설정
load_dotenv()
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 경로 설정
DATA_DIR = os.path.join(os.path.dirname(__file__), 'data')
MODELS_DIR = os.path.join(os.path.dirname(__file__), 'models')

# 디렉토리 생성
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)

# 먼저 FastAPI 앱 초기화
app = FastAPI(
    title="자살 위험도 분석 API",
    description="텍스트 대화에서 자살 위험도를 분석하는 API",
    version="1.0.0"
)

# CORS 설정
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 모든 오리진 허용 (프로덕션에서는 제한해야 함)
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 모델 클래스
class TextInput(BaseModel):
    text: str = Field(..., description="분석할 텍스트")

class ConversationResult(BaseModel):
    processed_text: str
    starts_with_doctor: bool
    
class SuicideAnalysisResult(BaseModel):
    risk_score: int
    is_suicide_risk: bool
    confidence: float
    
class SentimentAnalysisResult(BaseModel):
    risk_score: int
    risk_level: str
    keyword_matches: List[str]
    negative_sentiment: float
    
class FullAnalysisResult(BaseModel):
    conversation: ConversationResult
    ml_prediction: SuicideAnalysisResult
    sentiment_analysis: SentimentAnalysisResult
    final_risk_score: int = Field(..., description="최종 위험도 점수 (1-100)")

# 루트 엔드포인트 (미리 정의하여 초기 연결 테스트를 위함)
@app.get("/")
def read_root():
    return {"message": "자살 위험도 분석 API"}

# 이제 모듈 불러오기 (오류가 나더라도 앱은 실행될 수 있게)
try:
    from conversation_processor import ConversationProcessor
    from suicide_predictor import SuicidePredictor
    from sentiment_analyzer import SentimentAnalyzer

    # 모델 인스턴스 초기화
    conversation_processor = ConversationProcessor()
    suicide_predictor = SuicidePredictor(model_path=os.path.join(MODELS_DIR, 'suicide_model.joblib'))
    sentiment_analyzer = SentimentAnalyzer()
    
    models_loaded = True
    logger.info("All models loaded successfully")
    
except Exception as e:
    logger.error(f"Error loading models: {e}")
    logger.error(traceback.format_exc())
    models_loaded = False

# 에러 처리 미들웨어
@app.middleware("http")
async def errors_handling(request: Request, call_next):
    try:
        return await call_next(request)
    except Exception as exc:
        logger.error(f"Error in request: {exc}")
        logger.error(traceback.format_exc())
        return JSONResponse(
            status_code=500,
            content={"detail": str(exc)},
        )

# 상태 확인 엔드포인트
@app.get("/status")
def check_status():
    return {
        "status": "running",
        "models_loaded": models_loaded,
        "data_dir_exists": os.path.exists(DATA_DIR),
        "models_dir_exists": os.path.exists(MODELS_DIR),
        "data_files": os.listdir(DATA_DIR) if os.path.exists(DATA_DIR) else []
    }

# 대화 처리 엔드포인트
@app.post("/process-conversation", response_model=ConversationResult)
def process_conversation(text_input: TextInput):
    if not models_loaded:
        raise HTTPException(status_code=503, detail="Models not loaded")
    
    try:
        result = conversation_processor.process_conversation(text_input.text)
        return result
    except Exception as e:
        logger.error(f"Error in processing conversation: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# ML 예측 엔드포인트
@app.post("/predict-suicide", response_model=SuicideAnalysisResult)
def predict_suicide(text_input: TextInput):
    if not models_loaded:
        raise HTTPException(status_code=503, detail="Models not loaded")
    
    try:
        # 모델이 없으면 에러 발생
        if suicide_predictor.model is None:
            raise HTTPException(status_code=503, detail="Suicide prediction model not loaded")
            
        result = suicide_predictor.predict(text_input.text)
        return {
            "risk_score": result["risk_score"],
            "is_suicide_risk": result["is_suicide_risk"],
            "confidence": result["raw_probability"]
        }
    except Exception as e:
        logger.error(f"Error in suicide prediction: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# 감정 분석 엔드포인트
@app.post("/analyze-sentiment", response_model=SentimentAnalysisResult)
def analyze_sentiment(text_input: TextInput):
    if not models_loaded:
        raise HTTPException(status_code=503, detail="Models not loaded")
    
    try:
        result = sentiment_analyzer.analyze_suicide_sentiment(text_input.text)
        return result
    except Exception as e:
        logger.error(f"Error in sentiment analysis: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# 통합 분석 엔드포인트
@app.post("/analyze", response_model=FullAnalysisResult)
def analyze_text(text_input: TextInput):
    if not models_loaded:
        raise HTTPException(status_code=503, detail="Models not loaded")
    
    try:
        # 1. 대화 처리
        conversation_result = conversation_processor.process_conversation(text_input.text)
        
        # 2. 모델이 로드되어 있는지 확인
        if suicide_predictor.model is None:
            # 파일 우선순위: 1. 환경변수 2. 샘플 파일 3. 압축 파일 4. 전체 파일
            csv_path = os.getenv("CSV_PATH")
            sample_csv_path = os.path.join(DATA_DIR, "Suicide_Detection_sample.csv")
            compressed_csv_path = os.path.join(DATA_DIR, "Suicide_Detection.csv.gz")
            full_csv_path = os.path.join(DATA_DIR, "Suicide_Detection.csv")
            
            # 환경 변수로 설정된 경로 확인
            if csv_path and os.path.exists(csv_path):
                logger.info(f"Using CSV file from environment variable: {csv_path}")
                suicide_predictor.train(csv_path)
            # 샘플 파일 확인
            elif os.path.exists(sample_csv_path):
                logger.info(f"Using sample CSV file: {sample_csv_path}")
                suicide_predictor.train(sample_csv_path)
            # 압축 파일 확인
            elif os.path.exists(compressed_csv_path):
                logger.info(f"Using compressed CSV file: {compressed_csv_path}")
                import gzip
                import pandas as pd
                # 압축 파일 읽기
                with gzip.open(compressed_csv_path, 'rt') as f:
                    df = pd.read_csv(f)
                # 임시 파일로 저장
                temp_csv_path = os.path.join(DATA_DIR, 'temp_data.csv')
                df.to_csv(temp_csv_path, index=False)
                # 모델 학습
                suicide_predictor.train(temp_csv_path)
                # 임시 파일 삭제
                if os.path.exists(temp_csv_path):
                    os.remove(temp_csv_path)
            # 전체 파일 확인
            elif os.path.exists(full_csv_path):
                logger.info(f"Using full CSV file: {full_csv_path}")
                suicide_predictor.train(full_csv_path)
            else:
                logger.error("No suicide detection CSV file found")
                # 모델 없이 계속 진행
        
        # 3. 자살 위험도 예측
        ml_result = {}
        if suicide_predictor.model is not None:
            ml_result = suicide_predictor.predict(text_input.text)
        else:
            # 모델이 없으면 기본값 사용
            ml_result = {
                "risk_score": 0,
                "is_suicide_risk": False,
                "raw_probability": 0.0
            }
        
        # 4. 감정 분석
        sentiment_result = sentiment_analyzer.analyze_suicide_sentiment(text_input.text)
        
        # 5. 최종 위험도 계산 (ML 예측과 감정 분석의 평균)
        ml_score = ml_result.get("risk_score", 0)
        sentiment_score = sentiment_result.get("risk_score", 0)
        
        # 가중치 적용: ML 모델 60%, 감정 분석 40%
        final_risk_score = int(ml_score * 0.6 + sentiment_score * 0.4)
        
        return {
            "conversation": conversation_result,
            "ml_prediction": {
                "risk_score": ml_result.get("risk_score", 0),
                "is_suicide_risk": ml_result.get("is_suicide_risk", False),
                "confidence": ml_result.get("raw_probability", 0.0)
            },
            "sentiment_analysis": sentiment_result,
            "final_risk_score": final_risk_score
        }
    except Exception as e:
        logger.error(f"Error in full analysis: {e}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))

# 모델 학습 엔드포인트
@app.post("/train")
def train_model(csv_path: str = Body(..., embed=True)):
    if not models_loaded:
        raise HTTPException(status_code=503, detail="Models not loaded")
    
    try:
        if not os.path.exists(csv_path):
            # 상대 경로로 시도
            potential_path = os.path.join(DATA_DIR, csv_path)
            if os.path.exists(potential_path):
                csv_path = potential_path
            else:
                raise HTTPException(status_code=404, detail=f"CSV file not found: {csv_path}")
            
        result = suicide_predictor.train(csv_path)
        return {
            "message": "Model trained successfully",
            "accuracy": result.get("accuracy", 0),
            "model_path": suicide_predictor.model_path
        }
    except Exception as e:
        logger.error(f"Error in training model: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# 서버 실행 함수를 수정하여 직접 app을 전달
def start_server():
    """API 서버 실행"""
    # 중요: 실행 전 로그 메시지 출력
    logger.info("Starting server...")
    logger.info(f"Port: {os.getenv('PORT', 8000)}")
    logger.info(f"Current directory: {os.getcwd()}")
    
    # 직접 FastAPI 앱 전달하는 방식으로 Uvicorn 실행
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=int(os.getenv("PORT", 8000)),
        log_level="info"
    )

if __name__ == "__main__":
    # 시작 로그 추가
    logger.info("API script started")
    start_server() 