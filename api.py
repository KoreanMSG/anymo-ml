import os
import logging
import uvicorn
from fastapi import FastAPI, HTTPException, Body, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
from enum import Enum
from dotenv import load_dotenv
import traceback
import time
from datetime import datetime
from contextlib import asynccontextmanager
import sys

# Import SuicidePredictor
from suicide_predictor import SuicidePredictor

# Define RiskLevel enum
class RiskLevel(str, Enum):
    HIGH_RISK = "high_risk"
    MEDIUM_RISK = "medium_risk"
    LOW_RISK = "low_risk"
    NO_RISK = "no_risk"

# Environment setup
load_dotenv()
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Path setup
DATA_DIR = os.path.join(os.path.dirname(__file__), 'data')
MODELS_DIR = os.path.join(os.path.dirname(__file__), 'models')

# Directory creation
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)

# Initialize model variables
models_loaded = False
suicide_predictor = None

def create_sample_csv():
    """Create a sample CSV file if it doesn't exist"""
    sample_file_path = os.path.join(DATA_DIR, "Suicide_Detection_sample.csv")
    if not os.path.exists(sample_file_path):
        logger.info("Creating sample CSV file for keyword extraction")
        try:
            with open(sample_file_path, 'w', encoding='utf-8') as f:
                # Simple CSV format with properly escaped quotes
                f.write('''text,class
"I feel so hopeless and worthless I just want to end it all",1
"I cant see a reason to continue living anymore. The pain is too much",1
"I have been thinking about killing myself a lot lately",1
"Sometimes I feel like everyone would be better off if I wasnt here",1
"I wish I could just go to sleep and never wake up",1
"I am tired of living with this pain every day",1
"I dont see a future for myself. I just want it to end",1
"There is no point in living anymore",1
"I have been researching ways to commit suicide",1
"I feel like a burden to everyone around me",1
"I had a great day today! Everything is going well",0
"Just finished a new book it was really inspiring",0
"Looking forward to the weekend and seeing friends",0
"I am excited about my new job opportunity",0
"The weather is beautiful today perfect for a walk",0
"Just adopted a new puppy so happy",0
"Had a wonderful dinner with my family tonight",0
"Working on a new project that is really challenging but fun",0
"I am grateful for all the support from my friends lately",0
"Just got back from an amazing vacation",0''')
            logger.info(f"Sample CSV file created at {sample_file_path}")
            return True
        except Exception as e:
            logger.error(f"Failed to create sample CSV file: {e}")
            return False
    return True

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan manager for the FastAPI app"""
    global models_loaded, suicide_predictor
    
    # Initialize Suicide Predictor
    suicide_predictor = SuicidePredictor(model_path=os.path.join(MODELS_DIR, 'suicide_model.joblib'))
    
    # Ensure sample CSV exists
    create_sample_csv()
    
    # CSV-based keyword extraction - load model only when needed
    extract_data_dir = os.path.join(DATA_DIR, "Suicide_Detection_sample.csv")
    if os.path.exists(extract_data_dir):
        logger.info("Starting keyword extraction from CSV data")
        success = suicide_predictor.extract_keywords_from_csv(extract_data_dir)
        
        # If extraction failed, recreate the CSV file and try again
        if not success:
            logger.warning("CSV file extraction failed, creating new sample CSV file")
            os.remove(extract_data_dir)
            create_sample_csv()
            if os.path.exists(extract_data_dir):
                logger.info("Trying extraction again with new CSV file")
                suicide_predictor.extract_keywords_from_csv(extract_data_dir)
    else:
        logger.warning("No CSV file for keyword extraction. Using default keywords.")
        
    # Flag that startup completed
    models_loaded = True
    logger.info("API startup completed. Models will be loaded on demand.")
    
    yield
    
    # Cleanup code here (if needed)
    logger.info("Shutting down API")

# Initialize FastAPI app
app = FastAPI(
    title="Suicide Risk Analysis API",
    description="API for analyzing suicide risk in text conversations",
    version="1.0.0",
    lifespan=lifespan
)

# CORS setup
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins (restrict in production)
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Model classes
class TextInput(BaseModel):
    text: str = Field(..., description="Text to analyze")

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
    final_risk_score: int = Field(..., description="Final risk score (1-100)")

# Root endpoint (defined early for initial connection testing)
@app.get("/")
def read_root():
    return {"message": "Suicide Risk Analysis API"}

# Error handling middleware
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

# Status check endpoint
@app.get("/status")
def check_status():
    return {
        "status": "running",
        "models_loaded": models_loaded,
        "data_dir_exists": os.path.exists(DATA_DIR),
        "models_dir_exists": os.path.exists(MODELS_DIR),
        "data_files": os.listdir(DATA_DIR) if os.path.exists(DATA_DIR) else []
    }

# Conversation processing endpoint
@app.post("/process-conversation", response_model=ConversationResult)
def process_conversation(text_input: TextInput):
    if not models_loaded:
        raise HTTPException(status_code=503, detail="Models not loaded")
    
    try:
        # This endpoint is a placeholder - we don't have a conversation processor
        # Return a dummy response instead
        return {
            "processed_text": text_input.text,
            "starts_with_doctor": False
        }
    except Exception as e:
        logger.error(f"Error in processing conversation: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# ML prediction endpoint
@app.post("/predict-suicide", response_model=Dict[str, Any])
def predict_suicide(text_input: TextInput):
    if not models_loaded or suicide_predictor is None:
        raise HTTPException(status_code=503, detail="Models not loaded")
    
    try:
        # Use model if available, otherwise fall back to keyword-based
        use_model = suicide_predictor.model is not None
        result = suicide_predictor.predict(text_input.text, use_model)
        
        # Format for API response
        return {
            "risk_score": result["risk_score"],
            "is_suicide_risk": result["risk_level"] in ["medium", "high"],
            "raw_probability": 0.0 if not result.get("model_used") else result["risk_score"] / 100,
            "keywords_found": [kw["word"] for kw in result["keywords_found"]],
            "risk_level": result["risk_level"]
        }
    except Exception as e:
        logger.error(f"Error in suicide prediction: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Sentiment analysis endpoint
@app.post("/analyze-sentiment", response_model=Dict[str, Any])
def analyze_sentiment(text_input: TextInput):
    if not models_loaded or suicide_predictor is None:
        raise HTTPException(status_code=503, detail="Models not loaded")
    
    try:
        # We don't have a separate sentiment analyzer, so use the suicide_predictor's keywords
        result = suicide_predictor.predict(text_input.text, use_model=False)
        
        # Return a simplified response
        return {
            "risk_score": result["risk_score"],
            "risk_level": result["risk_level"],
            "keyword_matches": [kw["word"] for kw in result["keywords_found"]],
            "negative_sentiment": result["risk_score"] / 100  # Normalize to 0-1
        }
    except Exception as e:
        logger.error(f"Error in sentiment analysis: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Integrated analysis endpoint
@app.post("/analyze")
def analyze_text(text_input: TextInput):
    if not models_loaded:
        raise HTTPException(status_code=503, detail="API initializing, please try again in a moment")
    
    try:
        # Suicide risk analysis (try keyword-based first)
        start_time = time.time()
        result = suicide_predictor.predict(text_input.text)
        
        # Risk level conversion
        risk_level = _convert_risk_level(result["risk_level"])
        
        # Extract keyword list
        keywords_found = [kw["word"] for kw in result["keywords_found"]]
        
        # Generate analysis result
        analysis_result = {
            "risk_score": result["risk_score"],
            "risk_level": risk_level,
            "ml_confidence": 0.0 if not result.get("model_used") else 0.7,  # Keyword-based is 0, model-based is 0.7
            "keywords_found": keywords_found,
            "analysis_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "method": "keyword" if not result.get("model_used") else "ml+keyword",
            "emotion": "unknown",
            "is_conversation": False,
            "processing_time_ms": int((time.time() - start_time) * 1000)
        }
        
        return analysis_result
    except Exception as e:
        logger.error(f"Analysis error: {e}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))

def _convert_risk_level(level):
    """Risk level conversion function"""
    if level == "high":
        return RiskLevel.HIGH_RISK.value
    elif level == "medium":
        return RiskLevel.MEDIUM_RISK.value
    elif level == "low":
        return RiskLevel.LOW_RISK.value
    else:
        return RiskLevel.NO_RISK.value

# Model training endpoint
@app.post("/train")
def train_model(csv_path: str = Body(..., embed=True)):
    if not models_loaded:
        raise HTTPException(status_code=503, detail="Models not loaded")
    
    try:
        if not os.path.exists(csv_path):
            # Try relative path
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

# Server run function modified to directly pass app
def start_server():
    """API server run"""
    # Important: Run before logging output
    port = int(os.getenv("PORT", 8000))
    logger.info("Starting server...")
    logger.info(f"Port: {port}")
    logger.info(f"Current directory: {os.getcwd()}")
    
    # Directly pass FastAPI app to Uvicorn execution
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=port,
        log_level="info"
    )

if __name__ == "__main__":
    try:
        # Start log addition
        logger.info("API script started")
        logger.info(f"Python version: {sys.version}")
        logger.info(f"Environment variables: PORT={os.getenv('PORT', 'not set')}")
        start_server()
    except Exception as e:
        logger.error(f"Error starting server: {e}")
        logger.error(traceback.format_exc())
        sys.exit(1) 