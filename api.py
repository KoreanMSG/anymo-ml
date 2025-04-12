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
import json

# Import SuicidePredictor
from suicide_predictor import SuicidePredictor
from sentiment_analyzer import SentimentAnalyzer
from conversation_processor import ConversationProcessor

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
sentiment_analyzer = None
conversation_processor = None
extract_data_dir = None

def create_sample_csv():
    """Create a sample CSV file if it doesn't exist"""
    sample_file_path = os.path.join(DATA_DIR, "Suicide_Detection_sample.csv")
    if not os.path.exists(sample_file_path):
        logger.info("Creating sample CSV file for keyword extraction")
        try:
            with open(sample_file_path, 'w', encoding='utf-8') as f:
                # Simple CSV format with carefully escaped quotes - all on one line to avoid unexpected line breaks
                f.write('text,class\n')
                # High risk examples (class 1)
                f.write('"I feel so hopeless and worthless I just want to end it all",1\n')
                f.write('"I cannot see a reason to continue living anymore",1\n')
                f.write('"I have been thinking about killing myself lately",1\n')
                f.write('"Everyone would be better off if I was not here",1\n')
                f.write('"I wish I could just go to sleep and never wake up",1\n')
                f.write('"I am tired of living with this pain every day",1\n')
                f.write('"I do not see a future for myself",1\n')
                f.write('"There is no point in living anymore",1\n')
                f.write('"I have been researching ways to commit suicide",1\n')
                f.write('"I feel like a burden to everyone around me",1\n')
                # Low risk examples (class 0)
                f.write('"I had a great day today",0\n')
                f.write('"Just finished a new book",0\n')
                f.write('"Looking forward to the weekend",0\n')
                f.write('"I am excited about my new job opportunity",0\n')
                f.write('"The weather is beautiful today",0\n')
                f.write('"Just adopted a new puppy",0\n')
                f.write('"Had a wonderful dinner with family",0\n')
                f.write('"Working on a new project",0\n')
                f.write('"I am grateful for support from friends",0\n')
                f.write('"Just got back from vacation",0\n')
            
            logger.info(f"Sample CSV file created at {sample_file_path}")
            return True
        except Exception as e:
            logger.error(f"Failed to create sample CSV file: {e}")
            return False
    return True

def startup_status():
    """Return the status of the API startup"""
    return {
        "models_loaded": models_loaded,
        "suicide_predictor_loaded": suicide_predictor is not None,
        "sentiment_analyzer_loaded": sentiment_analyzer is not None,
        "conversation_processor_loaded": conversation_processor is not None,
        "extracted_data_path": extract_data_dir,
        "extracted_data_loaded": os.path.exists(extract_data_dir) if extract_data_dir else False,
        "training_data_path": os.getenv("TRAINING_DATA_PATH", ""),
        "training_data_loaded": os.path.exists(os.getenv("TRAINING_DATA_PATH", "")) if os.getenv("TRAINING_DATA_PATH", "") else False
    }

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Lifespan event handler that initializes resources on startup
    and cleans up on shutdown.
    """
    global models_loaded, suicide_predictor, sentiment_analyzer, conversation_processor, extract_data_dir
    
    # Initialize models and resources
    logger.info("Initializing API resources...")
    
    # Create directories if they don't exist
    os.makedirs(DATA_DIR, exist_ok=True)
    os.makedirs(MODELS_DIR, exist_ok=True)
    
    # Set up NLTK data directory
    nltk_data_dir = os.path.join(os.path.dirname(__file__), 'nltk_data')
    os.makedirs(nltk_data_dir, exist_ok=True)
    
    # Initialize models
    suicide_predictor = SuicidePredictor(model_path=os.path.join(MODELS_DIR, 'suicide_model.joblib'))
    sentiment_analyzer = SentimentAnalyzer()
    conversation_processor = ConversationProcessor()
    
    # Extract data path
    extract_data_dir = os.path.join(DATA_DIR, "Suicide_Detection_sample.csv")
    
    # Create sample data file if it doesn't exist
    if not os.path.exists(extract_data_dir) and not create_sample_csv():
        logger.warning("Failed to create sample CSV, using default keywords")
    
    # Check if the original large CSV file exists and use it for training
    full_csv_path = os.path.join(os.path.dirname(__file__), 'Suicide_Detection.csv')
    if os.path.exists(full_csv_path):
        logger.info("Found original large CSV file, using it for training with chunked processing")
        success = suicide_predictor.train_from_large_csv(full_csv_path, chunk_size=10000)
        if not success:
            logger.warning("Failed to train from large CSV file, falling back to sample CSV")
            # If large CSV processing failed, try the sample CSV
            if os.path.exists(extract_data_dir):
                logger.info("Using sample CSV for keyword extraction")
                suicide_predictor.extract_keywords_from_csv(extract_data_dir)
    else:
        # If large CSV doesn't exist, use sample CSV
        if os.path.exists(extract_data_dir):
            logger.info("Using sample CSV for keyword extraction")
            success = suicide_predictor.extract_keywords_from_csv(extract_data_dir)
            
            # If extraction failed, recreate the CSV file and try again
            if not success:
                logger.warning("CSV extraction failed, creating new sample CSV file")
                os.remove(extract_data_dir)
                create_sample_csv()
                if os.path.exists(extract_data_dir):
                    logger.info("Trying extraction again with new CSV file")
                    suicide_predictor.extract_keywords_from_csv(extract_data_dir)
        else:
            logger.warning("No CSV file available, using default keywords")
    
    # Set startup flag
    models_loaded = True
    
    logger.info("API startup complete")
    logger.info(f"Startup status: {json.dumps(startup_status(), indent=2)}")
    
    yield
    
    # Cleanup on shutdown
    logger.info("Shutting down API...")
    
    # Resources will be cleaned up by Python's garbage collection
    
    models_loaded = False
    logger.info("API shutdown complete")

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

@app.get("/load-full-model", response_model=dict)
async def load_full_model():
    """
    Load and train the model using the full original CSV file.
    This is a resource-intensive operation that processes the data in chunks.
    """
    try:
        full_csv_path = os.path.join(os.path.dirname(__file__), 'Suicide_Detection.csv')
        
        if not os.path.exists(full_csv_path):
            return JSONResponse(
                status_code=404,
                content={"error": "Original CSV file not found", "success": False}
            )
        
        # Train from the full CSV file using chunks
        chunk_size = int(os.getenv("CHUNK_SIZE", "5000"))
        max_chunks = int(os.getenv("MAX_CHUNKS", "20"))
        
        # Set max_chunks to None if 0 to process all chunks
        if max_chunks <= 0:
            max_chunks = None
            
        logger.info(f"Loading full model from {full_csv_path} with chunk_size={chunk_size}, max_chunks={max_chunks}")
        
        start_time = time.time()
        success = suicide_predictor.train_from_large_csv(
            csv_filepath=full_csv_path,
            chunk_size=chunk_size,
            max_chunks=max_chunks
        )
        end_time = time.time()
        elapsed_time = end_time - start_time
        
        if success:
            logger.info(f"Full model loaded successfully in {elapsed_time:.2f} seconds")
            return {
                "message": "Full model loaded successfully",
                "elapsed_time_seconds": round(elapsed_time, 2),
                "success": True
            }
        else:
            logger.error("Failed to load full model")
            return JSONResponse(
                status_code=500,
                content={"error": "Failed to load full model", "success": False}
            )
    except Exception as e:
        logger.error(f"Error loading full model: {e}")
        return JSONResponse(
            status_code=500,
            content={"error": f"Error loading full model: {str(e)}", "success": False}
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