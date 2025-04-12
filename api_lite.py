import os
import logging
import uvicorn
import re
from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
from enum import Enum
from datetime import datetime
from dotenv import load_dotenv

# Environment setup
load_dotenv()
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Path settings
current_dir = os.path.dirname(os.path.abspath(__file__))

# Initialize FastAPI app
app = FastAPI(
    title="Suicide Risk Analysis API - Lite Version",
    description="API for analyzing suicide risk in text conversations (lightweight version)",
    version="1.0.0-lite"
)

# CORS settings
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins (restrict in production)
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Define model classes
class TextInput(BaseModel):
    text: str = Field(..., description="Text to analyze")

class RiskLevel(str, Enum):
    HIGH_RISK = "high_risk"
    MEDIUM_RISK = "medium_risk"
    LOW_RISK = "low_risk"
    NO_RISK = "no_risk"

class AnalysisResult(BaseModel):
    risk_score: int = Field(..., description="Risk score (1-100)")
    risk_level: RiskLevel = Field(..., description="Risk level")
    keywords_found: List[str] = Field(default=[], description="Found risk keywords")
    analysis_time: str = Field(..., description="Analysis time")

# Suicide-related keywords (Korean and English)
SUICIDE_KEYWORDS = {
    "high": [
        "자살(suicide)", "죽고 싶다(want to die)", "목숨(life)", "목맬(hang myself)", "목을 매다(hang myself)", "죽어버리고 싶다(want to die)", "없어지고 싶다(want to disappear)", "사라지고 싶다(want to vanish)", 
        "suicide", "kill myself", "end my life", "take my life"
    ],
    "medium": [
        "우울하다(depressed)", "희망이 없다(hopeless)", "삶이 의미없다(life is meaningless)", "살아갈 이유(reason to live)", "혼자(alone)", "외롭다(lonely)", "고통(pain)", 
        "hopeless", "worthless", "alone", "lonely", "pain", "suffering", "depressed"
    ],
    "low": [
        "슬프다(sad)", "힘들다(hard/difficult)", "지쳤다(exhausted)", "피곤하다(tired)", "실패(failure)", "포기(give up)", "도움(help)", 
        "sad", "tired", "exhausted", "failed", "give up", "help me"
    ]
}

# Text preprocessing function
def preprocess_text(text):
    # Convert to lowercase
    text = text.lower()
    # Remove special characters
    text = re.sub(r'[^\w\s]', '', text)
    return text

# Function to analyze text for suicide risk
def analyze_suicide_risk(text: str) -> Dict[str, Any]:
    if not text:
        return {
            "risk_score": 0,
            "risk_level": RiskLevel.NO_RISK,
            "keywords_found": []
        }
    
    # Preprocess text
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
        risk_level = RiskLevel.HIGH_RISK
    elif score >= 60:
        risk_level = RiskLevel.MEDIUM_RISK
    elif score >= 40:
        risk_level = RiskLevel.LOW_RISK
    else:
        risk_level = RiskLevel.NO_RISK
    
    # Combine all keywords
    all_keywords = found_keywords["high"] + found_keywords["medium"] + found_keywords["low"]
    
    return {
        "risk_score": score,
        "risk_level": risk_level,
        "keywords_found": all_keywords
    }

# Root endpoint
@app.get("/")
def read_root():
    return {"message": "Suicide Risk Analysis API - Lite Version", "status": "online"}

# Status check endpoint
@app.get("/status")
def check_status():
    return {
        "status": "online",
        "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "version": "1.0.0-lite"
    }

# Analysis endpoint
@app.post("/analyze", response_model=AnalysisResult)
def analyze_text(text_input: TextInput):
    try:
        # Analyze text
        result = analyze_suicide_risk(text_input.text)
        
        # Add analysis time
        result["analysis_time"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        return result
    except Exception as e:
        logger.error(f"Analysis error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Error handling middleware
@app.middleware("http")
async def errors_handling(request: Request, call_next):
    try:
        return await call_next(request)
    except Exception as exc:
        logger.error(f"Error in request: {exc}")
        return JSONResponse(
            status_code=500,
            content={"detail": str(exc)},
        )

# Start server
def start_server():
    """Run API server"""
    port = int(os.getenv("PORT", 8000))
    logger.info("Starting server...")
    logger.info(f"Port: {port}")
    
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=port
    )

if __name__ == "__main__":
    logger.info("API script started")
    start_server() 