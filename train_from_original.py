import os
import logging
import argparse
import time
from dotenv import load_dotenv
from suicide_predictor import SuicidePredictor

# Load environment variables
load_dotenv()

# Set up logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def main():
    """
    Script to train the suicide prediction model from the original large CSV file in chunks.
    This allows for processing large datasets without running into memory issues.
    Designed to work both locally and in Render deployment.
    """
    parser = argparse.ArgumentParser(description='Train suicide prediction model from large CSV file')
    
    # Default values from environment variables with fallbacks
    default_csv = os.getenv('CSV_PATH', 'Suicide_Detection.csv')
    default_chunk_size = int(os.getenv('CHUNK_SIZE', '5000'))
    default_max_chunks = int(os.getenv('MAX_CHUNKS', '20'))
    default_model_path = os.getenv('MODEL_PATH', 'models/suicide_model.joblib')
    
    parser.add_argument('--csv_path', type=str, default=default_csv,
                        help=f'Path to the large CSV file (default: {default_csv})')
    parser.add_argument('--chunk_size', type=int, default=default_chunk_size,
                        help=f'Number of rows to process in each chunk (default: {default_chunk_size})')
    parser.add_argument('--max_chunks', type=int, default=default_max_chunks,
                        help=f'Maximum number of chunks to process (default: {default_max_chunks}, use 0 for all chunks)')
    parser.add_argument('--model_path', type=str, default=default_model_path,
                        help=f'Path to save the trained model (default: {default_model_path})')
    args = parser.parse_args()

    # Log environment information
    logger.info(f"Environment: {os.getenv('ENVIRONMENT', 'development')}")
    logger.info(f"Current directory: {os.getcwd()}")
    
    # Check if CSV file exists
    csv_filepath = args.csv_path
    if not os.path.exists(csv_filepath):
        # Try to find it relative to the script location
        script_dir = os.path.dirname(os.path.abspath(__file__))
        alternative_path = os.path.join(script_dir, csv_filepath)
        if os.path.exists(alternative_path):
            csv_filepath = alternative_path
            logger.info(f"Found CSV at alternative path: {csv_filepath}")
        else:
            # Try standard locations
            for location in ['data', '.', '..']:
                potential_path = os.path.join(location, os.path.basename(csv_filepath))
                if os.path.exists(potential_path):
                    csv_filepath = potential_path
                    logger.info(f"Found CSV at: {csv_filepath}")
                    break
            else:
                logger.error(f"CSV file not found: {args.csv_path}")
                return False

    # Create model directory if it doesn't exist
    model_dir = os.path.dirname(args.model_path)
    if model_dir:  # Only create if there's a directory component
        os.makedirs(model_dir, exist_ok=True)
        logger.info(f"Ensured model directory exists: {model_dir}")

    # Initialize the predictor
    logger.info(f"Initializing suicide predictor with model path: {args.model_path}")
    predictor = SuicidePredictor(model_path=args.model_path)

    # Train from large CSV file in chunks
    logger.info(f"Starting training from {csv_filepath} with chunk size {args.chunk_size}")
    
    start_time = time.time()
    
    # Convert max_chunks=0 to None (process all chunks)
    max_chunks = args.max_chunks if args.max_chunks > 0 else None
    
    success = predictor.train_from_large_csv(
        csv_filepath=csv_filepath,
        chunk_size=args.chunk_size,
        max_chunks=max_chunks
    )
    
    end_time = time.time()
    elapsed_time = end_time - start_time
    
    if success:
        logger.info(f"Training completed successfully in {elapsed_time:.2f} seconds")
        logger.info(f"Model saved to {args.model_path}")
        
        # Test the model with a sample text
        test_texts = [
            "I feel so hopeless and worthless, I just want to end it all",
            "I've been feeling down lately, but I'm trying to stay positive",
            "I want to kill myself, there's no point in living anymore",
            "Had a great day today, everything is looking up!"
        ]
        
        logger.info("Model test results:")
        for text in test_texts:
            result = predictor.predict(text)
            
            logger.info("-" * 50)
            logger.info(f"Text: {text}")
            logger.info(f"Risk level: {result['risk_level']}")
            logger.info(f"Risk score: {result['risk_score']}")
            logger.info(f"Model used: {result['model_used']}")
            
            # Print top keywords found (limit to 5 for readability)
            keywords = result['keywords_found']
            if keywords:
                top_keywords = keywords[:5]
                logger.info(f"Top keywords: {', '.join([k['word'] for k in top_keywords])}")
                
        return True
    else:
        logger.error("Training failed")
        return False

if __name__ == "__main__":
    main() 