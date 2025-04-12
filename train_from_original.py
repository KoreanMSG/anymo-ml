import os
import logging
import argparse
import time
from suicide_predictor import SuicidePredictor

# Set up logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def main():
    """
    Script to train the suicide prediction model from the original large CSV file in chunks.
    This allows for processing large datasets without running into memory issues.
    """
    parser = argparse.ArgumentParser(description='Train suicide prediction model from large CSV file')
    parser.add_argument('--csv_path', type=str, default='Suicide_Detection.csv',
                        help='Path to the large CSV file (default: Suicide_Detection.csv)')
    parser.add_argument('--chunk_size', type=int, default=5000,
                        help='Number of rows to process in each chunk (default: 5000)')
    parser.add_argument('--max_chunks', type=int, default=20,
                        help='Maximum number of chunks to process (default: 20, use 0 for all chunks)')
    parser.add_argument('--model_path', type=str, default='models/suicide_model.joblib',
                        help='Path to save the trained model (default: models/suicide_model.joblib)')
    args = parser.parse_args()

    # Check if CSV file exists
    if not os.path.exists(args.csv_path):
        logger.error(f"CSV file not found: {args.csv_path}")
        return False

    # Create model directory if it doesn't exist
    os.makedirs(os.path.dirname(args.model_path), exist_ok=True)

    # Initialize the predictor
    logger.info(f"Initializing suicide predictor with model path: {args.model_path}")
    predictor = SuicidePredictor(model_path=args.model_path)

    # Train from large CSV file in chunks
    logger.info(f"Starting training from {args.csv_path} with chunk size {args.chunk_size}")
    
    start_time = time.time()
    
    # Convert max_chunks=0 to None (process all chunks)
    max_chunks = args.max_chunks if args.max_chunks > 0 else None
    
    success = predictor.train_from_large_csv(
        csv_filepath=args.csv_path,
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