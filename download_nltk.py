import os
import nltk
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def main():
    """Download NLTK resources locally"""
    
    # NLTK data directory setup
    nltk_data_dir = os.path.join(os.path.dirname(__file__), 'nltk_data')
    os.makedirs(nltk_data_dir, exist_ok=True)
    
    # Set NLTK data path
    nltk.data.path.append(nltk_data_dir)
    
    # List of resources to download
    resources = [
        'punkt', 
        'stopwords', 
        'wordnet'
    ]
    
    # Download each resource
    for resource in resources:
        try:
            logger.info(f"Downloading NLTK resource: {resource}")
            nltk.download(resource, download_dir=nltk_data_dir)
            logger.info(f"Successfully downloaded {resource}")
        except Exception as e:
            logger.error(f"Failed to download {resource}: {str(e)}")
    
    # Verify downloads
    for resource in resources:
        try:
            nltk.data.find(f'tokenizers/{resource}' if resource == 'punkt' else f'corpora/{resource}')
            logger.info(f"Successfully verified {resource}")
        except LookupError:
            logger.error(f"Failed to verify {resource}")
    
    logger.info(f"NLTK resources downloaded to {nltk_data_dir}")
    
if __name__ == "__main__":
    main() 