import os
import sys
import nltk
import logging
import ssl
import shutil
from pathlib import Path

# Set up logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def main():
    """
    Download NLTK resources locally, designed to work in Render deployment.
    Includes fallbacks for SSL certificate issues and offline installation.
    """
    # NLTK data directory setup - use a writable path for Render
    script_dir = os.path.dirname(os.path.abspath(__file__))
    nltk_data_dir = os.path.join(script_dir, 'nltk_data')
    os.makedirs(nltk_data_dir, exist_ok=True)
    logger.info(f"NLTK data directory: {nltk_data_dir}")
    
    # Set NLTK data path
    nltk.data.path.append(nltk_data_dir)
    
    # List of resources to download
    resources = [
        'punkt', 
        'stopwords', 
        'wordnet'
    ]
    
    # First try: standard download
    ssl_context = None
    try:
        # Try downloading with default SSL settings
        for resource in resources:
            try:
                logger.info(f"Downloading NLTK resource: {resource}")
                nltk.download(resource, download_dir=nltk_data_dir)
                logger.info(f"Successfully downloaded {resource}")
            except Exception as e:
                logger.error(f"Failed standard download of {resource}: {str(e)}")
    except Exception as e:
        logger.error(f"Error in standard download mode: {str(e)}")
    
    # Second try: download with SSL verification disabled
    if not all_resources_available(resources, nltk_data_dir):
        logger.info("Trying download with SSL verification disabled")
        try:
            # Create unverified SSL context
            try:
                _create_unverified_https_context = ssl._create_unverified_context
            except AttributeError:
                pass
            else:
                ssl._create_default_https_context = _create_unverified_https_context
                
            # Try downloading with SSL verification disabled
            for resource in resources:
                if not resource_exists(resource, nltk_data_dir):
                    try:
                        logger.info(f"Downloading NLTK resource (SSL disabled): {resource}")
                        nltk.download(resource, download_dir=nltk_data_dir)
                        logger.info(f"Successfully downloaded {resource} with SSL disabled")
                    except Exception as e:
                        logger.error(f"Failed SSL-disabled download of {resource}: {str(e)}")
        except Exception as e:
            logger.error(f"Error in SSL-disabled download mode: {str(e)}")
    
    # Verify downloads and check what's missing
    missing_resources = []
    for resource in resources:
        if resource_exists(resource, nltk_data_dir):
            logger.info(f"Resource available: {resource}")
        else:
            logger.error(f"Resource still missing: {resource}")
            missing_resources.append(resource)
    
    # Create fallback data and structures if needed
    if missing_resources:
        logger.info("Creating fallback data structures for missing resources")
        create_fallback_resources(missing_resources, nltk_data_dir)
    
    logger.info(f"NLTK resources setup complete in {nltk_data_dir}")
    
def resource_exists(resource, nltk_data_dir):
    """Check if a resource exists in the nltk_data directory"""
    try:
        if resource == 'punkt':
            return os.path.exists(os.path.join(nltk_data_dir, 'tokenizers', 'punkt'))
        elif resource == 'stopwords':
            return os.path.exists(os.path.join(nltk_data_dir, 'corpora', 'stopwords'))
        elif resource == 'wordnet':
            return os.path.exists(os.path.join(nltk_data_dir, 'corpora', 'wordnet'))
        return False
    except Exception as e:
        logger.error(f"Error checking if resource {resource} exists: {str(e)}")
        return False

def all_resources_available(resources, nltk_data_dir):
    """Check if all resources are available"""
    return all(resource_exists(resource, nltk_data_dir) for resource in resources)

def create_fallback_resources(missing_resources, nltk_data_dir):
    """Create minimal fallback structures for missing resources"""
    for resource in missing_resources:
        try:
            if resource == 'punkt':
                # Create minimal punkt directory structure
                punkt_dir = os.path.join(nltk_data_dir, 'tokenizers', 'punkt')
                os.makedirs(punkt_dir, exist_ok=True)
                # Create an empty PY3 pickle file
                with open(os.path.join(punkt_dir, 'english.pickle'), 'wb') as f:
                    f.write(b'')
                logger.info(f"Created fallback punkt structure")
                
            elif resource == 'stopwords':
                # Create stopwords directory
                stopwords_dir = os.path.join(nltk_data_dir, 'corpora', 'stopwords')
                os.makedirs(stopwords_dir, exist_ok=True)
                # Create a basic English stopwords file
                with open(os.path.join(stopwords_dir, 'english'), 'w') as f:
                    f.write('\n'.join([
                        'i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves',
                        'you', 'your', 'yours', 'yourself', 'yourselves', 
                        'he', 'him', 'his', 'himself', 'she', 'her', 'hers', 'herself',
                        'it', 'its', 'itself', 'they', 'them', 'their', 'theirs', 'themselves',
                        'what', 'which', 'who', 'whom', 'this', 'that', 'these', 'those',
                        'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
                        'have', 'has', 'had', 'having', 'do', 'does', 'did', 'doing',
                        'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as',
                        'until', 'while', 'of', 'at', 'by', 'for', 'with', 'about',
                        'against', 'between', 'into', 'through', 'during', 'before',
                        'after', 'above', 'below', 'to', 'from', 'up', 'down', 'in',
                        'out', 'on', 'off', 'over', 'under', 'again', 'further', 'then',
                        'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all',
                        'any', 'both', 'each', 'few', 'more', 'most', 'other', 'some',
                        'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so', 'than',
                        'too', 'very', 's', 't', 'can', 'will', 'just', 'don', 'should', 'now'
                    ]))
                logger.info(f"Created fallback stopwords file")
                
            elif resource == 'wordnet':
                # Create minimal wordnet structure
                wordnet_dir = os.path.join(nltk_data_dir, 'corpora', 'wordnet')
                os.makedirs(wordnet_dir, exist_ok=True)
                # Create minimal files
                with open(os.path.join(wordnet_dir, 'index.noun'), 'w') as f:
                    f.write('')
                with open(os.path.join(wordnet_dir, 'index.verb'), 'w') as f:
                    f.write('')
                logger.info(f"Created fallback wordnet structure")
                
        except Exception as e:
            logger.error(f"Error creating fallback for {resource}: {str(e)}")

if __name__ == "__main__":
    main() 