#!/usr/bin/env python
import os
import sys
import importlib
import subprocess
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger()

def check_package_version(package_name):
    """Check if a package is installed and report its version"""
    try:
        module = importlib.import_module(package_name)
        version = getattr(module, '__version__', 'unknown')
        logger.info(f"{package_name}: INSTALLED (version: {version})")
        return True
    except ImportError:
        logger.error(f"{package_name}: NOT INSTALLED")
        return False

def install_package(package_name, version=None):
    """Install a package with pip"""
    try:
        package_spec = f"{package_name}=={version}" if version else package_name
        logger.info(f"Installing {package_spec}...")
        
        subprocess.check_call([sys.executable, "-m", "pip", "install", package_spec])
        logger.info(f"Successfully installed {package_spec}")
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"Failed to install {package_name}: {e}")
        return False

def check_nltk_resources():
    """Check if NLTK resources are available"""
    try:
        import nltk
        nltk_data_dir = os.path.join(os.path.dirname(__file__), 'nltk_data')
        os.makedirs(nltk_data_dir, exist_ok=True)
        
        nltk.data.path.append(nltk_data_dir)
        
        resources = ['punkt', 'stopwords', 'wordnet']
        for resource in resources:
            try:
                nltk.data.find(f'tokenizers/{resource}' if resource == 'punkt' else f'corpora/{resource}')
                logger.info(f"NLTK {resource}: FOUND")
            except LookupError:
                logger.warning(f"NLTK {resource}: NOT FOUND")
                # Download the resource
                nltk.download(resource, download_dir=nltk_data_dir)
                logger.info(f"Downloaded NLTK {resource}")
        
        return True
    except Exception as e:
        logger.error(f"Error checking NLTK resources: {e}")
        return False

def check_fastapi_dependencies():
    """Check FastAPI and its dependencies"""
    dependencies = {
        'fastapi': None, 
        'uvicorn': None,
        'python-multipart': '0.0.7',
        'pydantic': None,
        'starlette': None
    }
    
    all_installed = True
    for package, version in dependencies.items():
        if not check_package_version(package.replace('-', '_')):
            all_installed = False
            if version:
                install_package(package, version)
            else:
                install_package(package)
    
    return all_installed

def main():
    """Main function to check all deployment dependencies"""
    logger.info("Checking deployment dependencies...")
    
    # Check Python version
    python_version = sys.version
    logger.info(f"Python version: {python_version}")
    
    # Check key dependencies
    check_fastapi_dependencies()
    
    # Check NLTK resources
    check_nltk_resources()
    
    # Create required directories
    for directory in ['data', 'models']:
        os.makedirs(directory, exist_ok=True)
        logger.info(f"Ensured directory exists: {directory}")
    
    logger.info("Dependency check completed")

if __name__ == "__main__":
    main() 