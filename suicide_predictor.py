import os
import pandas as pd
import numpy as np
import logging
import joblib
import gzip
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import re
from enum import Enum
from typing import Optional, List, Dict, Any

# Environment settings
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# NLTK resource download
try:
    # NLTK data directory setup (writable path for Render environment)
    nltk_data_dir = os.path.join(os.path.dirname(__file__), 'nltk_data')
    os.makedirs(nltk_data_dir, exist_ok=True)
    
    # Set NLTK data path
    nltk.data.path.append(nltk_data_dir)
    
    # Download required data
    nltk.download('punkt', download_dir=nltk_data_dir, quiet=True)
    nltk.download('stopwords', download_dir=nltk_data_dir, quiet=True)
    nltk.download('wordnet', download_dir=nltk_data_dir, quiet=True)
    
    logger.info(f"NLTK resources downloaded to {nltk_data_dir}")
except Exception as e:
    logger.warning(f"NLTK download warning: {e}")
    # Prepare for basic processing even if there are errors
    try:
        from nltk.tokenize import word_tokenize as _word_tokenize
        def word_tokenize(text):
            try:
                return _word_tokenize(text)
            except:
                return text.split()
        
        if not hasattr(nltk, 'corpus') or not hasattr(nltk.corpus, 'stopwords') or not hasattr(nltk.corpus.stopwords, 'words'):
            nltk.corpus.stopwords.words = lambda lang: []
        
        if not hasattr(nltk, 'stem') or not hasattr(nltk.stem, 'WordNetLemmatizer'):
            class DummyLemmatizer:
                def lemmatize(self, word):
                    return word
            nltk.stem.WordNetLemmatizer = DummyLemmatizer
            
        logger.info("Set up fallback NLTK functionality")
    except Exception as inner_e:
        logger.error(f"Failed to set up fallback NLTK functionality: {inner_e}")

class SuicidePredictor:
    def __init__(self, model_path='models/suicide_model.joblib'):
        self.model_path = model_path
        self.model = None
        self.vectorizer = None
        self.keywords = {
            "high": [],
            "medium": [],
            "low": []
        }
        
        # Setup fallback NLTK functionality
        try:
            self._setup_nltk_resources()
        except Exception as e:
            logger.warning(f"Could not initialize NLTK resources properly: {e}")
            self._setup_nltk_fallbacks()
        
        # Load model if it exists
        if os.path.exists(model_path):
            try:
                loaded_model = joblib.load(model_path)
                
                # Check model format
                if isinstance(loaded_model, dict) and 'vectorizer' in loaded_model and 'classifier' in loaded_model:
                    # Model created via chunked training
                    logger.info(f"Loaded chunked training model from {model_path}")
                    self.model = loaded_model
                    self.vectorizer = loaded_model['vectorizer']
                elif hasattr(loaded_model, 'predict_proba'):
                    # Standard sklearn model/pipeline
                    logger.info(f"Loaded standard model from {model_path}")
                    self.model = loaded_model
                    
                    # Try to get vectorizer from pipeline if available
                    if hasattr(loaded_model, 'named_steps') and 'tfidf' in loaded_model.named_steps:
                        self.vectorizer = loaded_model.named_steps['tfidf']
                else:
                    logger.warning(f"Unknown model format in {model_path}, may cause issues")
                    self.model = loaded_model
                
                logger.info(f"Model loaded successfully from {model_path}")
            except Exception as e:
                logger.error(f"Failed to load model: {e}")
                self.model = None
                import traceback
                logger.error(traceback.format_exc())
    
    def _setup_nltk_resources(self):
        """Set up NLTK resources with proper error handling"""
        try:
            self.lemmatizer = WordNetLemmatizer()
            self.stop_words = set(stopwords.words('english'))
            logger.info("NLTK resources initialized successfully")
        except Exception as e:
            logger.warning(f"NLTK resource initialization warning: {e}")
            raise

    def _setup_nltk_fallbacks(self):
        """Set up fallback functionality when NLTK resources aren't available"""
        logger.info("Setting up NLTK fallbacks")
        
        # Create dummy lemmatizer
        class DummyLemmatizer:
            def lemmatize(self, word):
                return word
        self.lemmatizer = DummyLemmatizer()
        
        # Create basic English stopwords
        self.stop_words = set([
            'i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', 
            'your', 'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 
            'she', 'her', 'hers', 'herself', 'it', 'its', 'itself', 'they', 'them', 
            'their', 'theirs', 'themselves', 'what', 'which', 'who', 'whom', 'this', 
            'that', 'these', 'those', 'am', 'is', 'are', 'was', 'were', 'be', 'been', 
            'being', 'have', 'has', 'had', 'having', 'do', 'does', 'did', 'doing', 
            'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until', 
            'while', 'of', 'at', 'by', 'for', 'with', 'about', 'against', 'between', 
            'into', 'through', 'during', 'before', 'after', 'above', 'below', 'to', 
            'from', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again', 
            'further', 'then', 'once', 'here', 'there', 'when', 'where', 'why', 'how', 
            'all', 'any', 'both', 'each', 'few', 'more', 'most', 'other', 'some', 'such', 
            'no', 'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very', 
            's', 't', 'can', 'will', 'just', 'don', 'should', 'now'
        ])
        
        logger.info("NLTK fallbacks initialized")

    def preprocess_text(self, text):
        """
        텍스트 전처리 함수
        
        Parameters:
        -----------
        text : str
            전처리할 원본 텍스트
        
        Returns:
        --------
        str
            전처리된 텍스트
        """
        if not text:
            return ""
        
        # 소문자 변환
        text = text.lower()
        
        # 한글과 영어, 숫자, 공백만 남기고 나머지는 제거
        text = re.sub(r'[^\w\s가-힣]', ' ', text)
        
        # 연속된 공백 제거
        text = re.sub(r'\s+', ' ', text)
        
        # 앞뒤 공백 제거
        text = text.strip()
        
        return text
    
    def extract_keywords_from_csv(self, csv_filepath):
        """Extract suicide-related keywords from CSV file."""
        try:
            logger.info(f"Loading CSV file for keyword extraction: {csv_filepath}")
            
            # First try with default settings and skipping bad lines
            try:
                df = pd.read_csv(csv_filepath, encoding='utf-8', on_bad_lines='skip')
                logger.info(f"CSV file loaded successfully: {len(df)} rows")
            except Exception as e:
                logger.warning(f"First CSV parsing attempt failed: {str(e)}, trying alternative method")
                try:
                    # Try with Python engine and minimal quoting
                    import csv
                    df = pd.read_csv(csv_filepath, encoding='utf-8', engine='python', quoting=csv.QUOTE_MINIMAL)
                    logger.info(f"CSV file loaded successfully (alternative method): {len(df)} rows")
                except Exception as e:
                    logger.warning(f"Second CSV parsing attempt failed: {str(e)}, trying final method")
                    try:
                        # Final attempt with no quoting
                        df = pd.read_csv(csv_filepath, encoding='utf-8', engine='python', quoting=csv.QUOTE_NONE, escapechar='\\')
                        logger.info(f"CSV file loaded successfully (final method): {len(df)} rows")
                    except Exception as e:
                        raise Exception(f"All CSV parsing methods failed: {str(e)}")
            
            # Check if the dataframe has at least 2 columns (text and label)
            if len(df.columns) < 2:
                logger.warning(f"CSV file needs at least 2 columns (current: {len(df.columns)})")
                return False
            
            logger.info(f"CSV columns: {', '.join(df.columns)}")
            
            # Get the column names (typically 'text' and 'class' or 'label')
            text_col = df.columns[0]  # First column is typically text
            label_col = df.columns[1]  # Second column is typically label
            
            logger.info(f"Text column: {text_col}, Label column: {label_col}")
            
            # Extract suicide-related texts (where label is 1 or 'suicide')
            suicide_texts = []
            
            # Check if label is numeric or string
            if pd.api.types.is_numeric_dtype(df[label_col]):
                suicide_mask = df[label_col] == 1
            else:
                # If label is string, check for 'suicide', '1', etc.
                suicide_mask = df[label_col].str.lower().isin(['suicide', '1', 'yes', 'true'])
            
            suicide_texts = df.loc[suicide_mask, text_col].tolist()
            
            logger.info(f"Extracted suicide-related texts: {len(suicide_texts)}")
            
            if len(suicide_texts) == 0:
                logger.warning("No suicide-related texts found. Using default keywords.")
                return False
            
            # Use TF-IDF to extract keywords
            vectorizer = TfidfVectorizer(max_features=50, stop_words='english')
            tfidf_matrix = vectorizer.fit_transform(suicide_texts)
            feature_names = vectorizer.get_feature_names_out()
            
            # Calculate average TF-IDF score for each keyword
            avg_scores = tfidf_matrix.mean(axis=0).A1
            
            # Sort keywords by score
            keywords_with_scores = list(zip(feature_names, avg_scores))
            keywords_with_scores.sort(key=lambda x: x[1], reverse=True)
            
            # Categorize keywords into high, medium, low based on scores
            total_keywords = len(keywords_with_scores)
            high_count = max(5, total_keywords // 3)
            medium_count = max(10, total_keywords // 3)
            
            self.keywords = {
                "high": [kw for kw, _ in keywords_with_scores[:high_count]],
                "medium": [kw for kw, _ in keywords_with_scores[high_count:high_count+medium_count]],
                "low": [kw for kw, _ in keywords_with_scores[high_count+medium_count:]]
            }
            
            # Log some examples from each category
            logger.info(f"High risk keywords ({len(self.keywords['high'])}): " + 
                      ', '.join(self.keywords['high'][:5]))
            logger.info(f"Medium risk keywords ({len(self.keywords['medium'])}): " + 
                      ', '.join(self.keywords['medium'][:5]))
            logger.info(f"Low risk keywords ({len(self.keywords['low'])}): " + 
                      ', '.join(self.keywords['low'][:5]))
            
            # Ensure we have at least some keywords in each category
            if not self.keywords["high"]:
                self.keywords["high"] = ["kill myself", "want to die", "end my life", "suicide"]
            if not self.keywords["medium"]:
                self.keywords["medium"] = ["hopeless", "worthless", "dead", "can't go on"]
            if not self.keywords["low"]:
                self.keywords["low"] = ["depressed", "sad", "pain", "hurt"]
            
            return True
        
        except Exception as e:
            logger.error(f"Error during keyword extraction: {str(e)}")
            
            # Set default keywords
            self.keywords = {
                "high": ["kill myself", "want to die", "end my life", "suicide", "end it all"],
                "medium": ["hopeless", "worthless", "dead", "can't go on", "no point"],
                "low": ["depressed", "sad", "pain", "hurt", "tired of"]
            }
            logger.info("Using default keywords.")
            return False
    
    def _set_default_keywords(self):
        """Set default keywords"""
        self.keywords = {
            "high": [
                "suicide", "kill myself", "end my life", "want to die", "rather be dead",
                "take my own life", "ending it all", "ending my life", "killing myself", "commit suicide",
                "overdose", "jump", "hang myself", "shoot myself", "self-harm", "slit wrists", "noose", 
                "suicidal thoughts", "better off dead", "not worth living", "die soon", "suicidal ideation",
                "lethal", "razor", "gun", "pills", "fatal", "suicide note", "final goodbye",
                "want to end it all", "take my own life", "end it all", "last day",
                "life insurance", "will and testament", "sleeping pills", "painless death"
            ],
            "medium": [
                "hopeless", "worthless", "no reason to live", "can't go on", "tired of living",
                "what's the point", "better off dead", "no future", "no hope", "give up",
                "helpless", "depressed", "miserable", "suffering", "burden to others", 
                "no purpose", "no meaning", "darkness", "unbearable pain", "emptiness", 
                "nothingness", "despair", "trapped", "agony", "anguish", "torment", 
                "exhausted", "meaningless", "pointless", "alone", "lonely", "isolated", 
                "abandoned", "rejected", "giving up", "can't cope", "too much pain", "endless suffering"
            ],
            "low": [
                "depressed", "depression", "sad", "lonely", "alone", "pain", "suffering",
                "hurt", "failure", "miserable", "lost", "empty", "upset", "unhappy", 
                "frustrated", "disappointed", "tired", "exhausted", "dissatisfied", 
                "unfulfilled", "discouraged", "disheartened", "dejected", "heartbroken", 
                "defeated", "fatigued", "weary", "drained", "spent", "troubled", "distressed", 
                "blue", "down", "gloomy", "sorrowful", "tearful", "confused", "struggling",
                "numb", "anxiety", "worry", "stressed"
            ]
        }
        logger.info("Using default keywords.")

    def _add_default_keywords(self):
        """Add default keywords to existing keywords"""
        default_keywords = {
            "high": [
                "suicide", "kill myself", "end my life", "want to die", "rather be dead",
                "take my own life", "ending it all", "ending my life", "killing myself", "commit suicide",
                "overdose", "jump", "hang myself", "shoot myself", "self-harm", "slit wrists", "noose", 
                "suicidal thoughts", "better off dead", "not worth living", "die soon", "suicidal ideation",
                "lethal", "razor", "gun", "pills", "fatal", "suicide note", "final goodbye",
                "want to end it all", "take my own life", "end it all", "last day",
                "life insurance", "will and testament", "sleeping pills", "painless death"
            ],
            "medium": [
                "hopeless", "worthless", "no reason to live", "can't go on", "tired of living",
                "what's the point", "better off dead", "no future", "no hope", "give up",
                "helpless", "depressed", "miserable", "suffering", "burden to others", 
                "no purpose", "no meaning", "darkness", "unbearable pain", "emptiness", 
                "nothingness", "despair", "trapped", "agony", "anguish", "torment", 
                "exhausted", "meaningless", "pointless", "alone", "lonely", "isolated", 
                "abandoned", "rejected", "giving up", "can't cope", "too much pain", "endless suffering"
            ],
            "low": [
                "depressed", "depression", "sad", "lonely", "alone", "pain", "suffering",
                "hurt", "failure", "miserable", "lost", "empty", "upset", "unhappy", 
                "frustrated", "disappointed", "tired", "exhausted", "dissatisfied", 
                "unfulfilled", "discouraged", "disheartened", "dejected", "heartbroken", 
                "defeated", "fatigued", "weary", "drained", "spent", "troubled", "distressed", 
                "blue", "down", "gloomy", "sorrowful", "tearful", "confused", "struggling",
                "numb", "anxiety", "worry", "stressed"
            ]
        }
        
        # Add default keywords to each risk level (removing duplicates)
        for level in ["high", "medium", "low"]:
            existing_keywords_set = set(self.keywords[level])
            for keyword in default_keywords[level]:
                if keyword not in existing_keywords_set:
                    self.keywords[level].append(keyword)

    def train_from_csv(self, csv_filepath):
        """Train suicide prediction model from CSV file."""
        try:
            logger.info(f"Training model from CSV file: {csv_filepath}")
            
            # First try with default settings and skipping bad lines
            try:
                import csv
                df = pd.read_csv(csv_filepath, encoding='utf-8', on_bad_lines='skip')
                logger.info(f"CSV file loaded successfully: {len(df)} rows")
            except Exception as e:
                logger.warning(f"First CSV parsing attempt failed: {str(e)}, trying alternative method")
                try:
                    # Try with Python engine and minimal quoting
                    df = pd.read_csv(csv_filepath, encoding='utf-8', engine='python', quoting=csv.QUOTE_MINIMAL)
                    logger.info(f"CSV file loaded successfully (alternative method): {len(df)} rows")
                except Exception as e:
                    logger.warning(f"Second CSV parsing attempt failed: {str(e)}, trying final method")
                    try:
                        # Final attempt with no quoting
                        df = pd.read_csv(csv_filepath, encoding='utf-8', engine='python', quoting=csv.QUOTE_NONE, escapechar='\\')
                        logger.info(f"CSV file loaded successfully (final method): {len(df)} rows")
                    except Exception as e:
                        raise Exception(f"All CSV parsing methods failed: {str(e)}")
            
            # Check if the dataframe has at least 2 columns (text and label)
            if len(df.columns) < 2:
                logger.warning(f"CSV file needs at least 2 columns (current: {len(df.columns)})")
                return False
            
            logger.info(f"CSV columns: {', '.join(df.columns)}")
            
            # Prepare data for model training
            X = df.iloc[:, 0].values  # First column is typically text
            y = df.iloc[:, 1].values  # Second column is typically label
            
            # Convert labels to binary if they're not already
            if not np.issubdtype(y.dtype, np.number):
                # If y is not numeric, convert values like 'suicide', 'yes', 'true' to 1, others to 0
                y = np.array([1 if str(label).lower() in ['suicide', '1', 'yes', 'true'] else 0 for label in y])
            
            # Extract keywords before training
            self.extract_keywords_from_csv(csv_filepath)
            
            # Transform the text data
            vectorizer = TfidfVectorizer(max_features=5000)
            X_transformed = vectorizer.fit_transform(X)
            
            # Train the model
            model = LogisticRegression(max_iter=1000)
            model.fit(X_transformed, y)
            
            # Save the model components
            self.vectorizer = vectorizer
            self.model = model
            
            logger.info("Model training completed")
            return True
        
        except Exception as e:
            logger.error(f"Error during model training: {str(e)}")
            return False
    
    def train(self, csv_path, test_size=0.2, random_state=42):
        """CSV 파일로부터 자살 예측 모델 학습"""
        try:
            # 데이터 로드
            logger.info(f"Loading data from {csv_path}")
            try:
                df = pd.read_csv(csv_path, on_bad_lines='skip')
                logger.info(f"Loaded CSV with {len(df)} rows after skipping bad lines")
            except TypeError:
                # Fallback for older pandas versions
                df = pd.read_csv(csv_path, error_bad_lines=False)
                logger.info(f"Loaded CSV with {len(df)} rows after skipping bad lines")
            
            # 필요한 열만 선택 (텍스트와 라벨)
            # 주의: CSV 파일의 실제 열 이름에 맞게 조정 필요
            text_column = 'text'  # 실제 텍스트 열 이름으로 변경
            label_column = 'class'  # 실제 라벨 열 이름으로 변경
            
            if text_column not in df.columns or label_column not in df.columns:
                # 열 이름 추측
                if len(df.columns) >= 2:
                    text_column = df.columns[0]
                    label_column = df.columns[1]
                    logger.warning(f"Column names not found, using {text_column} and {label_column} instead")
                else:
                    raise ValueError("CSV file does not have enough columns")
            
            logger.info("Preprocessing data...")
            
            # 텍스트 전처리
            df['processed_text'] = df[text_column].fillna('').apply(self.preprocess_text)
            
            # 라벨 변환 (필요한 경우)
            # 예: 'suicide' -> 1, 'non-suicide' -> 0
            if not pd.api.types.is_numeric_dtype(df[label_column]):
                unique_labels = df[label_column].unique()
                if len(unique_labels) == 2:
                    # 라벨이 이미 바이너리인 경우
                    if all(label in [0, 1] for label in unique_labels):
                        pass
                    else:
                        # 문자열 라벨을 숫자로 변환
                        # 'suicide'와 관련된 라벨을 1로, 나머지를 0으로
                        suicide_keywords = ['suicide', 'suicidal', 'yes', 'positive', '1', 'true']
                        df['label'] = df[label_column].apply(
                            lambda x: 1 if str(x).lower() in suicide_keywords else 0
                        )
                        label_column = 'label'
                        logger.info(f"Converted labels: {df[label_column].value_counts().to_dict()}")
                else:
                    logger.warning(f"Found {len(unique_labels)} unique labels, expected 2")
                    # Handle multi-class by converting to binary
                    suicide_keywords = ['suicide', 'suicidal', 'yes', 'positive', '1', 'true']
                    df['label'] = df[label_column].apply(
                        lambda x: 1 if str(x).lower() in suicide_keywords else 0
                    )
                    label_column = 'label'
                    logger.info(f"Converted multiple labels to binary: {df[label_column].value_counts().to_dict()}")
            
            # 데이터 분할
            X_train, X_test, y_train, y_test = train_test_split(
                df['processed_text'], 
                df[label_column],
                test_size=test_size,
                random_state=random_state,
                stratify=df[label_column]
            )
            
            # 파이프라인 생성
            logger.info("Building and training model...")
            pipeline = Pipeline([
                ('tfidf', TfidfVectorizer(max_features=10000)),
                ('clf', LogisticRegression(C=1.0, class_weight='balanced', max_iter=1000, random_state=random_state))
            ])
            
            # 모델 학습
            pipeline.fit(X_train, y_train)
            
            # 성능 평가
            y_pred = pipeline.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            report = classification_report(y_test, y_pred)
            
            logger.info(f"Model accuracy: {accuracy:.4f}")
            logger.info(f"Classification report:\n{report}")
            
            # 모델 저장
            os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
            joblib.dump(pipeline, self.model_path)
            logger.info(f"Model saved to {self.model_path}")
            
            self.model = pipeline
            return {
                'accuracy': accuracy,
                'report': report
            }
            
        except Exception as e:
            logger.error(f"Error in training model: {e}")
            raise
    
    def predict(self, text, use_model=True):
        """
        Predict suicide risk level for text
        
        Parameters:
        -----------
        text : str
            Text to analyze
        use_model : bool, default=True
            Whether to use trained model. If False, only keyword-based prediction is used
            
        Returns:
        --------
        dict
            - risk_level: Risk level ('none', 'low', 'medium', 'high')
            - risk_score: Risk score (0-100)
            - keywords_found: List of keywords found in text
            - model_used: Whether model was used
        """
        if not text or len(text.strip()) == 0:
            return {
                "risk_level": "none",
                "risk_score": 0,
                "keywords_found": [],
                "model_used": False
            }
        
        # Calculate risk based on keywords
        risk_score = self._calculate_keyword_risk(text)
        keywords_found = self._find_keywords_in_text(text)
        
        # Use trained model if available and specified
        model_used = False
        if use_model and self.model is not None:
            try:
                # Check model format
                if isinstance(self.model, dict) and 'classifier' in self.model and 'vectorizer' in self.model:
                    # Using the chunked training model format
                    vectorizer = self.model['vectorizer']
                    classifier = self.model['classifier']
                    
                    # Transform the text using the vectorizer
                    X_transformed = vectorizer.transform([text])
                    
                    # Predict using the classifier
                    prediction = classifier.predict_proba(X_transformed)[0]
                    suicide_prob = prediction[1]  # Probability of suicide class (index 1)
                    logger.debug(f"Chunked model prediction: {suicide_prob:.4f}")
                elif hasattr(self.model, 'predict_proba') and hasattr(self.model, 'named_steps'):
                    # Using the pipeline format (standard scikit-learn pipeline)
                    prediction = self.model.predict_proba([text])[0]
                    suicide_prob = prediction[1]
                    logger.debug(f"Pipeline model prediction: {suicide_prob:.4f}")
                elif hasattr(self.model, 'predict_proba'):
                    # Direct classifier (no pipeline)
                    # We need to transform the text ourselves if we have a vectorizer
                    if self.vectorizer is not None:
                        X_transformed = self.vectorizer.transform([text])
                        prediction = self.model.predict_proba(X_transformed)[0]
                    else:
                        # Try direct prediction (might fail if text format is wrong)
                        prediction = self.model.predict_proba([text])[0]
                    suicide_prob = prediction[1]
                    logger.debug(f"Direct model prediction: {suicide_prob:.4f}")
                else:
                    # Unknown model format
                    logger.warning(f"Unknown model format: {type(self.model)}. Using keyword-based prediction only.")
                    return {
                        "risk_level": self._interpret_risk_score(risk_score),
                        "risk_score": round(risk_score, 2),
                        "keywords_found": keywords_found,
                        "model_used": False
                    }
                
                # Convert model score to 100-point scale (to combine with keywords)
                model_score = suicide_prob * 100
                logger.debug(f"Model prediction: {suicide_prob:.4f}, Model score: {model_score:.2f}")
                
                # Weighted combination of keyword score and model score (model 70%, keywords 30%)
                risk_score = model_score * 0.7 + risk_score * 0.3
                model_used = True
                
                logger.debug(f"Combined risk score: {risk_score:.2f}")
            except Exception as e:
                logger.error(f"Model prediction failed: {e}")
                # Use only keyword score if model prediction fails
                import traceback
                logger.error(traceback.format_exc())
        
        # Determine final risk level
        risk_level = self._interpret_risk_score(risk_score)
        
        return {
            "risk_level": risk_level,
            "risk_score": round(risk_score, 2),
            "keywords_found": keywords_found,
            "model_used": model_used
        }
    
    def _calculate_keyword_risk(self, text):
        """키워드 기반 위험도 계산"""
        text_lower = text.lower()
        words = set(re.findall(r'\b\w+\b', text_lower))  # Extract individual words for exact matching
        
        # 결과 저장할 변수들
        high_matches = []
        medium_matches = []
        low_matches = []
        
        # 카테고리별 키워드 검사
        for keyword in self.keywords["high"]:
            if keyword.lower() in text_lower or keyword.lower() in words:
                high_matches.append(keyword)
        
        for keyword in self.keywords["medium"]:
            if keyword.lower() in text_lower or keyword.lower() in words:
                medium_matches.append(keyword)
        
        for keyword in self.keywords["low"]:
            if keyword.lower() in text_lower or keyword.lower() in words:
                low_matches.append(keyword)
        
        # 위험도 점수 계산 (100점 만점)
        high_score = min(len(high_matches) * 25, 70)  # 각 고위험 키워드당 25점, 최대 70점
        medium_score = min(len(medium_matches) * 10, 20)  # 각 중위험 키워드당 10점, 최대 20점
        low_score = min(len(low_matches) * 5, 10)  # 각 저위험 키워드당 5점, 최대 10점
        
        # 복합 키워드 패턴 검사 (n-gram 패턴)
        phrase_patterns = [
            (r"(want|wish|going) to (die|end|kill myself)", 30),  # 특정 표현 추가 점수
            (r"plan(ning)? to (suicide|kill myself|end my life)", 40),
            (r"no (reason|point|purpose) (in|to|for) (living|life)", 25),
            (r"can'?t (take|bear|handle|stand) (it|this) anymore", 20),
            (r"(considering|contemplating) (suicide|ending|taking) my life", 35),
            (r"(made|written|prepared) (plans|notes|letters) (for|to|about) (suicide|death)", 40),
            (r"(ending|taking) my (life|own life)", 35),
            (r"don'?t want to (live|be alive|exist) (anymore|any longer)", 30),
            (r"(better off|world better) without me", 25),
            (r"(going to|will) (kill myself|end it all|end my life)", 40),
            (r"(tired|exhausted|done) (of|with) (living|life|everything)", 20)
        ]
        
        pattern_score = 0
        for pattern, score in phrase_patterns:
            if re.search(pattern, text_lower):
                pattern_score += score
        
        # 최종 점수 계산 (최대 100점)
        total_score = min(high_score + medium_score + low_score + pattern_score, 100)
        
        logger.debug(f"Risk score calculation - High: {high_score}, Medium: {medium_score}, Low: {low_score}, Phrases: {pattern_score}")
        
        return total_score
    
    def _find_keywords_in_text(self, text):
        """텍스트에서 발견된 키워드 목록 반환"""
        text_lower = text.lower()
        words = set(re.findall(r'\b\w+\b', text_lower))  # 개별 단어 추출
        found_keywords = []
        
        # 모든 카테고리의 키워드 검사
        for category, weight in [("high", "High"), ("medium", "Medium"), ("low", "Low")]:
            for keyword in self.keywords[category]:
                keyword_lower = keyword.lower()
                # 개별 단어 또는 구문으로 검사
                if keyword_lower in text_lower or (len(keyword_lower.split()) == 1 and keyword_lower in words):
                    found_keywords.append({
                        "word": keyword,
                        "category": weight
                    })
        
        # 복합 키워드 패턴 검사
        phrase_patterns = [
            (r"(want|wish|going) to (die|end|kill myself)", "High", "Want to die"),
            (r"plan(ning)? to (suicide|kill myself|end my life)", "High", "Suicide plan"),
            (r"no (reason|point|purpose) (in|to|for) (living|life)", "Medium", "No reason to live"),
            (r"can'?t (take|bear|handle|stand) (it|this) anymore", "Medium", "Can't take it anymore"),
            (r"(considering|contemplating) (suicide|ending|taking) my life", "High", "Considering suicide"),
            (r"(made|written|prepared) (plans|notes|letters) (for|to|about) (suicide|death)", "High", "Suicide preparation"),
            (r"(ending|taking) my (life|own life)", "High", "Ending my life"),
            (r"don'?t want to (live|be alive|exist) (anymore|any longer)", "High", "Don't want to live"),
            (r"(better off|world better) without me", "Medium", "World better without me"),
            (r"(going to|will) (kill myself|end it all|end my life)", "High", "Will kill myself"),
            (r"(tired|exhausted|done) (of|with) (living|life|everything)", "Medium", "Tired of living")
        ]
        
        for pattern, category, description in phrase_patterns:
            if re.search(pattern, text_lower):
                found_keywords.append({
                    "word": description,
                    "category": category
                })
        
        return found_keywords
    
    def _interpret_risk_score(self, score):
        """위험도 점수 해석"""
        if score >= 65:
            return "high"
        elif score >= 35:
            return "medium"
        elif score >= 15:
            return "low"
        else:
            return "none"

    def train_from_large_csv(self, csv_filepath, chunk_size=10000, max_chunks=None):
        """
        Train suicide prediction model from a large CSV file by processing it in chunks.
        
        Parameters:
        -----------
        csv_filepath : str
            Path to the CSV file
        chunk_size : int, default=10000
            Number of rows to process in each chunk
        max_chunks : int, default=None
            Maximum number of chunks to process (None = process all)
            
        Returns:
        --------
        bool
            True if training was successful, False otherwise
        """
        try:
            logger.info(f"Training model from large CSV file in chunks: {csv_filepath}")
            
            # Initialize reader for chunked processing
            chunks_processed = 0
            total_rows_processed = 0
            
            # Prepare vectorizer and classifier
            vectorizer = TfidfVectorizer(max_features=5000)
            classifier = LogisticRegression(max_iter=1000, warm_start=True)
            
            # Variables to collect data for keyword extraction
            all_suicide_texts = []
            first_chunk = True
            X_all = None
            y_all = None
            
            # Process file in chunks
            for chunk_num, chunk in enumerate(pd.read_csv(csv_filepath, chunksize=chunk_size, 
                                                        encoding='utf-8', on_bad_lines='skip')):
                
                if max_chunks is not None and chunk_num >= max_chunks:
                    logger.info(f"Reached maximum number of chunks ({max_chunks}), stopping")
                    break
                
                logger.info(f"Processing chunk {chunk_num+1} with {len(chunk)} rows")
                
                # Check if the dataframe has at least 2 columns (text and label)
                if len(chunk.columns) < 2:
                    logger.warning(f"CSV file needs at least 2 columns (current: {len(chunk.columns)})")
                    continue
                
                # The structure is likely a bit different - check if we have an index column
                if 'text' in chunk.columns and 'class' in chunk.columns:
                    # Standard format with text and class columns
                    text_col = 'text'
                    label_col = 'class'
                else:
                    # Try to guess - first non-index column is typically text, second is label
                    cols = [col for col in chunk.columns if col.lower() != 'unnamed: 0']
                    if len(cols) >= 2:
                        text_col = cols[0]
                        label_col = cols[1]
                    else:
                        text_col = chunk.columns[0]  # First column is typically text
                        label_col = chunk.columns[1]  # Second column is typically label
                
                logger.info(f"Using columns - Text: {text_col}, Label: {label_col}")
                
                # Clean data: Ensure text column contains strings and handle missing values
                chunk[text_col] = chunk[text_col].astype(str)
                chunk[text_col] = chunk[text_col].fillna('')
                
                # Prepare data for model training
                X = chunk[text_col].values
                
                # Convert labels to binary
                if pd.api.types.is_numeric_dtype(chunk[label_col]):
                    # Numeric labels
                    y = chunk[label_col].values
                else:
                    # String labels - convert 'suicide', 'yes', 'true', etc. to 1, others to 0
                    suicide_keywords = ['suicide', 'suicidal', 'yes', 'positive', '1', 'true']
                    y = np.array([1 if str(label).lower() in suicide_keywords else 0 
                                for label in chunk[label_col]])
                
                # Log label distribution to debug
                unique_labels, counts = np.unique(y, return_counts=True)
                logger.info(f"Label distribution in chunk {chunk_num+1}: {dict(zip(unique_labels, counts))}")
                
                if first_chunk:
                    # For the first chunk, just accumulate the data
                    logger.info("First chunk: collecting data for initial training")
                    X_all = X
                    y_all = y
                    first_chunk = False
                else:
                    # For subsequent chunks, stack with existing data
                    X_all = np.concatenate((X_all, X))
                    y_all = np.concatenate((y_all, y))
                
                # Collect suicide-related texts for keyword extraction (limited to first few chunks)
                if chunk_num < 5:  # Limit to first 5 chunks to avoid memory issues
                    if pd.api.types.is_numeric_dtype(chunk[label_col]):
                        suicide_mask = chunk[label_col] == 1
                    else:
                        suicide_mask = chunk[label_col].str.lower().isin(suicide_keywords)
                    
                    suicide_texts = chunk.loc[suicide_mask, text_col].tolist()
                    logger.info(f"Collected {len(suicide_texts)} suicide texts from chunk {chunk_num+1}")
                    all_suicide_texts.extend(suicide_texts)
                
                chunks_processed += 1
                total_rows_processed += len(chunk)
                logger.info(f"Processed {total_rows_processed} rows so far")
            
            if chunks_processed == 0:
                logger.error("No chunks were processed successfully")
                return False
            
            # Check if we have both classes in the data
            unique_labels = np.unique(y_all)
            if len(unique_labels) < 2:
                logger.warning(f"Data contains only one class: {unique_labels}. Need at least two classes for training.")
                if all_suicide_texts:
                    logger.info(f"However, we collected {len(all_suicide_texts)} suicide texts for keyword extraction")
                    # Extract keywords from collected suicide texts
                    logger.info(f"Extracting keywords from {len(all_suicide_texts)} suicide texts")
                    self._extract_keywords_from_texts(all_suicide_texts)
                    logger.info("Keyword extraction completed successfully")
                    return True
                else:
                    logger.error("No suicide texts collected for keyword extraction")
                    return False
            
            # Train on all collected data at once to ensure proper fitting
            logger.info(f"Training model on {len(X_all)} samples...")
            logger.info(f"Label distribution in all data: {dict(zip(*np.unique(y_all, return_counts=True)))}")
            
            # Transform collected text data
            X_transformed = vectorizer.fit_transform(X_all)
            
            # Train classifier on all data
            classifier.fit(X_transformed, y_all)
            
            # Save the model components
            self.vectorizer = vectorizer
            self.model = classifier
            
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
            
            # Save model to disk
            model_data = {
                'vectorizer': vectorizer,
                'classifier': classifier
            }
            joblib.dump(model_data, self.model_path)
            logger.info(f"Model saved to {self.model_path}")
            
            # Extract keywords from collected suicide texts
            if all_suicide_texts:
                logger.info(f"Extracting keywords from {len(all_suicide_texts)} collected suicide texts")
                self._extract_keywords_from_texts(all_suicide_texts)
            else:
                logger.warning("No suicide texts collected for keyword extraction, using defaults")
                self._set_default_keywords()
            
            logger.info(f"Model training completed with {chunks_processed} chunks ({total_rows_processed} rows)")
            return True
            
        except Exception as e:
            logger.error(f"Error during chunked model training: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            return False
    
    def _extract_keywords_from_texts(self, suicide_texts):
        """Extract keywords from a list of suicide-related texts"""
        try:
            logger.info(f"Extracting keywords from {len(suicide_texts)} texts")
            
            if len(suicide_texts) == 0:
                logger.warning("No suicide-related texts provided for keyword extraction")
                self._set_default_keywords()
                return False
            
            # Use TF-IDF to extract keywords
            vectorizer = TfidfVectorizer(max_features=100, stop_words='english')
            tfidf_matrix = vectorizer.fit_transform(suicide_texts)
            feature_names = vectorizer.get_feature_names_out()
            
            # Calculate average TF-IDF score for each keyword
            avg_scores = tfidf_matrix.mean(axis=0).A1
            
            # Sort keywords by score
            keywords_with_scores = list(zip(feature_names, avg_scores))
            keywords_with_scores.sort(key=lambda x: x[1], reverse=True)
            
            # Categorize keywords into high, medium, low based on scores
            total_keywords = len(keywords_with_scores)
            high_count = max(10, total_keywords // 3)
            medium_count = max(20, total_keywords // 3)
            
            self.keywords = {
                "high": [kw for kw, _ in keywords_with_scores[:high_count]],
                "medium": [kw for kw, _ in keywords_with_scores[high_count:high_count+medium_count]],
                "low": [kw for kw, _ in keywords_with_scores[high_count+medium_count:]]
            }
            
            # Add default high-risk keywords to ensure critical terms are included
            self._add_default_keywords()
            
            # Log some examples from each category
            logger.info(f"High risk keywords ({len(self.keywords['high'])}): " + 
                      ', '.join(self.keywords['high'][:10]))
            logger.info(f"Medium risk keywords ({len(self.keywords['medium'])}): " + 
                      ', '.join(self.keywords['medium'][:10]))
            logger.info(f"Low risk keywords ({len(self.keywords['low'])}): " + 
                      ', '.join(self.keywords['low'][:10]))
            
            return True
            
        except Exception as e:
            logger.error(f"Error during keyword extraction from texts: {str(e)}")
            self._set_default_keywords()
            return False

# 테스트 코드
if __name__ == "__main__":
    predictor = SuicidePredictor()
    
    # 모델이 없으면 학습
    if predictor.model is None:
        # 파일 우선순위: 1. 샘플 파일 2. 압축 파일 3. 전체 파일
        sample_csv_path = 'Suicide_Detection_sample.csv'
        compressed_csv_path = 'Suicide_Detection.csv.gz'
        full_csv_path = 'Suicide_Detection.csv'
        
        if os.path.exists(full_csv_path):
            logger.info(f"Using full CSV file with chunked processing: {full_csv_path}")
            predictor.train_from_large_csv(full_csv_path, chunk_size=10000)
        elif os.path.exists(compressed_csv_path):
            logger.info(f"Using compressed CSV file: {compressed_csv_path}")
            # 압축 파일 읽기
            with gzip.open(compressed_csv_path, 'rt') as f:
                df = pd.read_csv(f)
            csv_path = None  # df를 직접 전달하기 위해 경로는 None으로 설정
            temp_csv_path = 'temp_data.csv'
            df.to_csv(temp_csv_path, index=False)
            predictor.train_from_csv(temp_csv_path)
            # 임시 파일 삭제
            if os.path.exists(temp_csv_path):
                os.remove(temp_csv_path)
        elif os.path.exists(sample_csv_path):
            logger.info(f"Using sample CSV file: {sample_csv_path}")
            csv_path = sample_csv_path
            predictor.train_from_csv(csv_path)
        else:
            raise FileNotFoundError("No suicide detection CSV file found. Please provide either Suicide_Detection.csv, Suicide_Detection.csv.gz, or Suicide_Detection_sample.csv")
        
    # 테스트 텍스트로 예측
    test_text = "I don't want to live anymore. Everything feels so hopeless."
    result = predictor.predict(test_text)
    print(f"Risk level: {result['risk_level']}")
    print(f"Risk score: {result['risk_score']}")
    print(f"Keywords found: {result['keywords_found']}")

    # Add new keywords
    predictor.high_risk_keywords = [
        "suicide", "kill myself", "end my life", "take my life", 
        "want to die", "overdose", "jump", "hang myself", "shoot myself",
        "self-harm", "slit wrists", "noose", "suicidal thoughts", 
        "better off dead", "not worth living", "die soon", "suicidal ideation",
        "lethal", "razor", "gun", "pills", "fatal", "suicide note", "final goodbye",
        "want to end it all", "take my own life", "end it all", "last day",
        "life insurance", "will and testament", "sleeping pills", "painless death"
    ]
    predictor.medium_risk_keywords = [
        "hopeless", "worthless", "helpless", "depressed", "miserable", 
        "suffering", "can't go on", "tired of life", "burden to others", 
        "no purpose", "no meaning", "no hope", "no future", "darkness",
        "unbearable pain", "emptiness", "nothingness", "despair", "trapped", 
        "agony", "anguish", "torment", "exhausted", "meaningless", "pointless", 
        "alone", "lonely", "isolated", "abandoned", "rejected", 
        "giving up", "can't cope", "too much pain", "endless suffering"
    ]
    predictor.low_risk_keywords = [
        "sad", "upset", "unhappy", "frustrated", "disappointed", 
        "tired", "exhausted", "hurt", "pain", "dissatisfied", 
        "unfulfilled", "discouraged", "disheartened", "dejected", 
        "heartbroken", "defeated", "fatigued", "weary", "drained", 
        "spent", "troubled", "distressed", "blue", "down", "gloomy",
        "sorrowful", "tearful", "confused", "lost", "struggling",
        "empty", "numb", "anxiety", "worry", "stressed"
    ] 