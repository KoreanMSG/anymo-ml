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
        
        try:
            self.lemmatizer = WordNetLemmatizer()
            try:
                self.stop_words = set(stopwords.words('english'))
            except Exception as e:
                logger.warning(f"Could not load stopwords: {e}")
                self.stop_words = set()
        except Exception as e:
            logger.warning(f"Could not initialize lemmatizer: {e}")
            # 폴백 처리
            class DummyLemmatizer:
                def lemmatize(self, word):
                    return word
            self.lemmatizer = DummyLemmatizer()
            self.stop_words = set()
        
        # 모델 로드 또는 생성
        if os.path.exists(model_path):
            try:
                self.model = joblib.load(model_path)
                logger.info(f"Model loaded from {model_path}")
            except Exception as e:
                logger.error(f"Failed to load model: {e}")
                self.model = None
    
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
                "take my own life", "ending it all", "ending my life", "killing myself", "commit suicide"
            ],
            "medium": [
                "hopeless", "worthless", "no reason to live", "can't go on", "tired of living",
                "what's the point", "better off dead", "no future", "no hope", "give up"
            ],
            "low": [
                "depressed", "depression", "sad", "lonely", "alone", "pain", "suffering",
                "hurt", "failure", "miserable", "lost", "empty"
            ]
        }
        logger.info("Using default keywords.")

    def _add_default_keywords(self):
        """Add default keywords to existing keywords"""
        default_keywords = {
            "high": [
                "suicide", "kill myself", "end my life", "want to die", "rather be dead",
                "take my own life", "ending it all", "ending my life", "killing myself", "commit suicide"
            ],
            "medium": [
                "hopeless", "worthless", "no reason to live", "can't go on", "tired of living",
                "what's the point", "better off dead", "no future", "no hope", "give up"
            ],
            "low": [
                "depressed", "depression", "sad", "lonely", "alone", "pain", "suffering",
                "hurt", "failure", "miserable", "lost", "empty"
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
                # Model prediction
                prediction = self.model.predict_proba([text])[0]
                # Model prediction probability (0: non-suicide, 1: suicide)
                suicide_prob = prediction[1]
                
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
            (r"(want|wish) to (die|end|kill myself)", 30),  # 특정 표현 추가 점수
            (r"plan to (suicide|kill myself|end my life)", 40),
            (r"no (reason|point) (in|to) (living|life)", 25),
            (r"can'?t (take|bear|handle) (it|this) anymore", 20)
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
            (r"(want|wish) to (die|end|kill myself)", "High", "Want to die"),
            (r"plan to (suicide|kill myself|end my life)", "High", "Suicide plan"),
            (r"no (reason|point) (in|to) (living|life)", "Medium", "No reason to live"),
            (r"can'?t (take|bear|handle) (it|this) anymore", "Medium", "Can't take it anymore")
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

# 테스트 코드
if __name__ == "__main__":
    predictor = SuicidePredictor()
    
    # 모델이 없으면 학습
    if predictor.model is None:
        # 파일 우선순위: 1. 샘플 파일 2. 압축 파일 3. 전체 파일
        sample_csv_path = 'Suicide_Detection_sample.csv'
        compressed_csv_path = 'Suicide_Detection.csv.gz'
        full_csv_path = 'Suicide_Detection.csv'
        
        if os.path.exists(sample_csv_path):
            logger.info(f"Using sample CSV file: {sample_csv_path}")
            csv_path = sample_csv_path
            df = pd.read_csv(csv_path)
        elif os.path.exists(compressed_csv_path):
            logger.info(f"Using compressed CSV file: {compressed_csv_path}")
            # 압축 파일 읽기
            with gzip.open(compressed_csv_path, 'rt') as f:
                df = pd.read_csv(f)
            csv_path = None  # df를 직접 전달하기 위해 경로는 None으로 설정
        elif os.path.exists(full_csv_path):
            logger.info(f"Using full CSV file: {full_csv_path}")
            csv_path = full_csv_path
            df = pd.read_csv(csv_path)
        else:
            raise FileNotFoundError("No suicide detection CSV file found. Please provide either Suicide_Detection.csv, Suicide_Detection.csv.gz, or Suicide_Detection_sample.csv")
        
        # csv_path가 있으면 경로로 학습, 없으면 df로 학습
        if csv_path:
            predictor.train_from_csv(csv_path)
        else:
            # train 메소드를 DataFrame을 직접 받을 수 있도록 수정 필요
            # 임시로 DataFrame을 CSV로 저장하고 사용
            temp_csv_path = 'temp_data.csv'
            df.to_csv(temp_csv_path, index=False)
            predictor.train_from_csv(temp_csv_path)
            # 임시 파일 삭제
            if os.path.exists(temp_csv_path):
                os.remove(temp_csv_path)
        
    # 테스트 텍스트로 예측
    test_text = "I don't want to live anymore. Everything feels so hopeless."
    result = predictor.predict(test_text)
    print(f"Risk level: {result['risk_level']}")
    print(f"Risk score: {result['risk_score']}")
    print(f"Keywords found: {result['keywords_found']}") 