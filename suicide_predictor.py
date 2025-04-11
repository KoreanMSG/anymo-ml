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

# 환경 설정
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# NLTK 리소스 다운로드
try:
    # NLTK 데이터 디렉토리 설정 (Render 환경에서 쓰기 가능한 경로로)
    nltk_data_dir = os.path.join(os.path.dirname(__file__), 'nltk_data')
    os.makedirs(nltk_data_dir, exist_ok=True)
    
    # NLTK 데이터 경로 설정
    nltk.data.path.append(nltk_data_dir)
    
    # 필요한 데이터 다운로드
    nltk.download('punkt', download_dir=nltk_data_dir, quiet=True)
    nltk.download('stopwords', download_dir=nltk_data_dir, quiet=True)
    nltk.download('wordnet', download_dir=nltk_data_dir, quiet=True)
    
    logger.info(f"NLTK resources downloaded to {nltk_data_dir}")
except Exception as e:
    logger.warning(f"NLTK download warning: {e}")
    # 에러가 있더라도 기본적인 처리 가능하도록 준비
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
        """텍스트 전처리 함수"""
        # 소문자 변환
        text = text.lower()
        # 토큰화
        word_tokens = word_tokenize(text)
        # 불용어 제거 및 lemmatization
        filtered_tokens = [
            self.lemmatizer.lemmatize(word) for word in word_tokens 
            if word.isalnum() and word not in self.stop_words
        ]
        return ' '.join(filtered_tokens)
    
    def extract_keywords_from_csv(self, csv_path):
        """CSV 데이터에서 자살 관련 키워드를 추출하는 함수"""
        try:
            # CSV 읽기
            df = pd.read_csv(csv_path)
            
            # 데이터 구조 확인
            if len(df.columns) >= 2:
                text_column = df.columns[0] if df.columns[0] != '' else df.columns[1]
                label_column = df.columns[1]
                
                # 자살 관련 텍스트 분리
                suicide_texts = []
                
                if df[label_column].dtype == 'object':
                    suicide_keywords = ['suicide', 'suicidal', 'yes', 'positive', '1', 'true']
                    suicide_mask = df[label_column].apply(
                        lambda x: 1 if str(x).lower() in suicide_keywords else 0
                    ).astype(bool)
                else:
                    suicide_mask = df[label_column].astype(bool)
                
                suicide_texts = df.loc[suicide_mask, text_column].tolist()
                
                # TF-IDF를 사용하여 주요 키워드 추출
                tfidf = TfidfVectorizer(max_features=300, stop_words='english')
                tfidf_matrix = tfidf.fit_transform(suicide_texts)
                
                # 주요 단어 추출
                feature_names = tfidf.get_feature_names_out()
                
                # 점수 기반 정렬 및 상위 키워드 선택
                tfidf_sum = tfidf_matrix.sum(axis=0).A1
                word_scores = [(word, score) for word, score in zip(feature_names, tfidf_sum)]
                word_scores.sort(key=lambda x: x[1], reverse=True)
                
                # 점수 기반으로 키워드 분류
                total_words = len(word_scores)
                high_threshold = int(total_words * 0.2)  # 상위 20%
                medium_threshold = int(total_words * 0.5)  # 상위 50%
                
                self.keywords["high"] = [word for word, _ in word_scores[:high_threshold] if len(word) > 2]
                self.keywords["medium"] = [word for word, _ in word_scores[high_threshold:medium_threshold] if len(word) > 2]
                self.keywords["low"] = [word for word, _ in word_scores[medium_threshold:] if len(word) > 2]
                
                # 기본 중요 키워드 추가 (다른 언어 포함)
                self.keywords["high"].extend(["자살", "죽고 싶", "suicide", "kill myself"])
                self.keywords["medium"].extend(["우울", "희망이 없", "삶이 의미", "살아갈 이유", "혼자", "외롭", "고통", "hopeless", "worthless", "alone", "lonely", "pain", "suffering", "depressed"])
                self.keywords["low"].extend(["슬프", "힘들", "지쳤", "피곤", "실패", "포기", "도움", "sad", "tired", "exhausted", "failed", "give up", "help me"])
                
                print(f"CSV에서 키워드 추출 완료: {len(self.keywords['high'])} high, {len(self.keywords['medium'])} medium, {len(self.keywords['low'])} low")
            else:
                print(f"CSV에 충분한 열이 없음: {csv_path}")
                self._set_default_keywords()
        except Exception as e:
            print(f"키워드 추출 실패: {e}")
            self._set_default_keywords()
    
    def _set_default_keywords(self):
        """기본 키워드 설정"""
        self.keywords = {
            "high": [
                "자살", "죽고 싶", "목숨", "목맬", "목을 매", "죽어버리", "없어지고 싶", "사라지고 싶", 
                "suicide", "kill myself", "end my life", "take my life"
            ],
            "medium": [
                "우울", "희망이 없", "삶이 의미", "살아갈 이유", "혼자", "외롭", "고통", 
                "hopeless", "worthless", "alone", "lonely", "pain", "suffering", "depressed"
            ],
            "low": [
                "슬프", "힘들", "지쳤", "피곤", "실패", "포기", "도움", 
                "sad", "tired", "exhausted", "failed", "give up", "help me"
            ]
        }

    def train_from_csv(self, csv_path):
        """CSV 데이터로부터 모델 학습"""
        try:
            # 키워드 먼저 추출
            self.extract_keywords_from_csv(csv_path)
            
            # 데이터 로드
            df = pd.read_csv(csv_path)
            
            # 데이터 구조 확인
            if len(df.columns) >= 2:
                text_column = df.columns[0] if df.columns[0] != '' else df.columns[1]
                label_column = df.columns[1]
                
                # 데이터 준비
                X = df[text_column].fillna('')
                
                # 라벨 변환 (필요한 경우)
                if df[label_column].dtype == 'object':
                    suicide_keywords = ['suicide', 'suicidal', 'yes', 'positive', '1', 'true']
                    y = df[label_column].apply(
                        lambda x: 1 if str(x).lower() in suicide_keywords else 0
                    )
                else:
                    y = df[label_column]
                
                # 데이터 전처리 및 학습
                self.vectorizer = TfidfVectorizer(max_features=5000)
                X_transformed = self.vectorizer.fit_transform(X)
                
                # 모델 학습
                self.model = LogisticRegression(C=1.0, max_iter=200)
                self.model.fit(X_transformed, y)
                
                print(f"모델 학습 완료")
                return True
            else:
                print(f"CSV 파일 형식이 올바르지 않음: {csv_path}")
                return False
        except Exception as e:
            print(f"학습 실패: {e}")
            return False
    
    def train(self, csv_path, test_size=0.2, random_state=42):
        """CSV 파일로부터 자살 예측 모델 학습"""
        try:
            # 데이터 로드
            logger.info(f"Loading data from {csv_path}")
            df = pd.read_csv(csv_path)
            
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
            if df[label_column].dtype == 'object':
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
    
    def predict(self, text):
        """텍스트의 자살 위험도 예측"""
        if not text:
            return {"risk_level": "unknown", "risk_score": 0, "keywords_found": []}
        
        # 확률 계산을 위한 예측
        processed_text = self.preprocess_text(text)
        
        if self.model and self.vectorizer:
            # 모델이 로드된 경우 머신러닝 예측 사용
            X = self.vectorizer.transform([processed_text])
            probability = self.model.predict_proba(X)[0][1]
            
            # 예측 결과 해석
            risk_score = probability * 100
        else:
            # 모델이 없는 경우 키워드 기반 분석
            # 자살 위험도 키워드 검사 (기본 단어 기반)
            risk_score = self._calculate_keyword_risk(processed_text)
        
        # 결과 해석
        risk_level = self._interpret_risk_score(risk_score)
        
        # 발견된 키워드 목록
        keywords_found = self._find_keywords_in_text(processed_text)
        
        return {
            "risk_level": risk_level,
            "risk_score": round(risk_score, 2),
            "keywords_found": keywords_found
        }
    
    def _calculate_keyword_risk(self, text):
        """키워드 기반 위험도 계산"""
        text_lower = text.lower()
        
        # 키워드 가중치
        weights = {"high": 0.7, "medium": 0.2, "low": 0.1}
        
        high_count = sum(1 for word in self.keywords["high"] if word.lower() in text_lower)
        medium_count = sum(1 for word in self.keywords["medium"] if word.lower() in text_lower)
        low_count = sum(1 for word in self.keywords["low"] if word.lower() in text_lower)
        
        # 위험도 점수 계산 (100점 만점)
        total_score = 0
        
        if high_count > 0:
            total_score += min(high_count * 30, 70)  # 최대 70점
        
        if medium_count > 0:
            total_score += min(medium_count * 7, 20)  # 최대 20점
        
        if low_count > 0:
            total_score += min(low_count * 3, 10)  # 최대 10점
        
        return min(total_score, 100)  # 최대 100점
    
    def _find_keywords_in_text(self, text):
        """텍스트에서 발견된 키워드 목록 반환"""
        text_lower = text.lower()
        found_keywords = []
        
        # 모든 카테고리의 키워드 검사
        for category in ["high", "medium", "low"]:
            for keyword in self.keywords[category]:
                if keyword.lower() in text_lower:
                    found_keywords.append({
                        "word": keyword,
                        "category": category
                    })
        
        return found_keywords
    
    def _interpret_risk_score(self, score):
        """위험도 점수 해석"""
        if score >= 70:
            return "high"
        elif score >= 30:
            return "medium"
        elif score >= 10:
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