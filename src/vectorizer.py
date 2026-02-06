import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Union
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import TruncatedSVD
import logging

# Imports for embeddings (optional)
try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False
    logging.warning("sentence-transformers not available. Embeddings option will be unavailable.")


class Vectorizer:
    """
    Step 6: Numerical representation (TF-IDF, Embeddings)
    """
    
    def __init__(self):
        self.vectorizers = {}
        self.embeddings_model = None
        self.feature_names = {}
        
    def create_tfidf_vectorizer(
        self,
        max_features: int = 10000,
        ngram_range: Tuple[int, int] = (1, 2),
        min_df: int = 2,
        max_df: float = 0.95,
        stop_words: Optional[str] = 'english'
    ) -> TfidfVectorizer:
        """Tworzy TF-IDF wektoryzator"""
        vectorizer = TfidfVectorizer(
            max_features=max_features,
            ngram_range=ngram_range,
            min_df=min_df,
            max_df=max_df,
            stop_words=stop_words,
            lowercase=False,  # Zakładamy, że tekst jest już małymi literami
            token_pattern=r'(?u)\b\w\w+\b'  # Minimum 2 znaki
        )
        
        logging.info(f"Utworzono TF-IDF wektoryzator: max_features={max_features}, ngram_range={ngram_range}")
        return vectorizer
    
    def create_count_vectorizer(
        self,
        max_features: int = 10000,
        ngram_range: Tuple[int, int] = (1, 2),
        min_df: int = 2,
        max_df: float = 0.95,
        stop_words: Optional[str] = 'english'
    ) -> CountVectorizer:
        """Tworzy Count wektoryzator"""
        vectorizer = CountVectorizer(
            max_features=max_features,
            ngram_range=ngram_range,
            min_df=min_df,
            max_df=max_df,
            stop_words=stop_words,
            lowercase=False,
            token_pattern=r'(?u)\b\w\w+\b'
        )
        
        logging.info(f"Created Count vectorizer: max_features={max_features}, ngram_range={ngram_range}")
        return vectorizer
    
    def fit_transform_text(
        self,
        texts: List[str],
        vectorizer_name: str = "tfidf",
        vectorizer_type: str = "tfidf",
        **vectorizer_params
    ) -> np.ndarray:
        """Fit and transform texts to vectors"""
        
        if vectorizer_type == "tfidf":
            vectorizer = self.create_tfidf_vectorizer(**vectorizer_params)
        elif vectorizer_type == "count":
            vectorizer = self.create_count_vectorizer(**vectorizer_params)
        else:
            raise ValueError(f"Unsupported vectorizer type: {vectorizer_type}")
        
        # Fit and transform
        X = vectorizer.fit_transform(texts)
        
        # Save vectorizer and feature names
        self.vectorizers[vectorizer_name] = vectorizer
        self.feature_names[vectorizer_name] = vectorizer.get_feature_names_out()
        
        logging.info(f"Vectorized {len(texts)} texts, shape: {X.shape}")
        
        return X.toarray()
    
    def transform_text(
        self,
        texts: List[str],
        vectorizer_name: str = "tfidf"
    ) -> np.ndarray:
        """Transform texts using existing vectorizer"""
        
        if vectorizer_name not in self.vectorizers:
            raise ValueError(f"Vectorizer '{vectorizer_name}' has not been fitted")
        
        vectorizer = self.vectorizers[vectorizer_name]
        X = vectorizer.transform(texts)
        
        logging.info(f"Transformed {len(texts)} texts, shape: {X.shape}")
        
        return X.toarray()
    
    def load_embeddings_model(self, model_name: str = "all-MiniLM-L6-v2") -> None:
        """Load embeddings model"""
        if not SENTENCE_TRANSFORMERS_AVAILABLE:
            raise ImportError("sentence-transformers is not installed")
        
        try:
            self.embeddings_model = SentenceTransformer(model_name)
            logging.info(f"Loaded embeddings model: {model_name}")
        except Exception as e:
            logging.error(f"Error loading embeddings model: {e}")
            raise
    
    def create_embeddings(
        self,
        texts: List[str],
        batch_size: int = 32,
        show_progress: bool = True
    ) -> np.ndarray:
        """Create embeddings for texts"""
        if self.embeddings_model is None:
            self.load_embeddings_model()
        
        # Filter empty texts
        valid_texts = [text if text and text.strip() else "empty" for text in texts]
        
        embeddings = self.embeddings_model.encode(
            valid_texts,
            batch_size=batch_size,
            show_progress_bar=show_progress,
            convert_to_numpy=True
        )
        
        logging.info(f"Created embeddings for {len(texts)} texts, shape: {embeddings.shape}")
        
        return embeddings
    
    def reduce_dimensions(
        self,
        X: np.ndarray,
        n_components: int = 300,
        method: str = "svd"
    ) -> np.ndarray:
        """Reduce vector dimensions"""
        
        if method == "svd":
            reducer = TruncatedSVD(n_components=n_components, random_state=42)
        else:
            raise ValueError(f"Unsupported reduction method: {method}")
        
        X_reduced = reducer.fit_transform(X)
        
        logging.info(f"Reduced dimensionality from {X.shape[1]} to {n_components}")
        
        return X_reduced
    
    def combine_features(
        self,
        features_list: List[np.ndarray],
        method: str = "concatenate"
    ) -> np.ndarray:
        """Combine different feature representations"""
        
        if method == "concatenate":
            combined = np.concatenate(features_list, axis=1)
        elif method == "average":
            # Average features (must have same length)
            combined = np.mean(features_list, axis=0)
        else:
            raise ValueError(f"Unsupported combination method: {method}")
        
        logging.info(f"Combined {len(features_list)} feature representations, result shape: {combined.shape}")
        
        return combined
    
    def get_feature_importance(
        self,
        vectorizer_name: str,
        X: np.ndarray,
        top_k: int = 20
    ) -> Dict[str, float]:
        """Returns most important features (average TF-IDF)"""
        
        if vectorizer_name not in self.vectorizers:
            raise ValueError(f"Vectorizer '{vectorizer_name}' does not exist")
        
        feature_names = self.feature_names[vectorizer_name]
        mean_scores = np.mean(X, axis=0)
        
        # Sort by importance
        feature_scores = list(zip(feature_names, mean_scores))
        feature_scores.sort(key=lambda x: x[1], reverse=True)
        
        top_features = dict(feature_scores[:top_k])
        
        return top_features
    
    def vectorize_dataframe(
        self,
        df: pd.DataFrame,
        text_columns: List[str],
        vectorizer_configs: Optional[Dict[str, Dict]] = None,
        combine_method: str = "concatenate"
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Zwektoryzuj kolumny tekstowe w DataFrame"""
        
        if vectorizer_configs is None:
            vectorizer_configs = {}
        
        all_features = []
        metadata = {
            'vectorizers_used': [],
            'feature_shapes': [],
            'total_features': 0
        }
        
        for col in text_columns:
            if col not in df.columns:
                logging.warning(f"Kolumna '{col}' nie istnieje")
                continue
            
            texts = df[col].fillna("").tolist()
            
            # Konfiguracja wektoryzatora
            config = vectorizer_configs.get(col, {'vectorizer_type': 'tfidf'})
            vectorizer_name = f"{col}_{config.get('vectorizer_type', 'tfidf')}"
            
            # Zwektoryzuj
            if config.get('use_embeddings', False):
                features = self.create_embeddings(texts)
            else:
                features = self.fit_transform_text(texts, vectorizer_name, **config)
            
            all_features.append(features)
            metadata['vectorizers_used'].append(vectorizer_name)
            metadata['feature_shapes'].append(features.shape)
        
        # Połącz cechy
        if all_features:
            X_combined = self.combine_features(all_features, combine_method)
            metadata['total_features'] = X_combined.shape[1]
        else:
            X_combined = np.array([]).reshape(len(df), 0)
        
        return X_combined, metadata


if __name__ == "__main__":
    # Przykład użycia
    import pandas as pd
    
    # Przykładowe dane
    data = {
        'text': ['problem with app login', 'payment processing error', 'user interface bug', 'feature request'],
        'summary': ['login issue', 'payment problem', 'ui bug', 'new feature']
    }
    df = pd.DataFrame(data)
    
    vectorizer = Vectorizer()
    
    # Konfiguracje
    configs = {
        'text': {'vectorizer_type': 'tfidf', 'max_features': 1000},
        'summary': {'vectorizer_type': 'tfidf', 'max_features': 500}
    }
    
    # Zwektoryzuj
    X, metadata = vectorizer.vectorize_dataframe(df, ['text', 'summary'], configs)
    
    print(f"Kształt połączonych cech: {X.shape}")
    print(f"Metadane: {metadata}")
    
    # Najważniejsze cechy
    if metadata['vectorizers_used']:
        top_features = vectorizer.get_feature_importance(metadata['vectorizers_used'][0], X)
        print(f"\nTop 5 cech: {list(top_features.items())[:5]}")
