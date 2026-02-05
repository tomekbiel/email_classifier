"""
Email Classification Pipeline

Complete pipeline for email/support ticket classification.
Connects all modules into a coherent processing process.
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Tuple, Optional, Any, Union
from pathlib import Path
import json
import time
import os

# Module imports
from data_selector import DataSelector
from translator import Translator
from text_preprocessor import TextPreprocessor
from data_structurer import DataStructurer
from vectorizer import Vectorizer
from sampler import Sampler
from strategy import StrategySelector
from data_splitter import DataSplitter
from model_trainer import ModelTrainer
from model_evaluator import ModelEvaluator


class EmailClassificationPipeline:
    """
    Main pipeline for email classification
    """
    
    def __init__(self, config_path: Optional[str] = None, random_state: int = 42):
        self.random_state = random_state
        self.config = self._load_config(config_path) if config_path else self._default_config()
        
        # Component initialization
        self.data_selector = None
        self.translator = None
        self.preprocessor = None
        self.structurer = None
        self.vectorizer = None
        self.sampler = None
        self.strategy_selector = None
        self.data_splitter = None
        self.model_trainer = None
        self.model_evaluator = None
        
        # Pipeline state
        self.data = None
        self.processed_data = None
        self.X = None
        self.y = None
        self.results = {}
        
        # Logging configuration
        self._setup_logging()
        
    def _load_config(self, config_path: str) -> Dict:
        """Load configuration from JSON file"""
        with open(config_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    def _default_config(self) -> Dict:
        """Default configuration"""
        return {
            "data_selection": {
                "filter_frequency": True,
                "min_samples_per_class": 10
            },
            "translation": {
                "enable": False,
                "model_name": "facebook/m2m100_418M"
            },
            "preprocessing": {
                "clean_summary": True,
                "clean_interaction": True
            },
            "vectorization": {
                "method": "tfidf",
                "max_features": 10000,
                "ngram_range": [1, 2]
            },
            "sampling": {
                "enable": True,
                "method": "auto",
                "max_ratio": 2.0
            },
            "data_split": {
                "test_size": 0.2,
                "stratify": True,
                "method": "basic"
            },
            "modeling": {
                "models": ["random_forest", "logistic_regression", "naive_bayes"],
                "hyperparameter_tuning": False
            }
        }
    
    def _setup_logging(self) -> None:
        """Konfiguruje logowanie"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('email_classifier.log'),
                logging.StreamHandler()
            ]
        )
    
    def run_full_pipeline(
        self,
        data_path: str,
        target_column: str = "y2",
        text_columns: List[str] = ["Ticket Summary", "Interaction content"]
    ) -> Dict[str, Any]:
        """
        Uruchamia pełny pipeline klasyfikacji
        
        Args:
            data_path: Ścieżka do danych CSV
            target_column: Kolumna docelowa
            text_columns: Kolumny tekstowe do przetwarzania
            
        Returns:
            Dict z wynikami pipeline'u
        """
        
        start_time = time.time()
        logging.info("Rozpoczynanie pełnego pipeline'u klasyfikacji emaili")
        
        try:
            # Krok 1: Selekcja danych
            self.data = self._step_data_selection(data_path, target_column)
            
            # Krok 2: Tłumaczenie (opcjonalne)
            if self.config["translation"]["enable"]:
                self.data = self._step_translation(text_columns)
            
            # Krok 3: Preprocessing
            self.data = self._step_preprocessing(text_columns)
            
            # Krok 4: Strukturyzacja danych
            self.data = self._step_data_structuring(target_column)
            
            # Krok 5: Wektoryzacja
            self.X = self._step_vectorization(text_columns)
            self.y = self.data[target_column]
            
            # Krok 6: Próbkowanie (opcjonalne)
            if self.config["sampling"]["enable"]:
                self.X, self.y = self._step_sampling()
            
            # Krok 7: Selekcja strategii
            strategy = self._step_strategy_selection()
            
            # Krok 8: Podział danych
            X_train, X_test, y_train, y_test = self._step_data_split()
            
            # Krok 9: Modelowanie i ewaluacja
            modeling_results = self._step_modeling(X_train, X_test, y_train, y_test)
            
            # Podsumowanie
            total_time = time.time() - start_time
            self.results = {
                'pipeline_time': total_time,
                'data_shape': self.data.shape,
                'feature_shape': self.X.shape,
                'strategy': strategy,
                'modeling_results': modeling_results,
                'config': self.config
            }
            
            logging.info(f"Pipeline zakończony pomyślnie w {total_time:.2f} sekund")
            return self.results
            
        except Exception as e:
            logging.error(f"Błąd w pipeline: {e}")
            raise
    
    def _step_data_selection(self, data_path: str, target_column: str) -> pd.DataFrame:
        """Krok 1: Selekcja danych"""
        logging.info("Krok 1: Selekcja danych")
        
        self.data_selector = DataSelector(data_path)
        df, metadata = self.data_selector.process_data(
            filter_frequency=self.config["data_selection"]["filter_frequency"]
        )
        
        logging.info(f"Załadowano dane: {metadata['shape']}")
        return df
    
    def _step_translation(self, text_columns: List[str]) -> pd.DataFrame:
        """Krok 2: Tłumaczenie"""
        logging.info("Krok 2: Tłumaczenie tekstów")
        
        self.translator = Translator(self.config["translation"]["model_name"])
        
        for col in text_columns:
            if col in self.data.columns:
                self.data = self.translator.translate_dataframe_column(
                    self.data, col, f"{col}_en"
                )
        
        return self.data
    
    def _step_preprocessing(self, text_columns: List[str]) -> pd.DataFrame:
        """Krok 3: Preprocessing"""
        logging.info("Krok 3: Czyszczenie tekstu")
        
        self.preprocessor = TextPreprocessor()
        
        # Mapowanie kolumn
        summary_col = text_columns[0] if len(text_columns) > 0 else "Ticket Summary"
        interaction_col = text_columns[1] if len(text_columns) > 1 else "Interaction content"
        
        self.data = self.preprocessor.preprocess_dataframe(
            self.data, summary_col, interaction_col
        )
        
        return self.data
    
    def _step_data_structuring(self, target_column: str) -> pd.DataFrame:
        """Krok 4: Strukturyzacja danych"""
        logging.info("Krok 4: Strukturyzacja danych")
        
        self.structurer = DataStructurer()
        
        # Kodowanie etykiet
        label_columns = [target_column]
        if 'y1' in self.data.columns:
            label_columns.append('y1')
        
        self.data = self.structurer.encode_labels(
            self.data, label_columns, fit_transform=True
        )
        
        return self.data
    
    def _step_vectorization(self, text_columns: List[str]) -> np.ndarray:
        """Krok 5: Wektoryzacja"""
        logging.info("Krok 5: Wektoryzacja tekstu")
        
        self.vectorizer = Vectorizer()
        
        # Przygotuj kolumny tekstowe
        clean_columns = []
        for col in text_columns:
            clean_col = f"{'ts' if 'Summary' in col else 'ic'}"
            if clean_col in self.data.columns:
                clean_columns.append(clean_col)
        
        if not clean_columns:
            raise ValueError("Brak oczyszczonych kolumn tekstowych")
        
        # Konfiguracje wektoryzacji
        vectorizer_configs = {}
        for col in clean_columns:
            vectorizer_configs[col] = {
                'vectorizer_type': self.config["vectorization"]["method"],
                'max_features': self.config["vectorization"]["max_features"],
                'ngram_range': tuple(self.config["vectorization"]["ngram_range"])
            }
        
        X, metadata = self.vectorizer.vectorize_dataframe(
            self.data, clean_columns, vectorizer_configs
        )
        
        logging.info(f"Zwektoryzowano cechy: kształt {X.shape}")
        return X
    
    def _step_sampling(self) -> Tuple[np.ndarray, np.ndarray]:
        """Krok 6: Próbkowanie"""
        logging.info("Krok 6: Balansowanie danych")
        
        self.sampler = Sampler(random_state=self.random_state)
        
        # Analiza niezbalansowania
        analysis = self.sampler.analyze_imbalance(self.y)
        logging.info(f"Analiza niezbalansowania: ratio={analysis['imbalance_ratio']:.2f}")
        
        if not analysis['is_balanced']:
            method = self.config["sampling"]["method"]
            
            if method == "auto":
                X_resampled, y_resampled = self.sampler.auto_balance(
                    self.X, self.y, 
                    max_ratio=self.config["sampling"]["max_ratio"]
                )
            else:
                # Implementacja innych metod
                X_resampled, y_resampled = self.sampler.smote_oversample(self.X, self.y)
            
            logging.info(f"Zbalansowano dane: {len(self.X)} -> {len(X_resampled)}")
            return X_resampled, y_resampled
        
        return self.X, self.y
    
    def _step_strategy_selection(self) -> Dict[str, Any]:
        """Krok 7: Selekcja strategii"""
        logging.info("Krok 7: Analiza strategii uczenia")
        
        self.strategy_selector = StrategySelector(random_state=self.random_state)
        strategy = self.strategy_selector.recommend_strategy(self.X, self.y)
        
        logging.info(f"Rekomendowana strategia: {strategy['primary_strategy']}")
        return strategy
    
    def _step_data_split(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Krok 8: Podział danych"""
        logging.info("Krok 8: Podział danych na zbiór treningowy i testowy")
        
        self.data_splitter = DataSplitter(random_state=self.random_state)
        
        split_method = self.config["data_split"]["method"]
        
        if split_method == "basic":
            X_train, X_test, y_train, y_test = self.data_splitter.basic_split(
                self.X, self.y,
                test_size=self.config["data_split"]["test_size"],
                stratify=self.config["data_split"]["stratify"]
            )
        else:
            # Inne metody podziału
            X_train, X_test, y_train, y_test = self.data_splitter.basic_split(
                self.X, self.y,
                test_size=self.config["data_split"]["test_size"],
                stratify=self.config["data_split"]["stratify"]
            )
        
        logging.info(f"Podział danych: {len(X_train)} treningowych, {len(X_test)} testowych")
        return X_train, X_test, y_train, y_test
    
    def _step_modeling(
        self, 
        X_train: np.ndarray, 
        X_test: np.ndarray, 
        y_train: np.ndarray, 
        y_test: np.ndarray
    ) -> Dict[str, Any]:
        """Krok 9: Modelowanie i ewaluacja"""
        logging.info("Krok 9: Trenowanie i ewaluacja modeli")
        
        self.model_trainer = ModelTrainer(random_state=self.random_state)
        self.model_evaluator = ModelEvaluator()
        
        # Trenuj modele
        models_to_train = self.config["modeling"]["models"]
        training_results = self.model_trainer.train_multiple_models(
            X_train, y_train, models_to_train, X_test, y_test
        )
        
        # Ewaluacja modeli
        evaluation_results = {}
        
        for model_name, result in training_results.items():
            if 'error' not in result:
                # Predykcje
                y_pred = result['model'].predict(X_test)
                
                # Prawdopodobieństwa (jeśli dostępne)
                y_proba = None
                if hasattr(result['model'], 'predict_proba'):
                    y_proba = result['model'].predict_proba(X_test)
                
                # Ewaluacja
                eval_result = self.model_evaluator.evaluate_classification(
                    y_test, y_pred, y_proba, model_name=model_name
                )
                
                evaluation_results[model_name] = eval_result
        
        # Porównanie modeli
        model_comparison = self.model_evaluator.compare_models()
        best_model, best_score = self.model_evaluator.get_best_model()
        
        modeling_results = {
            'training_results': training_results,
            'evaluation_results': evaluation_results,
            'model_comparison': model_comparison.to_dict(),
            'best_model': best_model,
            'best_score': best_score
        }
        
        logging.info(f"Najlepszy model: {best_model} (score: {best_score:.4f})")
        return modeling_results
    
    def predict(
        self, 
        new_data: Union[pd.DataFrame, List[str]], 
        model_name: Optional[str] = None
    ) -> np.ndarray:
        """Predykcje na nowych danych"""
        
        if self.model_trainer is None:
            raise ValueError("Pipeline nie został wytrenowany")
        
        # Przygotuj dane
        if isinstance(new_data, list):
            # Konwertuj listę tekstów na DataFrame
            df = pd.DataFrame({'text': new_data})
            # Tutaj powinno być pełne przetwarzanie...
            # Uproszczenie - zakładamy że dane są już przetworzone
            X_processed = self.vectorizer.transform_text(new_data)
        else:
            # DataFrame - przetwórz kolumny tekstowe
            # Uproszczenie implementacji
            pass
        
        return self.model_trainer.predict(X_processed, model_name)
    
    def save_pipeline(self, filepath: str) -> None:
        """Zapisuje pipeline"""
        import joblib
        
        pipeline_data = {
            'config': self.config,
            'data_selector': self.data_selector,
            'preprocessor': self.preprocessor,
            'structurer': self.structurer,
            'vectorizer': self.vectorizer,
            'sampler': self.sampler,
            'strategy_selector': self.strategy_selector,
            'data_splitter': self.data_splitter,
            'model_trainer': self.model_trainer,
            'model_evaluator': self.model_evaluator,
            'results': self.results
        }
        
        joblib.dump(pipeline_data, filepath)
        logging.info(f"Pipeline zapisany w {filepath}")
    
    def load_pipeline(self, filepath: str) -> None:
        """Ładuje pipeline"""
        import joblib
        
        pipeline_data = joblib.load(filepath)
        
        self.config = pipeline_data['config']
        self.data_selector = pipeline_data['data_selector']
        self.preprocessor = pipeline_data['preprocessor']
        self.structurer = pipeline_data['structurer']
        self.vectorizer = pipeline_data['vectorizer']
        self.sampler = pipeline_data['sampler']
        self.strategy_selector = pipeline_data['strategy_selector']
        self.data_splitter = pipeline_data['data_splitter']
        self.model_trainer = pipeline_data['model_trainer']
        self.model_evaluator = pipeline_data['model_evaluator']
        self.results = pipeline_data['results']
        
        logging.info(f"Pipeline załadowany z {filepath}")


if __name__ == "__main__":
    # Przykład użycia
    pipeline = EmailClassificationPipeline()
    
    # Uruchom pipeline (zakładając że plik danych istnieje)
    try:
        # Get absolute path to data file
        data_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "AppGallery.csv")
        results = pipeline.run_full_pipeline(data_path)
        print("Pipeline zakończony pomyślnie!")
        print(f"Najlepszy model: {results['modeling_results']['best_model']}")
        print(f"Najlepszy score: {results['modeling_results']['best_score']:.4f}")
        
    except FileNotFoundError:
        print("Plik AppGallery.csv nie został znaleziony. Uruchom pipeline z odpowiednim plikiem danych.")
    except Exception as e:
        print(f"Błąd: {e}")
