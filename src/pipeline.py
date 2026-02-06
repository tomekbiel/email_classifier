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
    Run full classification pipeline
    
    Args:
            data_path: Path to CSV data
            target_column: Target column
            text_columns: Text columns to process
            
        Returns:
            Dict with pipeline results
        """
        
        start_time = time.time()
        logging.info("Starting full email classification pipeline")
        
        try:
            # Step 1: Data selection
            self.data = self._step_data_selection(data_path, target_column)
            
            # Step 2: Translation (optional)
            if self.config["translation"]["enable"]:
                self.data = self._step_translation(text_columns)
            
            # Step 3: Preprocessing
            self.data = self._step_preprocessing(text_columns)
            
            # Step 4: Data structuring
            self.data = self._step_data_structuring(target_column)
            
            # Step 5: Vectorization
            self.X = self._step_vectorization(text_columns)
            self.y = self.data[target_column]
            
            # Step 6: Sampling (optional)
            if self.config["sampling"]["enable"]:
                self.X, self.y = self._step_sampling()
            
            # Step 7: Strategy selection
            strategy = self._step_strategy_selection()
            
            # Step 8: Data split
            X_train, X_test, y_train, y_test = self._step_data_split()
            
            # Step 9: Modeling and evaluation
            modeling_results = self._step_modeling(X_train, X_test, y_train, y_test)
            
            # Summary
            total_time = time.time() - start_time
            self.results = {
                'pipeline_time': total_time,
                'data_shape': self.data.shape,
                'feature_shape': self.X.shape,
                'strategy': strategy,
                'modeling_results': modeling_results,
                'config': self.config
            }
            
            logging.info(f"Pipeline completed successfully in {total_time:.2f} seconds")
            return self.results
            
        except Exception as e:
            logging.error(f"Error in pipeline: {e}")
            raise
    
    def _step_data_selection(self, data_path: str, target_column: str) -> pd.DataFrame:
        """Step 1: Data selection"""
        logging.info("Step 1: Data selection")
        
        self.data_selector = DataSelector(data_path)
        df, metadata = self.data_selector.process_data(
            filter_frequency=self.config["data_selection"]["filter_frequency"]
        )
        
        logging.info(f"Loaded data: {metadata['shape']}")
        return df
    
    def _step_translation(self, text_columns: List[str]) -> pd.DataFrame:
        """Step 2: Translation"""
        logging.info("Step 2: Translating texts")
        
        self.translator = Translator(self.config["translation"]["model_name"])
        
        # Note: We can only translate a limited number of words, 
        # so we are only translating ticket summary and not interaction content
        summary_col = text_columns[0] if len(text_columns) > 0 else "Ticket Summary"
        
        if summary_col in self.data.columns:
            logging.info(f"Translating only '{summary_col}' due to API limitations")
            self.data = self.translator.translate_dataframe_column(
                self.data, summary_col, f"{summary_col}_en"
            )
        else:
            logging.warning(f"Summary column '{summary_col}' not found")
        
        return self.data
    
    def _step_preprocessing(self, text_columns: List[str]) -> pd.DataFrame:
        """Step 3: Preprocessing"""
        logging.info("Step 3: Text cleaning")
        
        self.preprocessor = TextPreprocessor()
        
        # Column mapping
        summary_col = text_columns[0] if len(text_columns) > 0 else "Ticket Summary"
        interaction_col = text_columns[1] if len(text_columns) > 1 else "Interaction content"
        
        self.data = self.preprocessor.preprocess_dataframe(
            self.data, summary_col, interaction_col
        )
        
        return self.data
    
    def _step_data_structuring(self, target_column: str) -> pd.DataFrame:
        """Step 4: Data structuring"""
        logging.info("Step 4: Data structuring")
        
        self.structurer = DataStructurer()
        
        # Label encoding
        label_columns = [target_column]
        if 'y1' in self.data.columns:
            label_columns.append('y1')
        
        self.data = self.structurer.encode_labels(
            self.data, label_columns, fit_transform=True
        )
        
        return self.data
    
    def _step_vectorization(self, text_columns: List[str]) -> np.ndarray:
        """Step 5: Vectorization"""
        logging.info("Step 5: Text vectorization")
        
        self.vectorizer = Vectorizer()
        
        # Prepare text columns
        clean_columns = []
        for col in text_columns:
            clean_col = f"{'ts' if 'Summary' in col else 'ic'}"
            if clean_col in self.data.columns:
                clean_columns.append(clean_col)
        
        if not clean_columns:
            raise ValueError("No clean text columns available")
        
        # Vectorization configurations
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
        
        logging.info(f"Vectorized features: shape {X.shape}")
        return X
    
    def _step_sampling(self) -> Tuple[np.ndarray, np.ndarray]:
        """Step 6: Sampling"""
        logging.info("Step 6: Data balancing")
        
        self.sampler = Sampler(random_state=self.random_state)
        
        # Imbalance analysis
        analysis = self.sampler.analyze_imbalance(self.y)
        logging.info(f"Imbalance analysis: ratio={analysis['imbalance_ratio']:.2f}")
        
        if not analysis['is_balanced']:
            method = self.config["sampling"]["method"]
            
            if method == "auto":
                X_resampled, y_resampled = self.sampler.auto_balance(
                    self.X, self.y, 
                    max_ratio=self.config["sampling"]["max_ratio"]
                )
            else:
                # Implementation of other methods
                X_resampled, y_resampled = self.sampler.smote_oversample(self.X, self.y)
            
            logging.info(f"Data balanced: {len(self.X)} -> {len(X_resampled)}")
            return X_resampled, y_resampled
        
        return self.X, self.y
    
    def _step_strategy_selection(self) -> Dict[str, Any]:
        """Step 7: Strategy selection"""
        logging.info("Step 7: Learning strategy analysis")
        
        self.strategy_selector = StrategySelector(random_state=self.random_state)
        strategy = self.strategy_selector.recommend_strategy(self.X, self.y)
        
        logging.info(f"Recommended strategy: {strategy['primary_strategy']}")
        return strategy
    
    def _step_data_split(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Step 8: Data split"""
        logging.info("Step 8: Splitting data into training and test sets")
        
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
        
        logging.info(f"Data split: {len(X_train)} training, {len(X_test)} test")
        return X_train, X_test, y_train, y_test
    
    def _step_modeling(
        self, 
        X_train: np.ndarray, 
        X_test: np.ndarray, 
        y_train: np.ndarray, 
        y_test: np.ndarray
    ) -> Dict[str, Any]:
        """Step 9: Modeling and evaluation"""
        logging.info("Step 9: Model training and evaluation")
        
        self.model_trainer = ModelTrainer(random_state=self.random_state)
        self.model_evaluator = ModelEvaluator()
        
        # Train models
        models_to_train = self.config["modeling"]["models"]
        training_results = self.model_trainer.train_multiple_models(
            X_train, y_train, models_to_train, X_test, y_test
        )
        
        # Model evaluation
        evaluation_results = {}
        
        for model_name, result in training_results.items():
            if 'error' not in result:
                # Predictions
                y_pred = result['model'].predict(X_test)
                
                # Probabilities (if available)
                y_proba = None
                if hasattr(result['model'], 'predict_proba'):
                    y_proba = result['model'].predict_proba(X_test)
                
                # Evaluation
                eval_result = self.model_evaluator.evaluate_classification(
                    y_test, y_pred, y_proba, model_name=model_name
                )
                
                evaluation_results[model_name] = eval_result
        
        # Model comparison
        model_comparison = self.model_evaluator.compare_models()
        best_model, best_score = self.model_evaluator.get_best_model()
        
        modeling_results = {
            'training_results': training_results,
            'evaluation_results': evaluation_results,
            'model_comparison': model_comparison.to_dict(),
            'best_model': best_model,
            'best_score': best_score
        }
        
        logging.info(f"Best model: {best_model} (score: {best_score:.4f})")
        return modeling_results
    
    def predict(
        self, 
        new_data: Union[pd.DataFrame, List[str]], 
        model_name: Optional[str] = None
    ) -> np.ndarray:
        """Predictions on new data"""
        
        if self.model_trainer is None:
            raise ValueError("Pipeline has not been trained")
        
        # Prepare data
        if isinstance(new_data, list):
            # Convert list of texts to DataFrame
            df = pd.DataFrame({'text': new_data})
            # Here should be full processing...
            # Simplification - assuming data is already processed
            X_processed = self.vectorizer.transform_text(new_data)
        else:
            # DataFrame - process text columns
            # Simplified implementation
            pass
        
        return self.model_trainer.predict(X_processed, model_name)
    
    def save_pipeline(self, filepath: str) -> None:
        """Save pipeline"""
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
        logging.info(f"Pipeline saved to {filepath}")
    
    def load_pipeline(self, filepath: str) -> None:
        """Load pipeline"""
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
        
        logging.info(f"Pipeline loaded from {filepath}")


if __name__ == "__main__":
    # Usage example
    pipeline = EmailClassificationPipeline()
    
    # Run pipeline (assuming data file exists)
    try:
        # Get absolute path to data file
        data_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "AppGallery.csv")
        results = pipeline.run_full_pipeline(data_path)
        print("Pipeline zakończony pomyślnie!")
        print(f"Najlepszy model: {results['modeling_results']['best_model']}")
        print(f"Najlepszy score: {results['modeling_results']['best_score']:.4f}")
        
    except FileNotFoundError:
        print("File AppGallery.csv not found. Run pipeline with appropriate data file.")
    except Exception as e:
        print(f"Error: {e}")
