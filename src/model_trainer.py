import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Union
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import xgboost as xgb
import lightgbm as lgb
import logging
import joblib
import time


class ModelTrainer:
    """
    Kroki 10 & 11: Model SOTA, Trening i Ewaluacja
    """
    
    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        self.models = {}
        self.training_history = {}
        self.best_model = None
        self.best_score = 0
        
    def get_available_models(self) -> Dict[str, Any]:
        """Zwraca dostępne modele z domyślnymi parametrami"""
        
        models = {
            'random_forest': {
                'model': RandomForestClassifier,
                'params': {
                    'n_estimators': 100,
                    'random_state': self.random_state,
                    'n_jobs': -1
                },
                'description': 'Random Forest - ensemble drzew decyzyjnych'
            },
            'gradient_boosting': {
                'model': GradientBoostingClassifier,
                'params': {
                    'n_estimators': 100,
                    'random_state': self.random_state
                },
                'description': 'Gradient Boosting - sekwencyjne drzewa'
            },
            'logistic_regression': {
                'model': LogisticRegression,
                'params': {
                    'random_state': self.random_state,
                    'max_iter': 1000
                },
                'description': 'Regresja logistyczna - model liniowy'
            },
            'svm': {
                'model': SVC,
                'params': {
                    'random_state': self.random_state,
                    'probability': True
                },
                'description': 'Support Vector Machine - model wektorowy'
            },
            'naive_bayes': {
                'model': MultinomialNB,
                'params': {},
                'description': 'Naiwny Bayes - probabilistyczny model'
            },
            'knn': {
                'model': KNeighborsClassifier,
                'params': {
                    'n_neighbors': 5
                },
                'description': 'K-Nearest Neighbors - model oparty na sąsiedztwie'
            },
            'decision_tree': {
                'model': DecisionTreeClassifier,
                'params': {
                    'random_state': self.random_state
                },
                'description': 'Drzewo decyzyjne - pojedyncze drzewo'
            },
            'mlp': {
                'model': MLPClassifier,
                'params': {
                    'random_state': self.random_state,
                    'max_iter': 1000
                },
                'description': 'Multi-layer Perceptron - sieć neuronowa'
            },
            'xgboost': {
                'model': xgb.XGBClassifier,
                'params': {
                    'random_state': self.random_state,
                    'n_jobs': -1
                },
                'description': 'XGBoost - gradient boosting optymalizowany'
            },
            'lightgbm': {
                'model': lgb.LGBMClassifier,
                'params': {
                    'random_state': self.random_state,
                    'n_jobs': -1
                },
                'description': 'LightGBM - lekki gradient boosting'
            }
        }
        
        return models
    
    def train_single_model(
        self,
        X_train: Union[pd.DataFrame, np.ndarray],
        y_train: Union[pd.Series, np.ndarray],
        model_name: str,
        model_params: Optional[Dict] = None,
        X_val: Optional[Union[pd.DataFrame, np.ndarray]] = None,
        y_val: Optional[Union[pd.Series, np.ndarray]] = None
    ) -> Dict[str, Any]:
        """Trenuje pojedynczy model"""
        
        available_models = self.get_available_models()
        
        if model_name not in available_models:
            raise ValueError(f"Model '{model_name}' nie jest dostępny. Dostępne modele: {list(available_models.keys())}")
        
        model_info = available_models[model_name]
        
        # Połącz parametry
        params = model_info['params'].copy()
        if model_params:
            params.update(model_params)
        
        # Inicjalizuj model
        model = model_info['model'](**params)
        
        # Trenuj model
        start_time = time.time()
        
        try:
            model.fit(X_train, y_train)
            training_time = time.time() - start_time
            
            # Predykcje
            y_train_pred = model.predict(X_train)
            train_accuracy = accuracy_score(y_train, y_train_pred)
            
            val_results = {}
            if X_val is not None and y_val is not None:
                y_val_pred = model.predict(X_val)
                val_accuracy = accuracy_score(y_val, y_val_pred)
                val_results = {
                    'val_accuracy': val_accuracy,
                    'val_predictions': y_val_pred,
                    'val_classification_report': classification_report(y_val, y_val_pred, output_dict=True),
                    'val_confusion_matrix': confusion_matrix(y_val, y_val_pred)
                }
            
            # Zapisz model
            self.models[model_name] = model
            
            # Zapisz historię treningu
            training_info = {
                'model': model,
                'model_name': model_name,
                'params': params,
                'training_time': training_time,
                'train_accuracy': train_accuracy,
                'train_predictions': y_train_pred,
                'train_classification_report': classification_report(y_train, y_train_pred, output_dict=True),
                'train_confusion_matrix': confusion_matrix(y_train, y_train_pred),
                **val_results
            }
            
            self.training_history[model_name] = training_info
            
            # Zaktualizuj najlepszy model
            score = val_results.get('val_accuracy', train_accuracy)
            if score > self.best_score:
                self.best_score = score
                self.best_model = model
            
            logging.info(f"Model {model_name} wytrenowany pomyślnie. Czas: {training_time:.2f}s, Dokładność: {train_accuracy:.4f}")
            
            return training_info
            
        except Exception as e:
            logging.error(f"Błąd podczas treningu modelu {model_name}: {e}")
            raise
    
    def train_multiple_models(
        self,
        X_train: Union[pd.DataFrame, np.ndarray],
        y_train: Union[pd.Series, np.ndarray],
        model_names: Optional[List[str]] = None,
        X_val: Optional[Union[pd.DataFrame, np.ndarray]] = None,
        y_val: Optional[Union[pd.Series, np.ndarray]] = None
    ) -> Dict[str, Dict[str, Any]]:
        """Trenuje wiele modeli"""
        
        if model_names is None:
            model_names = ['random_forest', 'logistic_regression', 'svm', 'naive_bayes']
        
        results = {}
        
        for model_name in model_names:
            try:
                result = self.train_single_model(X_train, y_train, model_name, X_val=X_val, y_val=y_val)
                results[model_name] = result
            except Exception as e:
                logging.warning(f"Nie udało się wytrenować modelu {model_name}: {e}")
                results[model_name] = {'error': str(e)}
        
        return results
    
    def hyperparameter_tuning(
        self,
        X_train: Union[pd.DataFrame, np.ndarray],
        y_train: Union[pd.Series, np.ndarray],
        model_name: str,
        param_grid: Dict[str, List],
        cv_folds: int = 3,
        scoring: str = 'accuracy'
    ) -> Dict[str, Any]:
        """Optymalizacja hiperparametrów"""
        
        from sklearn.model_selection import GridSearchCV
        
        available_models = self.get_available_models()
        
        if model_name not in available_models:
            raise ValueError(f"Model '{model_name}' nie jest dostępny")
        
        model_info = available_models[model_name]
        base_model = model_info['model'](**model_info['params'])
        
        # Grid Search
        grid_search = GridSearchCV(
            base_model,
            param_grid,
            cv=cv_folds,
            scoring=scoring,
            n_jobs=-1,
            verbose=1
        )
        
        start_time = time.time()
        grid_search.fit(X_train, y_train)
        tuning_time = time.time() - start_time
        
        # Zapisz najlepszy model
        best_model_name = f"{model_name}_tuned"
        self.models[best_model_name] = grid_search.best_estimator_
        
        # Zapisz historię
        tuning_info = {
            'model': grid_search.best_estimator_,
            'model_name': best_model_name,
            'best_params': grid_search.best_params_,
            'best_score': grid_search.best_score_,
            'cv_results': grid_search.cv_results_,
            'tuning_time': tuning_time,
            'param_grid': param_grid
        }
        
        self.training_history[best_model_name] = tuning_info
        
        # Zaktualizuj najlepszy model
        if grid_search.best_score_ > self.best_score:
            self.best_score = grid_search.best_score_
            self.best_model = grid_search.best_estimator_
        
        logging.info(f"Hiperparametry zoptymalizowane dla {model_name}. Najlepszy score: {grid_search.best_score_:.4f}")
        
        return tuning_info
    
    def get_model_comparison(self) -> pd.DataFrame:
        """Zwraca porównanie modeli"""
        
        comparison_data = []
        
        for model_name, history in self.training_history.items():
            if 'error' in history:
                continue
                
            row = {
                'model_name': model_name,
                'train_accuracy': history.get('train_accuracy', 0),
                'val_accuracy': history.get('val_accuracy', history.get('train_accuracy', 0)),
                'training_time': history.get('training_time', 0),
                'params': str(history.get('params', {}))
            }
            
            # Dodaj F1-score jeśli dostępny
            if 'train_classification_report' in history:
                train_report = history['train_classification_report']
                if 'weighted avg' in train_report:
                    row['train_f1'] = train_report['weighted avg']['f1-score']
            
            if 'val_classification_report' in history:
                val_report = history['val_classification_report']
                if 'weighted avg' in val_report:
                    row['val_f1'] = val_report['weighted avg']['f1-score']
            
            comparison_data.append(row)
        
        comparison_df = pd.DataFrame(comparison_data)
        
        if not comparison_df.empty:
            comparison_df = comparison_df.sort_values('val_accuracy', ascending=False)
        
        return comparison_df
    
    def save_model(self, model_name: str, filepath: str) -> None:
        """Zapisuje model do pliku"""
        
        if model_name not in self.models:
            raise ValueError(f"Model '{model_name}' nie został wytrenowany")
        
        joblib.dump(self.models[model_name], filepath)
        logging.info(f"Model {model_name} zapisany w {filepath}")
    
    def load_model(self, filepath: str, model_name: str) -> None:
        """Ładuje model z pliku"""
        
        model = joblib.load(filepath)
        self.models[model_name] = model
        logging.info(f"Model {model_name} załadowany z {filepath}")
    
    def predict(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        model_name: Optional[str] = None,
        return_proba: bool = False
    ) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """Wykonuje predykcje"""
        
        if model_name is None:
            if self.best_model is None:
                raise ValueError("Brak wytrenowanego modelu")
            model = self.best_model
        else:
            if model_name not in self.models:
                raise ValueError(f"Model '{model_name}' nie został wytrenowany")
            model = self.models[model_name]
        
        predictions = model.predict(X)
        
        if return_proba:
            if hasattr(model, 'predict_proba'):
                probabilities = model.predict_proba(X)
                return predictions, probabilities
            else:
                logging.warning("Model nie wspiera predict_proba")
                return predictions, None
        
        return predictions
    
    def get_feature_importance(self, model_name: str) -> Optional[np.ndarray]:
        """Zwraca ważność cech dla modelu"""
        
        if model_name not in self.models:
            raise ValueError(f"Model '{model_name}' nie został wytrenowany")
        
        model = self.models[model_name]
        
        if hasattr(model, 'feature_importances_'):
            return model.feature_importances_
        elif hasattr(model, 'coef_'):
            return np.abs(model.coef_).flatten()
        else:
            logging.warning(f"Model {model_name} nie wspiera ważności cech")
            return None


if __name__ == "__main__":
    # Przykład użycia
    from sklearn.datasets import make_classification
    from sklearn.model_selection import train_test_split
    
    # Wygeneruj dane
    X, y = make_classification(n_samples=1000, n_classes=3, n_features=10, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    trainer = ModelTrainer()
    
    # Trenuj wiele modeli
    models_to_train = ['random_forest', 'logistic_regression', 'naive_bayes']
    results = trainer.train_multiple_models(X_train, y_train, models_to_train, X_test, y_test)
    
    # Porównanie modeli
    comparison = trainer.get_model_comparison()
    print("Porównanie modeli:")
    print(comparison[['model_name', 'train_accuracy', 'val_accuracy', 'training_time']])
    
    # Hiperparametryzacja
    param_grid = {
        'n_estimators': [50, 100, 200],
        'max_depth': [None, 10, 20]
    }
    
    tuning_result = trainer.hyperparameter_tuning(X_train, y_train, 'random_forest', param_grid)
    print(f"\nNajlepsze parametry: {tuning_result['best_params']}")
    
    # Predykcje
    predictions = trainer.predict(X_test)
    print(f"\nDokładność na zbiorze testowym: {accuracy_score(y_test, predictions):.4f}")
