import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Union
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix, roc_auc_score,
    roc_curve, precision_recall_curve, average_precision_score
)
import matplotlib.pyplot as plt
import seaborn as sns
import logging
from sklearn.preprocessing import label_binarize


class ModelEvaluator:
    """
    Ewaluacja modeli klasyfikacji
    """
    
    def __init__(self):
        self.evaluation_results = {}
        
    def evaluate_classification(
        self,
        y_true: Union[pd.Series, np.ndarray],
        y_pred: Union[pd.Series, np.ndarray],
        y_proba: Optional[Union[pd.DataFrame, np.ndarray]] = None,
        class_names: Optional[List[str]] = None,
        model_name: str = "model"
    ) -> Dict[str, Any]:
        """Kompleksowa ewaluacja klasyfikacji"""
        
        # Podstawowe metryki
        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, average='weighted', zero_division=0)
        recall = recall_score(y_true, y_pred, average='weighted', zero_division=0)
        f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)
        
        # Raport klasyfikacji
        class_report = classification_report(y_true, y_pred, target_names=class_names, output_dict=True)
        
        # Macierz pomyłek
        cm = confusion_matrix(y_true, y_pred)
        
        # Wyniki
        results = {
            'model_name': model_name,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'classification_report': class_report,
            'confusion_matrix': cm,
            'class_names': class_names or [f'Class_{i}' for i in range(len(np.unique(y_true)))]
        }
        
        # Metryki dla danych prawdopodobieństwa
        if y_proba is not None:
            proba_results = self._evaluate_probabilities(y_true, y_pred, y_proba, class_names)
            results.update(proba_results)
        
        # Analiza per klasa
        per_class_results = self._analyze_per_class_performance(y_true, y_pred, class_names)
        results['per_class_analysis'] = per_class_results
        
        # Zapisz wyniki
        self.evaluation_results[model_name] = results
        
        logging.info(f"Ewaluacja modelu {model_name}: accuracy={accuracy:.4f}, f1={f1:.4f}")
        
        return results
    
    def _evaluate_probabilities(
        self,
        y_true: Union[pd.Series, np.ndarray],
        y_pred: Union[pd.Series, np.ndarray],
        y_proba: Union[pd.DataFrame, np.ndarray],
        class_names: Optional[List[str]]
    ) -> Dict[str, Any]:
        """Ewaluacja metryk opartych na prawdopodobieństwach"""
        
        results = {}
        
        # Konwertuj na numpy array
        if isinstance(y_proba, pd.DataFrame):
            y_proba = y_proba.values
        
        n_classes = len(np.unique(y_true))
        
        if n_classes == 2:
            # Binary classification
            if y_proba.shape[1] == 2:
                y_score = y_proba[:, 1]  # Prawdopodobieństwo klasy pozytywnej
            else:
                y_score = y_proba[:, 0]
            
            try:
                auc_roc = roc_auc_score(y_true, y_score)
                results['auc_roc'] = auc_roc
                
                # Precision-Recall AUC
                precision, recall, _ = precision_recall_curve(y_true, y_score)
                auc_pr = average_precision_score(y_true, y_score)
                results['auc_pr'] = auc_pr
                results['precision_recall_curve'] = {'precision': precision, 'recall': recall}
                
            except Exception as e:
                logging.warning(f"Błąd podczas obliczania AUC: {e}")
        
        else:
            # Multi-class classification
            try:
                # Binarizuj etykiety
                y_true_bin = label_binarize(y_true, classes=range(n_classes))
                
                if y_proba.shape[1] == n_classes:
                    auc_roc = roc_auc_score(y_true_bin, y_proba, multi_class='ovr', average='weighted')
                    results['auc_roc'] = auc_roc
                
            except Exception as e:
                logging.warning(f"Błąd podczas obliczania multi-class AUC: {e}")
        
        return results
    
    def _analyze_per_class_performance(
        self,
        y_true: Union[pd.Series, np.ndarray],
        y_pred: Union[pd.Series, np.ndarray],
        class_names: Optional[List[str]]
    ) -> Dict[str, Dict]:
        """Analiza wydajności per klasa"""
        
        unique_classes = np.unique(y_true)
        if class_names is None:
            class_names = [f'Class_{i}' for i in unique_classes]
        
        per_class = {}
        
        for i, class_label in enumerate(unique_classes):
            class_name = class_names[i] if i < len(class_names) else f'Class_{class_label}'
            
            # True Positive, False Positive, True Negative, False Negative
            tp = np.sum((y_true == class_label) & (y_pred == class_label))
            fp = np.sum((y_true != class_label) & (y_pred == class_label))
            tn = np.sum((y_true != class_label) & (y_pred != class_label))
            fn = np.sum((y_true == class_label) & (y_pred != class_label))
            
            # Metryki per klasa
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
            support = np.sum(y_true == class_label)
            
            per_class[class_name] = {
                'precision': precision,
                'recall': recall,
                'f1_score': f1,
                'support': support,
                'true_positives': tp,
                'false_positives': fp,
                'true_negatives': tn,
                'false_negatives': fn
            }
        
        return per_class
    
    def compare_models(self, model_names: Optional[List[str]] = None) -> pd.DataFrame:
        """Porównuje wyniki wielu modeli"""
        
        if model_names is None:
            model_names = list(self.evaluation_results.keys())
        
        comparison_data = []
        
        for model_name in model_names:
            if model_name not in self.evaluation_results:
                logging.warning(f"Brak wyników dla modelu {model_name}")
                continue
            
            results = self.evaluation_results[model_name]
            
            row = {
                'model': model_name,
                'accuracy': results.get('accuracy', 0),
                'precision': results.get('precision', 0),
                'recall': results.get('recall', 0),
                'f1_score': results.get('f1_score', 0),
                'auc_roc': results.get('auc_roc', 0)
            }
            
            comparison_data.append(row)
        
        comparison_df = pd.DataFrame(comparison_data)
        
        if not comparison_df.empty:
            comparison_df = comparison_df.sort_values('f1_score', ascending=False)
        
        return comparison_df
    
    def plot_confusion_matrix(
        self,
        model_name: str,
        figsize: Tuple[int, int] = (8, 6),
        normalize: bool = False
    ) -> plt.Figure:
        """Rysuje macierz pomyłek"""
        
        if model_name not in self.evaluation_results:
            raise ValueError(f"Brak wyników dla modelu {model_name}")
        
        results = self.evaluation_results[model_name]
        cm = results['confusion_matrix']
        class_names = results['class_names']
        
        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            title = f'Normalized Confusion Matrix - {model_name}'
        else:
            title = f'Confusion Matrix - {model_name}'
        
        fig, ax = plt.subplots(figsize=figsize)
        sns.heatmap(
            cm,
            annot=True,
            fmt='.2f' if normalize else 'd',
            cmap='Blues',
            xticklabels=class_names,
            yticklabels=class_names,
            ax=ax
        )
        
        ax.set_title(title)
        ax.set_xlabel('Predicted Label')
        ax.set_ylabel('True Label')
        
        plt.tight_layout()
        return fig
    
    def plot_class_performance(
        self,
        model_name: str,
        figsize: Tuple[int, int] = (12, 6)
    ) -> plt.Figure:
        """Rysuje wykres wydajności per klasa"""
        
        if model_name not in self.evaluation_results:
            raise ValueError(f"Brak wyników dla modelu {model_name}")
        
        results = self.evaluation_results[model_name]
        per_class = results['per_class_analysis']
        
        classes = list(per_class.keys())
        precision = [per_class[cls]['precision'] for cls in classes]
        recall = [per_class[cls]['recall'] for cls in classes]
        f1 = [per_class[cls]['f1_score'] for cls in classes]
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
        
        # Wykres metryk
        x = np.arange(len(classes))
        width = 0.25
        
        ax1.bar(x - width, precision, width, label='Precision', alpha=0.8)
        ax1.bar(x, recall, width, label='Recall', alpha=0.8)
        ax1.bar(x + width, f1, width, label='F1-Score', alpha=0.8)
        
        ax1.set_xlabel('Classes')
        ax1.set_ylabel('Score')
        ax1.set_title(f'Per-Class Performance - {model_name}')
        ax1.set_xticks(x)
        ax1.set_xticklabels(classes, rotation=45)
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Wykres support
        support = [per_class[cls]['support'] for cls in classes]
        
        ax2.bar(classes, support, alpha=0.8, color='skyblue')
        ax2.set_xlabel('Classes')
        ax2.set_ylabel('Support (Number of Samples)')
        ax2.set_title(f'Class Support - {model_name}')
        ax2.tick_params(axis='x', rotation=45)
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig
    
    def plot_model_comparison(self, figsize: Tuple[int, int] = (12, 8)) -> plt.Figure:
        """Rysuje wykres porównawczy modeli"""
        
        comparison_df = self.compare_models()
        
        if comparison_df.empty:
            raise ValueError("Brak wyników do porównania")
        
        metrics = ['accuracy', 'precision', 'recall', 'f1_score', 'auc_roc']
        available_metrics = [m for m in metrics if m in comparison_df.columns]
        
        fig, axes = plt.subplots(2, 3, figsize=figsize)
        axes = axes.flatten()
        
        for i, metric in enumerate(available_metrics):
            if i < len(axes):
                ax = axes[i]
                ax.bar(comparison_df['model'], comparison_df[metric], alpha=0.8)
                ax.set_title(f'{metric.upper()}')
                ax.tick_params(axis='x', rotation=45)
                ax.grid(True, alpha=0.3)
                
                # Dodaj wartości na słupkach
                for j, v in enumerate(comparison_df[metric]):
                    ax.text(j, v + 0.01, f'{v:.3f}', ha='center', va='bottom')
        
        # Ukryj nieużywane osie
        for i in range(len(available_metrics), len(axes)):
            axes[i].set_visible(False)
        
        plt.suptitle('Model Comparison')
        plt.tight_layout()
        return fig
    
    def generate_evaluation_report(
        self,
        model_name: str,
        save_path: Optional[str] = None
    ) -> str:
        """Generuje szczegółowy raport ewaluacji"""
        
        if model_name not in self.evaluation_results:
            raise ValueError(f"Brak wyników dla modelu {model_name}")
        
        results = self.evaluation_results[model_name]
        
        report = f"""
# Ewaluacja Modelu: {model_name}

## Podsumowanie Metryk
- **Dokładność (Accuracy)**: {results['accuracy']:.4f}
- **Precyzja (Precision)**: {results['precision']:.4f}
- **Czułość (Recall)**: {results['recall']:.4f}
- **F1-Score**: {results['f1_score']:.4f}
"""
        
        if 'auc_roc' in results:
            report += f"- **AUC-ROC**: {results['auc_roc']:.4f}\n"
        
        if 'auc_pr' in results:
            report += f"- **AUC-PR**: {results['auc_pr']:.4f}\n"
        
        report += f"""
## Analiza Per-Klasa
"""
        
        per_class = results['per_class_analysis']
        for class_name, metrics in per_class.items():
            report += f"""
### {class_name}
- Precyzja: {metrics['precision']:.4f}
- Czułość: {metrics['recall']:.4f}
- F1-Score: {metrics['f1_score']:.4f}
- Support: {metrics['support']}
- TP: {metrics['true_positives']}, FP: {metrics['false_positives']}
- TN: {metrics['true_negatives']}, FN: {metrics['false_negatives']}
"""
        
        report += f"""
## Macierz Pomyłek
{results['confusion_matrix']}
"""
        
        if save_path:
            with open(save_path, 'w', encoding='utf-8') as f:
                f.write(report)
            logging.info(f"Raport zapisany w {save_path}")
        
        return report
    
    def get_best_model(self, metric: str = 'f1_score') -> Tuple[str, float]:
        """Zwraca najlepszy model według wskazanej metryki"""
        
        if not self.evaluation_results:
            raise ValueError("Brak wyników ewaluacji")
        
        best_model = None
        best_score = 0
        
        for model_name, results in self.evaluation_results.items():
            score = results.get(metric, 0)
            if score > best_score:
                best_score = score
                best_model = model_name
        
        if best_model is None:
            raise ValueError(f"Brak wyników dla metryki {metric}")
        
        return best_model, best_score


if __name__ == "__main__":
    # Przykład użycia
    from sklearn.datasets import make_classification
    from sklearn.model_selection import train_test_split
    from sklearn.ensemble import RandomForestClassifier
    
    # Wygeneruj dane
    X, y = make_classification(n_samples=1000, n_classes=3, n_features=10, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Trenuj model
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    # Predykcje
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)
    
    # Ewaluacja
    evaluator = ModelEvaluator()
    results = evaluator.evaluate_classification(
        y_test, y_pred, y_proba, 
        class_names=['Class_0', 'Class_1', 'Class_2'],
        model_name="random_forest"
    )
    
    print("Wyniki ewaluacji:")
    print(f"Accuracy: {results['accuracy']:.4f}")
    print(f"F1-Score: {results['f1_score']:.4f}")
    
    # Generuj raport
    report = evaluator.generate_evaluation_report("random_forest")
    print(report[:500] + "...")  # Pierwsze 500 znaków raportu
