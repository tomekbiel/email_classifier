import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Union
from sklearn.utils import resample
from imblearn.over_sampling import SMOTE, ADASYN, RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler, NearMiss, TomekLinks
from imblearn.combine import SMOTETomek, SMOTEENN
import logging


class Sampler:
    """
    Step 7: Data balancing (Oversampling/Undersampling)
    """
    
    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        self.sampling_methods = {}
        
    def analyze_imbalance(self, y: Union[pd.Series, np.ndarray]) -> Dict[str, Any]:
        """Analyze class imbalance"""
        
        if isinstance(y, np.ndarray):
            y = pd.Series(y)
        
        value_counts = y.value_counts()
        total_samples = len(y)
        
        # Calculate imbalance metrics
        imbalance_ratio = value_counts.max() / value_counts.min()
        minority_class = value_counts.idxmin()
        majority_class = value_counts.idxmax()
        
        # Percentages
        percentages = (value_counts / total_samples * 100).round(2)
        
        analysis = {
            'class_counts': value_counts.to_dict(),
            'class_percentages': percentages.to_dict(),
            'imbalance_ratio': imbalance_ratio,
            'minority_class': minority_class,
            'majority_class': majority_class,
            'total_samples': total_samples,
            'num_classes': len(value_counts),
            'is_balanced': imbalance_ratio <= 1.5  # Consider balanced if ratio <= 1.5
        }
        
        logging.info(f"Imbalance analysis: ratio={imbalance_ratio:.2f}, classes={len(value_counts)}")
        
        return analysis
    
    def random_oversample(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        y: Union[pd.Series, np.ndarray],
        sampling_strategy: Union[str, float, Dict] = 'auto'
    ) -> Tuple[Union[pd.DataFrame, np.ndarray], Union[pd.Series, np.ndarray]]:
        """Random oversampling"""
        
        sampler = RandomOverSampler(
            sampling_strategy=sampling_strategy,
            random_state=self.random_state
        )
        
        X_resampled, y_resampled = sampler.fit_resample(X, y)
        
        logging.info(f"Random oversampling: {len(X)} -> {len(X_resampled)} samples")
        
        return X_resampled, y_resampled
    
    def random_undersample(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        y: Union[pd.Series, np.ndarray],
        sampling_strategy: Union[str, float, Dict] = 'auto'
    ) -> Tuple[Union[pd.DataFrame, np.ndarray], Union[pd.Series, np.ndarray]]:
        """Random undersampling"""
        
        sampler = RandomUnderSampler(
            sampling_strategy=sampling_strategy,
            random_state=self.random_state
        )
        
        X_resampled, y_resampled = sampler.fit_resample(X, y)
        
        logging.info(f"Random undersampling: {len(X)} -> {len(X_resampled)} samples")
        
        return X_resampled, y_resampled
    
    def smote_oversample(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        y: Union[pd.Series, np.ndarray],
        sampling_strategy: Union[str, float, Dict] = 'auto',
        k_neighbors: int = 5
    ) -> Tuple[Union[pd.DataFrame, np.ndarray], Union[pd.Series, np.ndarray]]:
        """SMOTE oversampling"""
        
        sampler = SMOTE(
            sampling_strategy=sampling_strategy,
            random_state=self.random_state,
            k_neighbors=k_neighbors
        )
        
        X_resampled, y_resampled = sampler.fit_resample(X, y)
        
        logging.info(f"SMOTE oversampling: {len(X)} -> {len(X_resampled)} próbek")
        
        return X_resampled, y_resampled
    
    def adasyn_oversample(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        y: Union[pd.Series, np.ndarray],
        sampling_strategy: Union[str, float, Dict] = 'auto',
        n_neighbors: int = 5
    ) -> Tuple[Union[pd.DataFrame, np.ndarray], Union[pd.Series, np.ndarray]]:
        """ADASYN oversampling"""
        
        sampler = ADASYN(
            sampling_strategy=sampling_strategy,
            random_state=self.random_state,
            n_neighbors=n_neighbors
        )
        
        X_resampled, y_resampled = sampler.fit_resample(X, y)
        
        logging.info(f"ADASYN oversampling: {len(X)} -> {len(X_resampled)} samples")
        
        return X_resampled, y_resampled
    
    def nearmiss_undersample(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        y: Union[pd.Series, np.ndarray],
        sampling_strategy: Union[str, float, Dict] = 'auto',
        version: int = 1
    ) -> Tuple[Union[pd.DataFrame, np.ndarray], Union[pd.Series, np.ndarray]]:
        """NearMiss undersampling"""
        
        sampler = NearMiss(
            sampling_strategy=sampling_strategy,
            version=version,
            n_jobs=-1
        )
        
        X_resampled, y_resampled = sampler.fit_resample(X, y)
        
        logging.info(f"NearMiss undersampling (v{version}): {len(X)} -> {len(X_resampled)} samples")
        
        return X_resampled, y_resampled
    
    def tomek_links_undersample(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        y: Union[pd.Series, np.ndarray]
    ) -> Tuple[Union[pd.DataFrame, np.ndarray], Union[pd.Series, np.ndarray]]:
        """Tomek Links undersampling"""
        
        sampler = TomekLinks()
        
        X_resampled, y_resampled = sampler.fit_resample(X, y)
        
        logging.info(f"Tomek Links undersampling: {len(X)} -> {len(X_resampled)} samples")
        
        return X_resampled, y_resampled
    
    def smote_tomek_combine(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        y: Union[pd.Series, np.ndarray],
        sampling_strategy: Union[str, float, Dict] = 'auto'
    ) -> Tuple[Union[pd.DataFrame, np.ndarray], Union[pd.Series, np.ndarray]]:
        """SMOTE + Tomek Links combined"""
        
        sampler = SMOTETomek(
            sampling_strategy=sampling_strategy,
            random_state=self.random_state
        )
        
        X_resampled, y_resampled = sampler.fit_resample(X, y)
        
        logging.info(f"SMOTE+Tomek: {len(X)} -> {len(X_resampled)} samples")
        
        return X_resampled, y_resampled
    
    def smote_enn_combine(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        y: Union[pd.Series, np.ndarray],
        sampling_strategy: Union[str, float, Dict] = 'auto'
    ) -> Tuple[Union[pd.DataFrame, np.ndarray], Union[pd.Series, np.ndarray]]:
        """SMOTE + ENN combined"""
        
        sampler = SMOTEENN(
            sampling_strategy=sampling_strategy,
            random_state=self.random_state
        )
        
        X_resampled, y_resampled = sampler.fit_resample(X, y)
        
        logging.info(f"SMOTE+ENN: {len(X)} -> {len(X_resampled)} samples")
        
        return X_resampled, y_resampled
    
    def custom_balanced_sampling(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        y: Union[pd.Series, np.ndarray],
        target_samples_per_class: Optional[int] = None,
        method: str = 'oversample'
    ) -> Tuple[Union[pd.DataFrame, np.ndarray], Union[pd.Series, np.ndarray]]:
        """Niestandardowe zbalansowane próbkowanie"""
        
        if isinstance(X, pd.DataFrame):
            X_df = X.copy()
            X_df['target'] = y
        else:
            X_df = pd.DataFrame(X)
            X_df['target'] = y
        
        # Oblicz docelową liczbę próbek na klasę
        class_counts = X_df['target'].value_counts()
        
        if target_samples_per_class is None:
            target_samples_per_class = class_counts.max()
        
        balanced_dfs = []
        
        for class_label in class_counts.index:
            class_df = X_df[X_df['target'] == class_label]
            current_count = len(class_df)
            
            if current_count < target_samples_per_class:
                # Oversampling
                if method == 'oversample':
                    resampled_df = resample(
                        class_df,
                        replace=True,
                        n_samples=target_samples_per_class,
                        random_state=self.random_state
                    )
                else:
                    resampled_df = class_df  # Do nothing if not oversample
            elif current_count > target_samples_per_class:
                # Undersampling
                resampled_df = resample(
                    class_df,
                    replace=False,
                    n_samples=target_samples_per_class,
                    random_state=self.random_state
                )
            else:
                resampled_df = class_df
            
            balanced_dfs.append(resampled_df)
        
        # Combine all balanced classes
        balanced_df = pd.concat(balanced_dfs, ignore_index=True)
        
        # Split back to X and y
        y_resampled = balanced_df['target']
        X_resampled = balanced_df.drop('target', axis=1)
        
        logging.info(f"Custom balanced sampling: {len(X)} -> {len(X_resampled)} samples")
        
        return X_resampled, y_resampled
    
    def auto_balance(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        y: Union[pd.Series, np.ndarray],
        max_ratio: float = 2.0,
        prefer_oversampling: bool = True
    ) -> Tuple[Union[pd.DataFrame, np.ndarray], Union[pd.Series, np.ndarray]]:
        """Automatic balancing based on analysis"""
        
        analysis = self.analyze_imbalance(y)
        
        if analysis['is_balanced']:
            logging.info("Data is already balanced")
            return X, y
        
        if analysis['num_classes'] == 2:
            # Binary classification - use RandomOverSampler instead of SMOTE
            if prefer_oversampling:
                return self.random_oversample(X, y)
            else:
                return self.random_undersample(X, y)
        else:
            # Multi-class - use RandomOverSampler instead of SMOTE
            if prefer_oversampling:
                return self.random_oversample(X, y)
            else:
                return self.nearmiss_undersample(X, y)
    
    def compare_sampling_methods(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        y: Union[pd.Series, np.ndarray],
        methods: List[str] = None
    ) -> Dict[str, Dict[str, Any]]:
        """Porównuje różne metody próbkowania"""
        
        if methods is None:
            methods = ['original', 'random_oversample', 'random_undersample', 'smote', 'adasyn']
        
        results = {}
        
        for method in methods:
            if method == 'original':
                X_resampled, y_resampled = X, y
            elif method == 'random_oversample':
                X_resampled, y_resampled = self.random_oversample(X, y)
            elif method == 'random_undersample':
                X_resampled, y_resampled = self.random_undersample(X, y)
            elif method == 'smote':
                X_resampled, y_resampled = self.smote_oversample(X, y)
            elif method == 'adasyn':
                X_resampled, y_resampled = self.adasyn_oversample(X, y)
            else:
                logging.warning(f"Nieznana metoda: {method}")
                continue
            
            analysis = self.analyze_imbalance(y_resampled)
            results[method] = {
                'samples': len(X_resampled),
                'features': X_resampled.shape[1] if hasattr(X_resampled, 'shape') else len(X_resampled[0]),
                'imbalance_ratio': analysis['imbalance_ratio'],
                'is_balanced': analysis['is_balanced']
            }
        
        return results


if __name__ == "__main__":
    # Przykład użycia
    import numpy as np
    from sklearn.datasets import make_classification
    
    # Wygeneruj niezbalansowane dane
    X, y = make_classification(
        n_samples=1000,
        n_classes=3,
        weights=[0.7, 0.2, 0.1],
        random_state=42
    )
    
    sampler = Sampler()
    
    # Analiza niezbalansowania
    analysis = sampler.analyze_imbalance(y)
    print(f"Analiza początkowa: {analysis}")
    
    # Różne metody próbkowania
    X_over, y_over = sampler.smote_oversample(X, y)
    X_under, y_under = sampler.random_undersample(X, y)
    
    print(f"SMOTE: {len(X)} -> {len(X_over)}")
    print(f"Random undersample: {len(X)} -> {len(X_under)}")
    
    # Porównanie metod
    comparison = sampler.compare_sampling_methods(X, y)
    print("\nPorównanie metod:")
    for method, result in comparison.items():
        print(f"{method}: {result['samples']} próbek, ratio={result['imbalance_ratio']:.2f}")
