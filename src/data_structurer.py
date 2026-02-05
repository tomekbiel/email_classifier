import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from sklearn.preprocessing import LabelEncoder
import logging


class DataStructurer:
    """
    Krok 5: Obsługa danych multi-level / multi-class
    """
    
    def __init__(self):
        self.label_encoders = {}
        self.class_mappings = {}
        
    def analyze_class_distribution(self, df: pd.DataFrame, target_columns: List[str]) -> Dict[str, Dict]:
        """Analizuje dystrybucję klas dla wielu kolumn docelowych"""
        distributions = {}
        
        for col in target_columns:
            if col in df.columns:
                value_counts = df[col].value_counts()
                distributions[col] = {
                    'counts': value_counts.to_dict(),
                    'percentages': (value_counts / len(df) * 100).round(2).to_dict(),
                    'num_classes': len(value_counts),
                    'class_balance': self._calculate_balance_score(value_counts)
                }
                
                logging.info(f"Kolumna {col}: {len(value_counts)} klas, balans: {distributions[col]['class_balance']:.3f}")
        
        return distributions
    
    def _calculate_balance_score(self, value_counts: pd.Series) -> float:
        """Oblicza wynik balansu klas (0 = idealnie zbalansowany, 1 = ekstremalnie niezbalansowany)"""
        if len(value_counts) <= 1:
            return 0.0
            
        # Normalizacja counts
        normalized = value_counts / value_counts.sum()
        
        # Oblicz odchylenie standardowe od idealnej dystrybucji
        ideal_distribution = 1 / len(value_counts)
        variance = ((normalized - ideal_distribution) ** 2).sum()
        
        return variance
    
    def create_hierarchical_labels(
        self, 
        df: pd.DataFrame, 
        primary_col: str, 
        secondary_col: Optional[str] = None,
        tertiary_col: Optional[str] = None
    ) -> pd.DataFrame:
        """Tworzy hierarchiczne etykiety"""
        result_df = df.copy()
        
        # Podstawowa etykieta
        if primary_col in df.columns:
            result_df['primary_label'] = df[primary_col]
        
        # Drugorzędna etykieta
        if secondary_col and secondary_col in df.columns:
            result_df['secondary_label'] = df[secondary_col]
            # Połącz etykiety
            result_df['hierarchical_label'] = (
                result_df['primary_label'].astype(str) + '_' + 
                result_df['secondary_label'].astype(str)
            )
        
        # Trzeciorzędna etykieta
        if tertiary_col and tertiary_col in df.columns:
            result_df['tertiary_label'] = df[tertiary_col]
            if 'hierarchical_label' in result_df.columns:
                result_df['hierarchical_label'] = (
                    result_df['hierarchical_label'].astype(str) + '_' + 
                    result_df['tertiary_label'].astype(str)
                )
        
        return result_df
    
    def encode_labels(
        self, 
        df: pd.DataFrame, 
        label_columns: List[str],
        fit_transform: bool = True
    ) -> pd.DataFrame:
        """Koduje etykiety kategoryczne"""
        result_df = df.copy()
        
        for col in label_columns:
            if col not in df.columns:
                logging.warning(f"Kolumna '{col}' nie istnieje w DataFrame")
                continue
                
            if col not in self.label_encoders:
                self.label_encoders[col] = LabelEncoder()
            
            encoder = self.label_encoders[col]
            
            if fit_transform:
                # Dopasuj i transformuj
                encoded_labels = encoder.fit_transform(df[col].astype(str))
            else:
                # Tylko transformuj (dla danych testowych)
                encoded_labels = encoder.transform(df[col].astype(str))
            
            # Zapisz mapowanie
            self.class_mappings[col] = dict(zip(encoder.classes_, encoder.transform(encoder.classes_)))
            
            # Dodaj zakodowaną kolumnę
            result_df[f'{col}_encoded'] = encoded_labels
            
            logging.info(f"Zakodowano {col}: {len(encoder.classes_)} klas")
        
        return result_df
    
    def create_multilabel_structure(
        self, 
        df: pd.DataFrame, 
        label_columns: List[str]
    ) -> Tuple[np.ndarray, List[str]]:
        """Tworzy strukturę multi-label"""
        valid_columns = [col for col in label_columns if col in df.columns]
        
        if not valid_columns:
            raise ValueError("Brak prawidłowych kolumn etykiet")
        
        # Połącz wszystkie etykiety w jedną macierz
        label_matrix = []
        for col in valid_columns:
            encoded = self.label_encoders.get(col, LabelEncoder()).fit_transform(df[col].astype(str))
            label_matrix.append(encoded)
        
        label_matrix = np.array(label_matrix).T
        
        return label_matrix, valid_columns
    
    def filter_rare_classes(
        self, 
        df: pd.DataFrame, 
        label_column: str, 
        min_samples: int = 10
    ) -> pd.DataFrame:
        """Filtruje rzadkie klasy"""
        if label_column not in df.columns:
            raise ValueError(f"Kolumna '{label_column}' nie istnieje")
        
        value_counts = df[label_column].value_counts()
        common_classes = value_counts[value_counts >= min_samples].index
        
        initial_count = len(df)
        filtered_df = df[df[label_column].isin(common_classes)]
        final_count = len(filtered_df)
        
        logging.info(f"Usunięto {initial_count - final_count} wierszy z rzadkimi klasami")
        logging.info(f"Pozostało {len(common_classes)} klas z min {min_samples} próbek")
        
        return filtered_df
    
    def create_stratified_groups(
        self, 
        df: pd.DataFrame, 
        label_column: str,
        group_size: int = 50
    ) -> pd.DataFrame:
        """Tworzy zgrupowane dane dla stratyfikacji"""
        if label_column not in df.columns:
            raise ValueError(f"Kolumna '{label_column}' nie istnieje")
        
        # Dodaj grupę dla każdej klasy
        df_with_groups = df.copy()
        df_with_groups['strata_group'] = -1
        
        current_group = 0
        for class_label in df[label_column].unique():
            class_mask = df[label_column] == class_label
            class_indices = df[class_mask].index
            
            # Podziel na grupy
            for i in range(0, len(class_indices), group_size):
                group_indices = class_indices[i:i+group_size]
                df_with_groups.loc[group_indices, 'strata_group'] = current_group
                current_group += 1
        
        logging.info(f"Utworzono {current_group} grup stratyfikacyjnych")
        
        return df_with_groups
    
    def get_data_summary(self, df: pd.DataFrame, label_columns: List[str]) -> Dict[str, Any]:
        """Zwraca podsumowanie struktury danych"""
        summary = {
            'total_samples': len(df),
            'features': [col for col in df.columns if col not in label_columns],
            'labels': label_columns,
            'label_distributions': self.analyze_class_distribution(df, label_columns),
            'missing_values': df.isnull().sum().to_dict(),
            'data_types': df.dtypes.to_dict()
        }
        
        return summary


if __name__ == "__main__":
    # Przykład użycia
    import pandas as pd
    
    # Przykładowe dane
    data = {
        'text': ['problem with app', 'login issue', 'payment error', 'bug report', 'feature request'],
        'primary_type': ['technical', 'technical', 'billing', 'technical', 'feature'],
        'secondary_type': ['app_crash', 'auth', 'payment', 'ui_bug', 'enhancement'],
        'priority': ['high', 'medium', 'high', 'low', 'medium']
    }
    df = pd.DataFrame(data)
    
    structurer = DataStructurer()
    
    # Analiza dystrybucji
    distributions = structurer.analyze_class_distribution(df, ['primary_type', 'secondary_type', 'priority'])
    print("Dystrybucja klas:")
    for col, dist in distributions.items():
        print(f"{col}: {dist['num_classes']} klas")
    
    # Kodowanie etykiet
    df_encoded = structurer.encode_labels(df, ['primary_type', 'secondary_type', 'priority'])
    print("\nZakodowane dane:")
    print(df_encoded[['primary_type_encoded', 'secondary_type_encoded', 'priority_encoded']].head())
    
    # Podsumowanie
    summary = structurer.get_data_summary(df, ['primary_type', 'secondary_type', 'priority'])
    print(f"\nPodsumowanie: {summary['total_samples']} próbek")
