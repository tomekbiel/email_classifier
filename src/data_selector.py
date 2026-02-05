import numpy as np
import pandas as pd
from typing import Tuple, Optional
import os


class DataSelector:
    """
    Steps 1 & 2: Column cleaning and preliminary grouping
    """
    
    def __init__(self, file_path: str):
        self.file_path = file_path
        self.df = None
        
    def load_data(self) -> pd.DataFrame:
        """Load data from CSV file"""
        self.df = pd.read_csv(self.file_path)
        return self.df
    
    def clean_data_types(self) -> pd.DataFrame:
        """Convert dtype object to unicode string"""
        if self.df is None:
            raise ValueError("Data not loaded. Use load_data() first.")
            
        self.df['Interaction content'] = self.df['Interaction content'].values.astype('U')
        self.df['Ticket Summary'] = self.df['Ticket Summary'].values.astype('U')
        return self.df
    
    def rename_columns(self) -> pd.DataFrame:
        """Rename columns for easier memorization"""
        if self.df is None:
            raise ValueError("Data not loaded. Use load_data() first.")
            
        # Optionally: rename variables for easier memorization
        self.df["y1"] = self.df["Type 1"]
        self.df["y2"] = self.df["Type 2"]
        self.df["y3"] = self.df["Type 3"]
        self.df["y4"] = self.df["Type 4"]
        self.df["x"] = self.df['Interaction content']
        
        # Use Type 2 as main target variable
        self.df["y"] = self.df["y2"]
        
        return self.df
    
    def remove_empty_targets(self) -> pd.DataFrame:
        """Remove empty y values"""
        if self.df is None:
            raise ValueError("Data not loaded. Use load_data() first.")
            
        self.df = self.df.loc[(self.df["y"] != '') & (~self.df["y"].isna())]
        return self.df
    
    def filter_by_frequency(self, column: str = "y1", min_count: int = 10) -> pd.DataFrame:
        """Filter data based on frequency"""
        if self.df is None:
            raise ValueError("Data not loaded. Use load_data() first.")
            
        value_counts = self.df[column].value_counts()
        good_values = value_counts[value_counts > min_count].index
        self.df = self.df.loc[self.df[column].isin(good_values)]
        return self.df
    
    def process_data(self, filter_frequency: bool = True) -> Tuple[pd.DataFrame, dict]:
        """
        Complete data processing process
        
        Returns:
            Tuple[pd.DataFrame, dict]: Processed data and metadata
        """
        self.load_data()
        self.clean_data_types()
        self.rename_columns()
        self.remove_empty_targets()
        
        if filter_frequency:
            self.filter_by_frequency()
        
        metadata = {
            'shape': self.df.shape,
            'columns': list(self.df.columns),
            'target_distribution': self.df['y'].value_counts().to_dict(),
            'type1_distribution': self.df['y1'].value_counts().to_dict()
        }
        
        return self.df, metadata


if __name__ == "__main__":
    # Example usage
    data_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "AppGallery.csv")
    selector = DataSelector(data_path)
    df, metadata = selector.process_data()
    
    print(f"Data shape: {metadata['shape']}")
    print(f"Target distribution: {metadata['target_distribution']}")
