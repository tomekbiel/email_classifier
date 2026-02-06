import re
import pandas as pd
from typing import List, Dict, Any
import logging


class TextPreprocessor:
    """
    Step 4: Regex, noise removal (stop-words)
    """
    
    def __init__(self):
        self.noise_patterns = self._initialize_noise_patterns()
        
    def _initialize_noise_patterns(self) -> Dict[str, List[str]]:
        """Initialize noise patterns"""
        return {
            'email_headers': [
                r"(from :)|(subject :)|(sent :)|(r\s*:)|(re\s*:)"
            ],
            'months': [
                "(january|february|march|april|may|june|july|august|september|october|november|december)",
                "(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)"
            ],
            'days': [
                "(monday|tuesday|wednesday|thursday|friday|saturday|sunday)"
            ],
            'times': [
                r"\d{2}(:|.)\d{2}"
            ],
            'emails': [
                r"(xxxxx@xxxx\.com)|(\*{5}\([a-z]+\))"
            ],
            'greetings': [
                "dear ((customer)|(user))",
                "dear",
                "(hello)|(hallo)|(hi )|(hi there)",
                "good morning"
            ],
            'thanks': [
                "thank you for your patience ((during (our)? investigation)|(and cooperation))?",
                "thank you for contacting us",
                "thank you for your availability",
                "thank you for providing us this information",
                "thank you for contacting",
                "thank you for reaching us (back)?",
                "thank you for patience",
                "thank you for (your)? reply",
                "thank you for (your)? response",
                "thank you for (your)? cooperation",
                "thank you for providing us with more information",
                "thank you very kindly",
                "thank you( very much)?"
            ],
            'follow_up': [
                "i would like to follow up on the case you raised on the date",
                "i will do my very best to assist you",
                "in order to give you the best solution",
                "could you please clarify your request with following information:",
                "in this matter",
                "we hope you(( are)|('re)) doing ((fine)|(well))",
                "i would like to follow up on the case you raised on"
            ],
            'company_info': [
                "we apologize for the inconvenience",
                "sent from my huawei (cell )?phone",
                "original message",
                "customer support team",
                "(aspiegel )?se is a company incorporated under the laws of ireland with its headquarters in dublin, ireland.",
                "(aspiegel )?se is the provider of huawei mobile services to huawei and honor device owners in",
                "canada, australia, new zealand and other countries"
            ],
            'numbers': [
                r"\d+"
            ],
            'special_chars': [
                "[^0-9a-zA-Z]+"
            ],
            'single_chars': [
                r"(\s|^).(\s|$)"
            ],
            'ticket_noise': [
                r"(sv\s*:)|(wg\s*:)|(ynt\s*:)|(fw(d)?\s*:)|(r\s*:)|(re\s*:)|(\[|\])|(aspiegel support issue submit)|(null)|(nan)|((bonus place my )?support.pt 自动回复:)"
            ]
        }
    
    def clean_text(self, text: str, lowercase: bool = True) -> str:
        """Czyści pojedynczy tekst"""
        if not isinstance(text, str) or text == "":
            return ""
            
        if lowercase:
            text = text.lower()
        
        # Usuń szum ticketowy
        for pattern in self.noise_patterns['ticket_noise']:
            text = re.sub(pattern, " ", text)
        
        # Usuń dodatkowe białe znaki
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def clean_ticket_summary(self, text: str) -> str:
        """Czyści podsumowanie ticketu"""
        return self.clean_text(text)
    
    def clean_interaction_content(self, text: str) -> str:
        """Czyści treść interakcji"""
        if not isinstance(text, str) or text == "":
            return ""
            
        text = text.lower()
        
        # Zastosuj wszystkie wzorce szumu
        for category, patterns in self.noise_patterns.items():
            if category == 'ticket_noise':
                continue  # Już zastosowane w clean_text
                
            for pattern in patterns:
                text = re.sub(pattern, " ", text)
                logging.debug(f"Zastosowano wzorzec {category}: {pattern}")
        
        # Usuń dodatkowe białe znaki
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def preprocess_dataframe(
        self, 
        df: pd.DataFrame, 
        summary_col: str = "Ticket Summary",
        interaction_col: str = "Interaction content",
        clean_summary_col: str = "ts",
        clean_interaction_col: str = "ic"
    ) -> pd.DataFrame:
        """Process DataFrame with text cleaning"""
        
        logging.info(f"Starting data preprocessing...")
        
        # Clean ticket summary
        logging.info(f"Cleaning column: {summary_col}")
        df[clean_summary_col] = df[summary_col].apply(self.clean_ticket_summary)
        
        # Clean interaction content
        logging.info(f"Cleaning column: {interaction_col}")
        df[clean_interaction_col] = df[interaction_col].apply(self.clean_interaction_content)
        
        # Remove rows with empty texts after cleaning
        initial_shape = df.shape[0]
        df = df[(df[clean_summary_col] != "") & (df[clean_interaction_col] != "")]
        final_shape = df.shape[0]
        
        logging.info(f"Removed {initial_shape - final_shape} rows with empty texts")
        logging.info(f"Preprocessing completed. Data shape: {df.shape}")
        
        return df
    
    def add_custom_pattern(self, category: str, pattern: str) -> None:
        """Add custom noise pattern"""
        if category not in self.noise_patterns:
            self.noise_patterns[category] = []
        self.noise_patterns[category].append(pattern)
        logging.info(f"Added pattern '{pattern}' to category '{category}'")
    
    def get_text_statistics(self, df: pd.DataFrame, text_column: str) -> Dict[str, Any]:
        """Returns text statistics"""
        if text_column not in df.columns:
            raise ValueError(f"Column '{text_column}' does not exist in DataFrame")
        
        texts = df[text_column].dropna()
        
        stats = {
            'total_texts': len(texts),
            'empty_texts': (texts == "").sum(),
            'avg_length': texts.str.len().mean(),
            'min_length': texts.str.len().min(),
            'max_length': texts.str.len().max(),
            'unique_texts': texts.nunique()
        }
        
        return stats


if __name__ == "__main__":
    # Przykład użycia
    import pandas as pd
    
    # Przykładowe dane
    data = {
        'Ticket Summary': ['RE: Issue with app', 'FW: Customer complaint', ''],
        'Interaction content': ['Dear customer, thank you for contacting us. We will help you with your issue.', 'Hello, I have a problem with my application.', '']
    }
    df = pd.DataFrame(data)
    
    preprocessor = TextPreprocessor()
    df_clean = preprocessor.preprocess_dataframe(df)
    
    print("Oryginalne dane:")
    print(data)
    print("\nPrzetworzone dane:")
    print(df_clean[['ts', 'ic']].head())
    
    # Statystyki
    stats = preprocessor.get_text_statistics(df_clean, 'ic')
    print(f"\nStatystyki tekstu: {stats}")
