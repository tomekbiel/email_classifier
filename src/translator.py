import stanza
from stanza.pipeline.core import DownloadMethod
from transformers import M2M100ForConditionalGeneration, M2M100Tokenizer
from typing import List, Optional
import logging


class Translator:
    """
    Step 3: Translation of NLP text to English
    """
    
    def __init__(self, model_name: str = "facebook/m2m100_418M"):
        self.model_name = model_name
        self.model = None
        self.tokenizer = None
        self.nlp_stanza = None
        self._initialized = False
        
    def initialize(self) -> None:
        """Initialize translation models"""
        if self._initialized:
            return
            
        try:
            logging.info("Initializing translation models...")
            
            # M2M100 model and tokenizer
            self.model = M2M100ForConditionalGeneration.from_pretrained(self.model_name)
            self.tokenizer = M2M100Tokenizer.from_pretrained(self.model_name)
            
            # Stanza for language detection
            self.nlp_stanza = stanza.Pipeline(
                lang="multilingual", 
                processors="langid",
                download_method=DownloadMethod.REUSE_RESOURCES
            )
            
            self._initialized = True
            logging.info("Translation models initialized successfully")
            
        except Exception as e:
            logging.error(f"Error during model initialization: {e}")
            raise
    
    def _map_language_code(self, lang_code: str) -> str:
        """Map language codes to standard M2M100 codes"""
        lang_mapping = {
            "fro": "fr",  # Old French
            "la": "it",   # Latin
            "nn": "no",   # Norwegian (Nynorsk)
            "kmr": "tr",  # Kurmanji
        }
        return lang_mapping.get(lang_code, lang_code)
    
    def translate_text(self, text: str) -> str:
        """Translate single text to English"""
        if not self._initialized:
            self.initialize()
            
        if text == "" or text is None:
            return text
            
        try:
            # Language detection
            doc = self.nlp_stanza(text)
            detected_lang = doc.lang
            
            if detected_lang == "en":
                return text
            
            # Map language code
            source_lang = self._map_language_code(detected_lang)
            
            # Translation
            self.tokenizer.src_lang = source_lang
            encoded_text = self.tokenizer(text, return_tensors="pt")
            generated_tokens = self.model.generate(
                **encoded_text, 
                forced_bos_token_id=self.tokenizer.get_lang_id("en")
            )
            translated_text = self.tokenizer.batch_decode(
                generated_tokens, 
                skip_special_tokens=True
            )[0]
            
            return translated_text
            
        except Exception as e:
            logging.warning(f"Error during text translation: {e}")
            return text  # Return original text on error
    
    def translate_batch(self, texts: List[str], batch_size: int = 10) -> List[str]:
        """Translate batch of texts"""
        if not self._initialized:
            self.initialize()
            
        translated_texts = []
        
        for i, text in enumerate(texts):
            if i % batch_size == 0:
                logging.info(f"Translated {i}/{len(texts)} texts")
            
            translated = self.translate_text(text)
            translated_texts.append(translated)
            
            # Debug info for first few texts
            if i < 5 and text != translated:
                logging.info(f"Original: {text}")
                logging.info(f"Translation: {translated}")
        
        return translated_texts
    
    def translate_dataframe_column(
        self, 
        df, 
        column_name: str, 
        new_column_name: Optional[str] = None
    ):
        """Translate DataFrame column"""
        # Input validation
        if df is None or df.empty:
            logging.warning("DataFrame is empty or None")
            return df
            
        if column_name not in df.columns:
            logging.error(f"Column '{column_name}' does not exist in DataFrame")
            raise ValueError(f"Column '{column_name}' not found")
            
        if new_column_name is None:
            new_column_name = f"{column_name}_en"
            
        logging.info(f"Translating column '{column_name}' to English...")
        
        texts = df[column_name].tolist()
        translated_texts = self.translate_batch(texts)
        
        df[new_column_name] = translated_texts
        
        logging.info(f"Translation completed. New column: '{new_column_name}'")
        return df


if __name__ == "__main__":
    # Example usage
    import pandas as pd
    
    # Sample data
    data = {
        'text': ['Hello world', 'Bonjour le monde', 'Hola mundo', '']
    }
    df = pd.DataFrame(data)
    
    translator = Translator()
    df_translated = translator.translate_dataframe_column(df, 'text')
    
    print(df_translated)
