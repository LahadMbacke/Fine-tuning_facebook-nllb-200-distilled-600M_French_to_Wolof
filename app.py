#%%
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# Chargement du dataset
dataset = load_dataset("galsenai/centralized_wolof_french_translation_data")



# Chargement du tokenizer et du modèle NLLB
tokenizer = AutoTokenizer.from_pretrained("facebook/nllb-200-distilled-600M")
model = AutoModelForSeq2SeqLM.from_pretrained("facebook/nllb-200-distilled-600M")

#%%

