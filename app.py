#%%
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# Chargement du dataset
dataset = load_dataset("galsenai/centralized_wolof_french_translation_data")



# Chargement du tokenizer et du modèle NLLB
tokenizer = AutoTokenizer.from_pretrained("facebook/nllb-200-distilled-600M")
model = AutoModelForSeq2SeqLM.from_pretrained("facebook/nllb-200-distilled-600M")

#%%

# Récupérer les codes de langue via la configuration du modèle

FR_CODE = tokenizer.convert_tokens_to_ids("fr_Latn")
WO_CODE = tokenizer.convert_tokens_to_ids("wol_Latn")

# Configuration du tokenizer pour indiquer la langue source
tokenizer.src_lang = "fr_Latn"

# Forcer la langue cible lors de la génération en définissant forced_bos_token_id
model.config.forced_bos_token_id = WO_CODE