# NLLB-200 Fine-tuned pour la traduction Français-Wolof 🇫🇷↔️🇸🇳

[![Hugging Face Model](https://img.shields.io/badge/🤗%20Hugging%20Face-Model-yellow)](https://huggingface.co/Lahad/nllb200-francais-wolof)

Modèle NLLB-200 (version distillée 600M) fine-tuné pour la traduction Français → Wolof.

---

## 📥 Installation

```bash
pip install transformers datasets
```

## 🚀 Utilisation rapide

### Chargement du modèle

```python
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

tokenizer = AutoTokenizer.from_pretrained("Lahad/nllb200-francais-wolof")
model = AutoModelForSeq2SeqLM.from_pretrained("Lahad/nllb200-francais-wolof")
```

### Fonction de traduction

```python
def translate(text, max_length=128):
    inputs = tokenizer(
        text,
        max_length=max_length,
        padding="max_length",
        truncation=True,
        return_tensors="pt"
    )
    
    outputs = model.generate(
        input_ids=inputs["input_ids"],
        attention_mask=inputs["attention_mask"],
        forced_bos_token_id=tokenizer.convert_tokens_to_ids("wol_Latn"),
        max_length=max_length
    )
    
    return tokenizer.decode(outputs[0], skip_special_tokens=True)
```

### Exemples

```python
print(translate("Bonjour comment ça va ?"))  # Nanga def ?
print(translate("J’ai décidé de quitter mon boulot"))  # Dama jël dogal ni damay bàyyi sama liggéey
```

## 🧠 Configuration technique

### Architecture

* **Modèle de base:** `facebook/nllb-200-distilled-600M`
* **Paramètres:** 600M
* **Context Window:** 128 tokens
* **Langues:**
   * Source: `fr_Latn` (Français)
   * Cible: `wol_Latn` (Wolof)

### Hyperparamètres d'entraînement

```python
Seq2SeqTrainingArguments(
    output_dir="./results_nllb_finetuned",
    learning_rate=2e-5,
    per_device_train_batch_size=8,
    num_train_epochs=3,
    fp16=True,
    evaluation_strategy="epoch"
)
```

## 📊 Données

* **Dataset:** galsenai/centralized_wolof_french_translation_data
* **Split:** 80/20 (Train/Test)
* **Préprocessing:**
   * Tokenization avec padding dynamique
   * Longueur maximale: 128 tokens
   * Format:

```json
{"fr": "Texte français", "wo": "Traduction wolof"}
```

## 📈 Performances

| Métrique | Valeur |
|----------|--------|
| Loss final | 1.23 |
| BLEU Score | 42.1 |
| Temps d'entraînement | 5h (T4) |

## ⚠️ Limitations

1. Performances réduites sur textes techniques
2. Limité à 128 tokens par défaut

## 📜 Licence

* **Modèle:** CC-BY-NC-4.0
* **Données:** Licence originale du dataset

## 🙏 Crédits

* Meta AI pour NLLB-200
* GalsenAI pour les données
* Communauté Hugging Face 🤗


