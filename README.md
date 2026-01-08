# Text Summarization (Hugging Face Transformers)

This notebook demonstrates **text summarization** using pre-trained Transformer models from Hugging Face as a part of **AitroniX** internship. [web:358]  
The core inference workflow is: **load model + tokenizer → tokenize → generate → decode**. [web:358]

## Models tested
- **PEGASUS**: `google/pegasus-cnn_dailymail` (commonly used for English news summarization). [web:36]  
- **BART**: `facebook/bart-large-cnn` (fine-tuned for summarization). [web:274]  
- **T5 / mT5**: T5 is text-to-text; multilingual variants are typically more suitable for Arabic than English-only checkpoints. [web:361]

## How it works (steps)
1. Download/load the tokenizer and model using `from_pretrained()` (files are cached locally the first time). [web:358]  
2. Tokenize the input text into tensors. [web:358]  
3. Generate the summary token ids using `model.generate()` with suitable decoding parameters (e.g., beam search). [web:358][web:359]  
4. Decode the generated ids back to text with `tokenizer.decode(..., skip_special_tokens=True)`. [web:358]

## Minimal inference code (PEGASUS/BART/T5)
```python
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

model_id = "facebook/bart-large-cnn"  # or "google/pegasus-cnn_dailymail"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForSeq2SeqLM.from_pretrained(model_id)

text = "Paste your input text here..."

inputs = tokenizer(text, return_tensors="pt", truncation=True)
summary_ids = model.generate(
    **inputs,
    max_new_tokens=128,
    num_beams=4,
    do_sample=False
)

summary = tokenizer.decode(summary_ids, skip_special_tokens=True)
print(summary)
