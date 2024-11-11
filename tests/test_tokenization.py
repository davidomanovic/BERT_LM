import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from models.bert_lm import BertLM
model = BertLM()

def tokenize_function(examples):
    
    tokenized = model.encode_input(examples['text'])
    print(tokenized)  # Print tokenized sample to debug
    return tokenized

example_text = "Hello, this is a test sentence used to demonstrate basic text processing techniques."
tokenized_example = model.encode_input([example_text])
print(tokenized_example)

from transformers import BertTokenizer, TFBertForMaskedLM

# Load the tokenizer and model
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = TFBertForMaskedLM.from_pretrained('bert-base-uncased')

# Example input text
text = "This is an example sentence."

# Tokenize the input text with padding/truncation
inputs = tokenizer(text, padding=True, truncation=True, max_length=128, return_tensors='tf')

# Print the input IDs, token type IDs, and attention mask
print("Input IDs:", inputs['input_ids'])
print("Token Type IDs:", inputs['token_type_ids'])
print("Attention Mask:", inputs['attention_mask'])